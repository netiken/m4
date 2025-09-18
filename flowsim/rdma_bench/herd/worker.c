#include "hrd.h"
#include "main.h"
#include "mica.h"
#include <stdio.h>
#include <inttypes.h>

/* Deterministically derive value length from a 128-bit key (two 64-bit parts) */
static inline uint8_t herd_val_len_from_key_parts(uint64_t part0, uint64_t part1) {
  const uint8_t min_len = 8;
  const uint8_t max_len = MICA_MAX_VALUE; /* 46 bytes for 64B mica_op */
  const uint32_t range = (uint32_t)(max_len - min_len + 1);
  uint64_t mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL);
  return (uint8_t)(min_len + (mix % range));
}

void* run_worker(void* arg) {
  int i, ret;
  struct thread_params params = *(struct thread_params*)arg;
  int wrkr_lid = params.id; /* Local ID of this worker thread*/
  int num_server_ports = params.num_server_ports;
  int base_port_index = params.base_port_index;
  int postlist = params.postlist;
  printf("worker started\n");
  /* Open per-thread server log file */
  char slog_path[256];
  sprintf(slog_path, "/home/zabreyko/rdma_bench/herd/worker_log_%d.txt", wrkr_lid);
  FILE* slog_fp = fopen(slog_path, "w");
  assert(slog_fp != NULL);
  setvbuf(slog_fp, NULL, _IOLBF, 0);
  /*
   * MICA-related checks. Note that @postlist is the largest batch size we
   * feed into MICA. The average postlist per port in a dual-port NIC should
   * be @postlist / 2.
   */
  assert(MICA_MAX_BATCH_SIZE >= postlist);
  assert(HERD_VALUE_SIZE <= MICA_MAX_VALUE);

  assert(UNSIG_BATCH >= postlist); /* Postlist check */
  assert(postlist <= NUM_CLIENTS); /* Static sizing of arrays below */

  /* MICA instance id = wrkr_lid, NUMA node = 0 */
  struct mica_kv kv;
  mica_init(&kv, wrkr_lid, 0, HERD_NUM_BKTS, HERD_LOG_CAP);
  /* Populate with deterministically variable-length values per key */
  {
    uint128* key_arr = mica_gen_keys(HERD_NUM_KEYS);
    struct mica_op op;
    struct mica_resp resp;
    unsigned long long* op_key = (unsigned long long*)&op.key;
    op.opcode = MICA_OP_PUT;

    for (i = 0; i < HERD_NUM_KEYS; i++) {
      op_key[0] = key_arr[i].first;
      op_key[1] = key_arr[i].second;

      uint8_t vlen = herd_val_len_from_key_parts(op_key[0], op_key[1]);
      op.val_len = vlen;

      /* Fill value bytes deterministically for debuggability */
      uint8_t val = (uint8_t)(op_key[1] & 0xff);
      memset(op.value, val, vlen);

      mica_insert_one(&kv, &op, &resp);
    }
  }

  assert(num_server_ports < MAX_SERVER_PORTS); /* Avoid dynamic alloc */
  struct hrd_ctrl_blk* cb[MAX_SERVER_PORTS];

  for (i = 0; i < num_server_ports; i++) {
    int ib_port_index = base_port_index + i;

    cb[i] = hrd_ctrl_blk_init(wrkr_lid,          /* local_hid */
                              ib_port_index, -1, /* port index, numa node */
                              0, 0,              /* #conn qps, uc */
                              NULL, 0, -1, /*prealloc conn buf, buf size, key */
                              NUM_UD_QPS, 4096,
                              -1); /* num_dgram_qps, dgram_buf_size, key */
  }
  /* Map the request region created by the master */
  volatile struct mica_op* req_buf;
  int sid = shmget(MASTER_SHM_KEY, RR_SIZE, SHM_HUGETLB | 0666);
  assert(sid != -1);
  req_buf = shmat(sid, 0, 0);
  assert(req_buf != (void*)-1);

  /* Create an address handle for each client */
  struct ibv_ah* ah[NUM_CLIENTS];
  memset(ah, 0, NUM_CLIENTS * sizeof(uintptr_t));
  struct hrd_qp_attr* clt_qp[NUM_CLIENTS];

  for (i = 0; i < NUM_CLIENTS; i++) {
    /* Compute the control block and physical port index for client @i */
    int cb_i = i % num_server_ports;
    int local_port_i = base_port_index + cb_i;

    char clt_name[HRD_QP_NAME_SIZE];
    sprintf(clt_name, "client-dgram-%d", i);

    /* Get the UD queue pair for the ith client */
    clt_qp[i] = NULL;
    while (clt_qp[i] == NULL) {
      clt_qp[i] = hrd_get_published_qp(clt_name);
      if (clt_qp[i] == NULL) {
        usleep(200000);
      }
    }

    printf("main: Worker %d found client %d of %d clients. Client LID: %d\n",
           wrkr_lid, i, NUM_CLIENTS, clt_qp[i]->lid);

    /* Build an address handle for RoCE (uses Global Route Header) */
    struct ibv_ah_attr ah_attr = {0};
    ah_attr.is_global = 1; /* RoCE requires GRH */
    ah_attr.dlid       = 0; /* Unused for RoCE */
    ah_attr.sl         = 0;
    ah_attr.src_path_bits = 0;

    /* port_num (> 1): device-local port for responses to this client */
    ah_attr.port_num   = local_port_i + 1;

    /* Fill GRH fields */
    ah_attr.grh.dgid = clt_qp[i]->gid;
    {
      int gid_index = 0;
      char* env = getenv("HRD_ROCE_GID_INDEX");
      if (env != NULL) gid_index = atoi(env);
      ah_attr.grh.sgid_index = gid_index; /* Use configured GID index */
    }
    ah_attr.grh.hop_limit  = 1;

    ah[i] = ibv_create_ah(cb[cb_i]->pd, &ah_attr);
    assert(ah[i] != NULL);
  }
  printf("exiting client loop\n");

  int ws[NUM_CLIENTS] = {0}; /* Per-client window slot */

  /* We can detect at most NUM_CLIENTS requests in each step */
  struct mica_op* op_ptr_arr[NUM_CLIENTS];
  struct mica_resp resp_arr[NUM_CLIENTS];
  struct ibv_send_wr wr[NUM_CLIENTS], *bad_send_wr = NULL;
  struct ibv_sge sgl[NUM_CLIENTS];
  int resp_slot_for_wr[NUM_CLIENTS];
  int resp_clt_for_wr[NUM_CLIENTS];
  uint32_t resp_size_for_wr[NUM_CLIENTS];
  int cb_for_wr[NUM_CLIENTS];

  /* If postlist is diabled, remember the cb to send() each @wr from */
  /* cb_for_wr used for logging in both postlist modes */

  /* If postlist is enabled, we instead create per-cb linked lists of wr's */
  struct ibv_send_wr* first_send_wr[MAX_SERVER_PORTS] = {NULL};
  struct ibv_send_wr* last_send_wr[MAX_SERVER_PORTS] = {NULL};
  struct ibv_wc wc;
  long long rolling_iter = 0; /* For throughput measurement */
  long long nb_tx[MAX_SERVER_PORTS][NUM_UD_QPS] = {{0}}; /* CQE polling */
  int ud_qp_i = 0; /* UD QP index: same for both ports */
  long long nb_tx_tot = 0;
  long long nb_post_send = 0;

  /*
   * @cb_i is the control block to use for @clt_i's response. If NUM_CLIENTS
   * is a multiple of @num_server_ports, we can cycle @cb_i in the client loop
   * to maintain cb_i = clt_i % num_server_ports.
   *
   * @wr_i is the work request to use for @clt_i's response. We use contiguous
   * work requests for the responses in a batch. This is because (in HERD) we
   * need to  pass a contiguous array of operations to MICA, and marshalling
   * and unmarshalling the contiguous array will be expensive.
   */
  int cb_i = -1, clt_i = -1;
  int poll_i, wr_i;
  assert(NUM_CLIENTS % num_server_ports == 0);

  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);
  printf("starting the main loop\n");
  while (1) {
    if (unlikely(rolling_iter >= M_4)) {
      clock_gettime(CLOCK_REALTIME, &end);
      double seconds = (end.tv_sec - start.tv_sec) +
                       (double)(end.tv_nsec - start.tv_nsec) / 1000000000;
      printf(
          "main: Worker %d: %.2f IOPS. Avg per-port postlist = %.2f. "
          "HERD lookup fail rate = %.4f\n",
          wrkr_lid, M_4 / seconds, (double)nb_tx_tot / nb_post_send,
          (double)kv.num_get_fail / kv.num_get_op);

      rolling_iter = 0;
      nb_tx_tot = 0;
      nb_post_send = 0;

      clock_gettime(CLOCK_REALTIME, &start);
    }

    /* Do a pass over requests from all clients */
    int nb_new_req[MAX_SERVER_PORTS] = {0}; /* New requests from port i*/
    wr_i = 0;

    /*
     * Request region prefetching needs to be done w/o messing up @clt_i,
     * so the loop below is wrong.
     * Recomputing @req_offset in the loop below is as expensive as storing
     * results in an extra array.
     */
    /*for(clt_i = 0; clt_i < NUM_CLIENTS; clt_i++) {
            int req_offset = OFFSET(wrkr_lid, clt_i, ws[clt_i]);
            __builtin_prefetch((void *) &req_buf[req_offset], 0, 2);
    }*/

    for (poll_i = 0; poll_i < NUM_CLIENTS; poll_i++) {
      /*
       * This cycling of @clt_i and @cb_i needs to be before polling. This
       * should be the only place where we modify @clt_i and @cb_i.
       */
      HRD_MOD_ADD(clt_i, NUM_CLIENTS);
      HRD_MOD_ADD(cb_i, num_server_ports);
      // assert(cb_i == clt_i % num_server_ports);	/* XXX */

      int req_offset = OFFSET(wrkr_lid, clt_i, ws[clt_i]);
      if (req_buf[req_offset].opcode < HERD_OP_GET) {
        continue;
      }

      /* Log request receive event (server detects request) */
      {
        uint8_t herd_opcode = req_buf[req_offset].opcode;
        uint32_t req_size = (herd_opcode == HERD_OP_GET)
                                ? HERD_GET_REQ_SIZE
                                : (16 + 1 + 1 + req_buf[req_offset].val_len);
        int local_slot = ws[clt_i];
        struct timespec ts_now;
        clock_gettime(CLOCK_REALTIME, &ts_now);
        uint64_t ts_ns = (uint64_t)ts_now.tv_sec * 1000000000ULL + (uint64_t)ts_now.tv_nsec;
        fprintf(slog_fp,
                "event=req_recv ts_ns=%" PRIu64 " clt=%d wrkr=%d slot=%d size=%u src=client:%d dst=worker:%d\n",
                ts_ns, clt_i, wrkr_lid, local_slot, req_size, clt_i, wrkr_lid);
      }

      /* Convert to a MICA opcode */
      req_buf[req_offset].opcode -= HERD_MICA_OFFSET;
      // assert(req_buf[req_offset].opcode == MICA_OP_GET ||	/* XXX */
      //		req_buf[req_offset].opcode == MICA_OP_PUT);

      op_ptr_arr[wr_i] = (struct mica_op*)&req_buf[req_offset];

      if (USE_POSTLIST == 1) {
        /* Add the SEND response for this client to the postlist */
        if (nb_new_req[cb_i] == 0) {
          first_send_wr[cb_i] = &wr[wr_i];
          last_send_wr[cb_i] = &wr[wr_i];
        } else {
          last_send_wr[cb_i]->next = &wr[wr_i];
          last_send_wr[cb_i] = &wr[wr_i];
        }
        cb_for_wr[wr_i] = cb_i;
      } else {
        wr[wr_i].next = NULL;
        cb_for_wr[wr_i] = cb_i;
      }

      /* Fill in the work request */
      wr[wr_i].wr.ud.ah = ah[clt_i];
      wr[wr_i].wr.ud.remote_qpn = clt_qp[clt_i]->qpn;
      wr[wr_i].wr.ud.remote_qkey = HRD_DEFAULT_QKEY;

      wr[wr_i].opcode = IBV_WR_SEND_WITH_IMM;
      wr[wr_i].num_sge = 1;
      wr[wr_i].sg_list = &sgl[wr_i];
      /* Encode worker ID (high 16 bits) and window slot (low 16 bits) */
      int local_slot = ws[clt_i];
      uint32_t imm = ((uint32_t)wrkr_lid << 16) | ((uint32_t)local_slot & 0xFFFF);
      wr[wr_i].imm_data = imm;
      resp_slot_for_wr[wr_i] = local_slot;
      resp_clt_for_wr[wr_i] = clt_i;

      wr[wr_i].send_flags =
          ((nb_tx[cb_i][ud_qp_i] & UNSIG_BATCH_) == 0) ? IBV_SEND_SIGNALED : 0;
      if ((nb_tx[cb_i][ud_qp_i] & UNSIG_BATCH_) == UNSIG_BATCH_) {
        hrd_poll_cq(cb[cb_i]->dgram_send_cq[ud_qp_i], 1, &wc);
      }
      wr[wr_i].send_flags |= IBV_SEND_INLINE;

      HRD_MOD_ADD(ws[clt_i], WINDOW_SIZE);

      rolling_iter++;
      nb_tx[cb_i][ud_qp_i]++; /* Must increment inside loop */
      nb_tx_tot++;
      nb_new_req[cb_i]++;
      wr_i++;

      if (wr_i == postlist) {
        break;
      }
    }

    mica_batch_op(&kv, wr_i, op_ptr_arr, resp_arr);

    /*
     * Fill in the computed @val_ptr's. For non-postlist mode, this loop
     * must go from 0 to (@wr_i - 1) to follow the signaling logic.
     */
    int nb_new_req_tot = wr_i;
    for (wr_i = 0; wr_i < nb_new_req_tot; wr_i++) {
      sgl[wr_i].length = resp_arr[wr_i].val_len;
      sgl[wr_i].addr = (uint64_t)(uintptr_t)resp_arr[wr_i].val_ptr;
      resp_size_for_wr[wr_i] = sgl[wr_i].length;

      if (USE_POSTLIST == 0) {
        ret = ibv_post_send(cb[cb_for_wr[wr_i]]->dgram_qp[ud_qp_i], &wr[wr_i],
                            &bad_send_wr);
        CPE(ret, "ibv_post_send error", ret);
        nb_post_send++;
        struct timespec ts_now;
        clock_gettime(CLOCK_REALTIME, &ts_now);
        uint64_t ts_ns = (uint64_t)ts_now.tv_sec * 1000000000ULL + (uint64_t)ts_now.tv_nsec;
        fprintf(slog_fp,
                "event=resp_send ts_ns=%" PRIu64 " clt=%d wrkr=%d slot=%d size=%u src=worker:%d dst=client:%d\n",
                ts_ns, resp_clt_for_wr[wr_i], wrkr_lid, resp_slot_for_wr[wr_i],
                resp_size_for_wr[wr_i], wrkr_lid, resp_clt_for_wr[wr_i]);
      }
    }

    for (i = 0; i < num_server_ports; i++) {
      if (nb_new_req[i] == 0) {
        continue;
      }

      /* If postlist is off, we should post replies in the loop above */
      if (USE_POSTLIST == 1) {
        last_send_wr[i]->next = NULL;
        ret = ibv_post_send(cb[i]->dgram_qp[ud_qp_i], first_send_wr[i],
                            &bad_send_wr);
        CPE(ret, "ibv_post_send error", ret);
        nb_post_send++;
        /* Log response sends for this postlist on cb i */
        struct timespec ts_now;
        clock_gettime(CLOCK_REALTIME, &ts_now);
        uint64_t ts_ns = (uint64_t)ts_now.tv_sec * 1000000000ULL + (uint64_t)ts_now.tv_nsec;
        for (int wr_j = 0; wr_j < nb_new_req_tot; wr_j++) {
          if (cb_for_wr[wr_j] == i) {
            fprintf(slog_fp,
                    "event=resp_send ts_ns=%" PRIu64 " clt=%d wrkr=%d slot=%d size=%u src=worker:%d dst=client:%d\n",
                    ts_ns, resp_clt_for_wr[wr_j], wrkr_lid, resp_slot_for_wr[wr_j],
                    resp_size_for_wr[wr_j], wrkr_lid, resp_clt_for_wr[wr_j]);
          }
        }
      }
    }

    /* Use a different UD QP for the next postlist */
    HRD_MOD_ADD(ud_qp_i, NUM_UD_QPS);
  }

  return NULL;
}
