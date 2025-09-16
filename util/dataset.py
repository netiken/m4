from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import json
import logging
import os
from .consts import (
    balance_len_bins,
    get_base_delay_transmission,
    get_base_delay_path,
)


def collate_fn(batch):
    (
        inputs,
        outputs,
        specs,
        remainsize_matrix,
        queuelen_matrix,
        queuelen_link_matrix,
        flow_active_matrix,
        time_delta_matrix,
        edge_index_matrix,
        n_links_list,
    ) = zip(*batch)

    # Get lengths of each sequence in the batch
    n_flows = np.array([x.shape[0] for x in inputs]).astype(np.int64)
    batch_size = len(n_flows)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    n_flows_accu = np.cumsum(n_flows)
    n_links_accu = np.cumsum(n_links_list)

    num_events = np.array([x.shape[0] for x in time_delta_matrix])
    max_len = max(num_events)
    padded_time_delta_matrix = np.zeros(
        (batch_size, max_len, 1),
        dtype=np.float32,
    )
    for i, time_delta_list in enumerate(time_delta_matrix):
        padded_time_delta_matrix[i, : time_delta_list.shape[0], 0] = time_delta_list

    if queuelen_matrix[0] is not None:
        padded_queuelen_matrix = []
        padded_queuelen_link_matrix = []
        for i in range(max_len):
            tmp = []
            tmp_link = []
            for j in range(batch_size):
                for k in range(flow_active_matrix[j].shape[0]):
                    if (
                        flow_active_matrix[j][k, 0] == i
                        and (queuelen_matrix[j][k]).any()
                    ):
                        tmp.append(queuelen_matrix[j][k])
                        link_idx = queuelen_link_matrix[j][k]
                        if j > 0:
                            link_idx += n_links_accu[j - 1]
                        tmp_link.append(link_idx)
            if len(tmp) != 0:
                tmp = torch.tensor(np.concatenate(tmp))
                tmp_link = torch.tensor(np.concatenate(tmp_link))
            padded_queuelen_matrix.append(tmp)
            padded_queuelen_link_matrix.append(tmp_link)
    else:
        padded_queuelen_matrix = queuelen_matrix
        padded_queuelen_link_matrix = queuelen_link_matrix

    flow_active_matrix = np.concatenate(flow_active_matrix)

    batch_index = np.zeros(n_flows_accu[-1], dtype=np.int32)
    batch_index_link = np.zeros(n_links_accu[-1], dtype=np.int32)
    for i in range(1, batch_size):
        batch_index[n_flows_accu[i - 1] : n_flows_accu[i]] = i
        batch_index_link[n_links_accu[i - 1] : n_links_accu[i]] = i
        edge_index_matrix[i][0] += n_flows_accu[i - 1]
        edge_index_matrix[i][1] += n_links_accu[i - 1]
    edge_index_matrix = np.concatenate(edge_index_matrix, axis=1)

    if remainsize_matrix[0] is not None:
        padded_remainsize_matrix = []
        for i in range(max_len):
            tmp = []
            for j in range(batch_size):
                if i < len(remainsize_matrix[j]):
                    tmp.append(remainsize_matrix[j][i])
            tmp = torch.tensor(np.concatenate(tmp))
            padded_remainsize_matrix.append(tmp)
    else:
        padded_remainsize_matrix = remainsize_matrix

    return (
        torch.tensor(inputs),
        torch.tensor(outputs),
        torch.tensor(batch_index),
        torch.tensor(batch_index_link),
        specs,
        padded_remainsize_matrix,
        padded_queuelen_matrix,
        padded_queuelen_link_matrix,
        torch.tensor(flow_active_matrix),
        torch.tensor(padded_time_delta_matrix),
        torch.tensor(edge_index_matrix),
    )


class DataModulePerFlow(LightningDataModule):
    def __init__(
        self,
        dir_input,
        shard_list,
        n_flows_list,
        n_hosts_list,
        batch_size,
        num_workers,
        train_frac,
        dir_output,
        lr,
        topo_type="",
        mode="train",
        enable_segmentation=False,
        enable_positional_encoding=False,
        flow_size_threshold=100000,
        enable_flowsim_gt=False,
        enable_remainsize=False,
        enable_queuelen=False,
        sampling_method="uniform",  # uniform, weighted, balanced
        enable_topo=True,
        test_on_train=False,
        test_on_empirical=False,
        test_on_manual=False,
    ) -> None:
        """
        Initializes a new instance of the class with the specified parameters.

        Args:
            positive_ratio (float, optional): The ratio of positive to negative samples to use for training.
                Defaults to 0.8.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.lr = lr
        self.topo_type = topo_type
        self.enable_segmentation = enable_segmentation
        self.sampling_method = sampling_method
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_flowsim_gt = enable_flowsim_gt
        self.enable_remainsize = enable_remainsize
        self.enable_queuelen = enable_queuelen
        self.enable_topo = enable_topo
        logging.info(
            f"call DataModulePerFlow: lr={lr}, topo_type={topo_type}, enable_segmentation={enable_segmentation}, sampling_method={sampling_method}, enable_topo={enable_topo}"
        )
        data_list = []
        if mode == "train":
            if enable_segmentation:
                len_per_period_all = []
                len_per_period_active_all = []
                len_per_period_stats_all = []
            # for shard in shard_list:
            #     for n_flows in n_flows_list:
            #         for n_hosts in n_hosts_list:
            for spec in os.listdir(dir_input):
                n_hosts = 12
                if float(spec.split("_")[0])>1000:
                    continue
                spec+=f"/ns3"
                if enable_topo:
                    topo_type_cur = topo_type
                    # spec = f"{shard}/ns3"
                    file_suffix = ""
                    fid = np.load(
                        f"{dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy"
                    )
                    if (
                        len(fid) == len(set(fid))
                        and np.all(fid[:-1] <= fid[1:])
                        # and len(fid) % n_flows == 0
                        # and os.path.exists(
                        #     f"{dir_input}/{spec}/flowsim_fct.npy"
                        # )
                    ):
                        busy_periods = np.load(
                            f"{dir_input}/{spec}/period{topo_type_cur}{file_suffix}_t{flow_size_threshold}.npy",
                            allow_pickle=True,
                        )

                        len_per_period_stats = [
                            len(period) for period in busy_periods
                        ]

                        remainsize_path = f"{dir_input}/{spec}/period_remainsize_num{topo_type_cur}{file_suffix}_t{flow_size_threshold}.npy"

                        if os.path.exists(remainsize_path):
                            len_per_period_active = np.load(remainsize_path)
                        else:
                            len_per_period_active = len_per_period_stats

                        len_per_period = len_per_period_stats

                        len_per_period = [
                            (
                                len_per_period[i]
                                # if len_per_period_active[i] < 150
                                if len_per_period_active[i] < 1500
                                else 0
                            )
                            for i in range(len(len_per_period))
                        ]

                        if np.sum(len_per_period) > 0:
                            data_list_per_period = [
                                (
                                    spec,
                                    (0, n_hosts - 1),
                                    topo_type_cur + file_suffix,
                                    int(segment_id),
                                    len_per_period_stats[segment_id],
                                )
                                for segment_id in range(len(busy_periods))
                            ]
                            sample_indices = np.arange(len(len_per_period))

                            len_per_period_all.extend(
                                [len_per_period[i] for i in sample_indices]
                            )
                            len_per_period_stats_all.extend(
                                [
                                    len_per_period_stats[i]
                                    for i in sample_indices
                                ]
                            )
                            len_per_period_active_all.extend(
                                [
                                    len_per_period_active[i]
                                    for i in sample_indices
                                ]
                            )
                            data_list.extend(
                                [
                                    data_list_per_period[i]
                                    for i in sample_indices
                                ]
                            )

            if enable_segmentation:
                len_per_period_all = np.array(len_per_period_all)
                n_samples = (
                    len(shard_list)
                    * len(n_flows_list)
                    * len(n_hosts_list)
                )
                # Sample indices from the array based on the weights
                if sampling_method == "uniform":
                    weights = len_per_period_all > 0
                elif sampling_method == "weighted":
                    weights = len_per_period_all
                elif sampling_method == "balanced":
                    # Bin the lengths
                    binned_lengths = np.digitize(len_per_period_all, balance_len_bins)

                    # Create a dictionary to count the number of periods for each length
                    unique_lengths, counts = np.unique(
                        binned_lengths, return_counts=True
                    )
                    logging.info(
                        f"# of unique_lengths: {len(unique_lengths)}, {unique_lengths}, # of counts: {counts}"
                    )
                    # Assign equal weight to each length category
                    length_weights = 1.0 / unique_lengths.size
                    # Calculate the weight for each period
                    weights = np.zeros(len(binned_lengths))
                    for length, count in zip(unique_lengths, counts):
                        if length == unique_lengths[0]:
                            continue
                        period_indices = np.where(binned_lengths == length)[0]
                        weights[period_indices] = length_weights / count
                else:
                    raise ValueError(f"Unsupported sampling method: {sampling_method}")

                weights = weights / np.sum(weights)
                sample_indices = np.random.choice(
                    len(weights), min(max(n_samples,600), len(weights)), replace=False, p=weights
                )

                data_list = [data_list[i] for i in sample_indices]

                n_mean = np.mean([len_per_period_active_all[i] for i in sample_indices])
                n_max = np.max([len_per_period_active_all[i] for i in sample_indices])
                logging.info(
                    f"# of active flows per busy period: mean-{n_mean}, max-{n_max}"
                )

                n_mean = np.mean([len_per_period_stats_all[i] for i in sample_indices])
                n_max = np.max([len_per_period_stats_all[i] for i in sample_indices])
                logging.info(f"# of flows per busy period: mean-{n_mean}, max-{n_max}")

            np.random.shuffle(data_list)
        self.data_list = data_list
        self.test_on_train = test_on_train
        self.test_on_empirical = test_on_empirical
        self.sampling_method = sampling_method
        self.test_on_manual = test_on_manual

    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders.

        Args:
            stage (str): The current stage of the training process. Either "fit" or "test".

        Returns:
            None
        """
        if stage == "fit":
            self.train_list, self.val_list = self.__random_split_list(
                self.data_list,
                self.train_frac,
            )
            num_train, num_val = (
                len(self.train_list),
                len(self.val_list),
            )
            logging.info(f"#tracks: train-{num_train}, val-{num_val}")
            self.train = self.__create_dataset(
                self.train_list,
            )
            self.val = self.__create_dataset(
                self.val_list,
            )

            self.__dump_data_list(self.dir_output)

        if stage == "test":
            if self.test_on_manual or self.test_on_empirical:
                data_list_test = []
                if self.enable_topo:
                    if self.test_on_manual:
                        shard_list = np.arange(0, 1000)

                    elif self.test_on_empirical:
                        shard_list = np.arange(0, 100)
                    n_hosts_list = [32]
                    n_flows_list = [2000]

                if self.enable_segmentation:
                    len_per_period_all = []
                    len_per_period_stats_all = []
                    len_per_period_active_all = []

                for shard in shard_list:
                    for n_flows in n_flows_list:
                        for n_hosts in n_hosts_list:
                            if self.enable_topo:
                                topo_type_cur = self.topo_type
                                spec = f"{shard}/ns3"
                                # statss = np.load(f'{self.dir_input}/{spec}/stats.npy', allow_pickle=True)
                                # if float(statss.item().get("load_bottleneck_target")) > 0.8: continue

                                # file_suffix = f"s{sample}_i0"
                                file_suffix = ""
                                fid = np.load(
                                    f"{self.dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy"
                                )
                                if (
                                    len(fid) == len(set(fid))
                                    and np.all(fid[:-1] <= fid[1:])
                                    and len(fid) % n_flows == 0
                                    and os.path.exists(
                                        f"{self.dir_input}/{spec}/flowsim_fct.npy"
                                    )
                                ):
                                    busy_periods = np.load(
                                        f"{self.dir_input}/{spec}/period{topo_type_cur}{file_suffix}_t{self.flow_size_threshold}.npy",
                                        allow_pickle=True,
                                    )

                                    len_per_period_stats = [
                                        len(period) for period in busy_periods
                                    ]

                                    remainsize_path = f"{self.dir_input}/{spec}/period_remainsize_num{topo_type_cur}{file_suffix}_t{self.flow_size_threshold}.npy"

                                    if os.path.exists(remainsize_path):
                                        len_per_period_active = np.load(
                                            remainsize_path
                                        )
                                    else:
                                        len_per_period_active = len_per_period_stats

                                    len_per_period = len_per_period_stats

                                    if np.sum(len_per_period) > 0:
                                        data_list_per_period = [
                                            (
                                                spec,
                                                (0, n_hosts - 1),
                                                topo_type_cur + file_suffix,
                                                int(segment_id),
                                                len_per_period_stats[segment_id],
                                            )
                                            for segment_id in range(
                                                len(busy_periods)
                                            )
                                        ]
                                        sample_indices = np.arange(
                                            len(len_per_period)
                                        )

                                        len_per_period_all.extend(
                                            [
                                                len_per_period[i]
                                                for i in sample_indices
                                            ]
                                        )
                                        len_per_period_stats_all.extend(
                                            [
                                                len_per_period_stats[i]
                                                for i in sample_indices
                                            ]
                                        )
                                        len_per_period_active_all.extend(
                                            [
                                                len_per_period_active[i]
                                                for i in sample_indices
                                            ]
                                        )
                                        data_list_test.extend(
                                            [
                                                data_list_per_period[i]
                                                for i in sample_indices
                                            ]
                                        )
                                    assert len(len_per_period_all) == len(
                                        data_list_test
                                    )

                if self.enable_segmentation:
                    len_per_period_all = np.array(len_per_period_all)
                    n_samples = 1000

                    if self.sampling_method == "uniform":
                        weights = len_per_period_all > 0
                    elif self.sampling_method == "balanced":
                        # Bin the lengths
                        binned_lengths = np.digitize(
                            len_per_period_all, balance_len_bins
                        )

                        # Create a dictionary to count the number of periods for each length
                        unique_lengths, counts = np.unique(
                            binned_lengths, return_counts=True
                        )
                        logging.info(
                            f"# of unique_lengths: {len(unique_lengths)}, # of counts: {counts}"
                        )
                        # Assign equal weight to each length category
                        length_weights = 1.0 / unique_lengths.size
                        # Calculate the weight for each period
                        weights = np.zeros(len(binned_lengths))
                        for length, count in zip(unique_lengths, counts):
                            if length == unique_lengths[0]:
                                continue
                            period_indices = np.where(binned_lengths == length)[0]
                            weights[period_indices] = length_weights / count

                    weights = weights / np.sum(weights)

                    sample_indices = np.random.choice(
                        len(weights),
                        min(n_samples, len(weights)),
                        replace=False,
                        p=weights,
                    )

                    data_list_test = [data_list_test[i] for i in sample_indices]

                    n_mean = np.mean(
                        [len_per_period_active_all[i] for i in sample_indices]
                    )
                    n_max = np.max(
                        [len_per_period_active_all[i] for i in sample_indices]
                    )
                    logging.info(
                        f"# of active flows per busy period: mean-{n_mean}, max-{n_max}"
                    )

                    n_mean = np.mean(
                        [len_per_period_stats_all[i] for i in sample_indices]
                    )
                    n_max = np.max(
                        [len_per_period_stats_all[i] for i in sample_indices]
                    )
                    logging.info(
                        f"# of flows per busy period: mean-{n_mean}, max-{n_max}"
                    )
            else:
                data_list = self.__read_data_list(self.dir_output)
                if self.test_on_train:
                    data_list_test = data_list["train"]
                else:
                    data_list_test = data_list["test"]
                sample_index = np.random.choice(
                    np.arange(len(data_list_test)),
                    min(500, len(data_list_test)),
                    replace=False,
                )
                data_list_test = [data_list_test[i] for i in sample_index]
            self.test = self.__create_dataset(
                data_list_test,
            )
            logging.info(f"#tracks: test-{len(data_list_test)}")

    def train_dataloader(self):
        """
        Returns a PyTorch DataLoader for the training data.

        :return: A PyTorch DataLoader object.
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Returns a PyTorch DataLoader for the validation set.

        :return: A PyTorch DataLoader object.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    # Create test dataloader
    def test_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the test dataset.

        :return: DataLoader object with test dataset
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def __random_split_list(self, lst, percentage):
        length = len(lst)
        split_index = int(length * percentage / self.batch_size) * self.batch_size

        train_part = lst[:split_index]
        test_part = lst[split_index:]

        return train_part, test_part

    def __create_dataset(
        self,
        data_list,
    ):
        dir_input = self.dir_input
        enable_positional_encoding = self.enable_positional_encoding
        flow_size_threshold = self.flow_size_threshold
        enable_flowsim_gt = self.enable_flowsim_gt
        enable_remainsize = self.enable_remainsize
        enable_queuelen = self.enable_queuelen

        data_list_filtered = data_list
        logging.info(f"Using all samples: {len(data_list_filtered)}")

        if self.enable_segmentation:
            if self.enable_topo:
                return TopoFctSldnSegment(
                    data_list_filtered,
                    dir_input,
                    enable_positional_encoding,
                    flow_size_threshold,
                    enable_flowsim_gt,
                    enable_remainsize=enable_remainsize,
                    enable_queuelen=enable_queuelen,
                )

    def __dump_data_list(self, path):
        with open(f"{path}/data_list.json", "w") as fp:
            data_dict = {
                "train": self.train_list,
                "val": self.val_list,
                "test": self.val_list,
            }
            json.dump(data_dict, fp)

    def __read_data_list(self, path):
        f = open(f"{path}/data_list.json", "r")
        return json.loads(f.read())


class TopoFctSldnSegment(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
        enable_positional_encoding,
        flow_size_threshold,
        enable_flowsim_gt=False,
        enable_remainsize=False,
        enable_queuelen=False,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        self.lr = 10.0
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_flowsim_gt = enable_flowsim_gt
        self.enable_remainsize = enable_remainsize
        self.enable_queuelen = enable_queuelen
        self.n_links = 96
        logging.info(
            f"call TopoFctSldnSegment. data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}, enable_positional_encoding={enable_positional_encoding}, flow_size_threshold={flow_size_threshold}, enable_flowsim_gt={enable_flowsim_gt}, enable_remainsize={enable_remainsize}, enable_queuelen={enable_queuelen}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id, _ = self.data_list[idx]
        src_dst_pair_target_str = (
            "_".join([str(x) for x in src_dst_pair_target]) + f"_seg{segment_id}"
        )

        dir_input_tmp = f"{self.dir_input}/{spec}"

        busy_periods = np.load(
            f"{dir_input_tmp}/period{topo_type}_t{self.flow_size_threshold}.npy",
            allow_pickle=True,
        )

        fid = np.array(busy_periods[segment_id]).astype(np.int32)
        assert np.all(fid[:-1] <= fid[1:])
        fid_rank = {fid: rank for rank, fid in enumerate(fid)}

        n_flows = len(fid)
        with open(f"{dir_input_tmp}/fsize.npy", "rb") as f:
            sizes = np.load(f)[fid]
        with open(f"{dir_input_tmp}/fat.npy", "rb") as f:
            fats = np.load(f)[fid]
        with open(f"{dir_input_tmp}/flink.npy", "rb") as f:
            link_list = np.load(f)
        link_dict = {link: idx for idx, link in enumerate(link_list)}
        link_info = np.load(
            f"{dir_input_tmp}/flow_to_path.npy",
            allow_pickle=True,
        )
        link_info = [[link_dict[link] for link in link_info[i]] for i in fid]

        n_links_passed = np.array([len(path) for path in link_info])
        fcts_flowsim_path = f"{dir_input_tmp}/flowsim_fct.npy"
        if os.path.exists(fcts_flowsim_path):
            base_delay = get_base_delay_path(
                sizes=sizes,
                n_links_passed=n_links_passed,
                lr_bottleneck=self.lr,
            )
            i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
            fcts_flowsim = np.load(fcts_flowsim_path)[fid] + base_delay
            sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
        else:
            sldn_flowsim = np.ones_like(fats)

        if self.enable_flowsim_gt:
            fcts = fcts_flowsim
            i_fcts = i_fcts_flowsim
        else:
            with open(f"{dir_input_tmp}/fct{topo_type}.npy", "rb") as f:
                fcts = np.load(f)[fid]
            with open(f"{dir_input_tmp}/fct_i{topo_type}.npy", "rb") as f:
                i_fcts = np.load(f)[fid]
        assert len(fcts) == len(fid)==len(fats)==len(i_fcts)==len(sizes)

        fats = fats - fats[0]
        fcts_stamp = fats + fcts
        events = []
        for i in range(len(fats)):
            events.append((fats[i], "arrival", fid_rank[fid[i]]))
            events.append((fcts_stamp[i], "departure", fid_rank[fid[i]]))
        events.sort(key=lambda x: x[0])

        # Concatenate the arrays
        flow_active_list = np.zeros((n_flows, 2), dtype=int)
        time_delta_list = np.zeros(n_flows * 2, dtype=np.float32)
        time_last = 0
        for event_idx, event in enumerate(events):
            time, event_type, flowid = event
            if event_type == "arrival":
                flow_active_list[flowid, 0] = event_idx
            elif event_type == "departure":
                flow_active_list[flowid, 1] = event_idx
            time_delta_list[event_idx] = (time - time_last) / 1000.0
            time_last = time
        active_flow_idx_list = [
            np.logical_and(flow_active_list[:, 0] <= j, flow_active_list[:, 1] > j)
            for j in range(len(events))
        ]

        if self.enable_queuelen:
            queuelen_list_total = np.load(
                f"{dir_input_tmp}/qlen{topo_type}.npy",
                allow_pickle=True,
            ).item()
            queuelen_list = [np.array(queuelen_list_total[i]) for i in fid]
            queuelen_list = [np.power(x + 1e-6, 1 / 3) for x in queuelen_list]
            queuelen_link_list = link_info
        else:
            queuelen_list = None
            queuelen_link_list = None

        if self.enable_remainsize:
            busy_periods_remainsize = np.load(
                f"{dir_input_tmp}/period_remainsize{topo_type}_t{self.flow_size_threshold}.npy",
                allow_pickle=True,
            )
            receivedsize_list = busy_periods_remainsize[segment_id]
            receivedsize_list = [
                np.array(receivedsize) for receivedsize in receivedsize_list
            ]
            assert (
                len(receivedsize_list) == 2 * n_flows
            ), f"len(remain_size): {len(receivedsize_list)}, len(fid): {n_flows}"

            remainsize_list = []
            for idx, receivedsize in enumerate(receivedsize_list):
                active_flow_idx = active_flow_idx_list[idx]
                if sum(active_flow_idx) == len(receivedsize):
                    total_size = sizes[active_flow_idx]
                    # remain size ratio
                    remainsize_tmp = (total_size - receivedsize) / total_size
                    if not (remainsize_tmp >= 0).all():
                        remainsize_list.append([])
                    else:
                        remainsize_list.append(remainsize_tmp)
                else:
                    remainsize_list.append([])
        else:
            remainsize_list = None

        output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
        assert (output_data >= 1.0).all()

        sizes = np.log2(sizes + 1)
        param_path = f"{dir_input_tmp}/param{topo_type}.npy"
        if os.path.exists(param_path):
            param_data = np.load(f"{dir_input_tmp}/param{topo_type}.npy")
        else:
            param_data = np.zeros(13)
        param_data_repeat = np.repeat(param_data[:, np.newaxis], n_flows, axis=1).T
        if not self.enable_flowsim_gt:
            input_data = np.column_stack(
                (
                    sizes[:, np.newaxis],
                    sldn_flowsim[:, np.newaxis],
                    n_links_passed[:, np.newaxis],
                    param_data_repeat,
                )
            ).astype(np.float32)
        else:
            input_data = np.column_stack(
                (
                    sizes[:, np.newaxis],
                    np.zeros_like(sizes)[:, np.newaxis],
                    n_links_passed[:, np.newaxis],
                    param_data_repeat,
                )
            ).astype(np.float32)

        # Compute the adjacency matrix for the bipartite graph
        edge_index = self.compute_edge_index(link_info)

        return (
            input_data,  # (n_flows,2)
            output_data,  # (n_flows,1)
            spec + topo_type + "_" + src_dst_pair_target_str,
            remainsize_list,
            queuelen_list,
            queuelen_link_list,
            flow_active_list,  # (n_flows,2)
            time_delta_list,  # (n_events,1)
            edge_index,  # (2, n_edges)
            self.n_links,
        )

    def compute_edge_index(self, link_info):
        edge_index = []
        n_flows = len(link_info)
        for flow_node_idx in range(n_flows):
            for link_idx in link_info[flow_node_idx]:
                edge_index.append([flow_node_idx, link_idx])

        edge_index = np.array(edge_index).T
        # Sort edge_index by destination node (second row)
        sorted_indices = np.lexsort((edge_index[0, :], edge_index[1, :]))
        edge_index = edge_index[:, sorted_indices]
        return edge_index
