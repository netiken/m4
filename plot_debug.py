import numpy as np
from util.plot import plot_cdf, color_list
from util.consts import (
    balance_len_bins,
    balance_len_bins_label,
    balance_size_bins,
    balance_size_bins_label,
    get_base_delay_path,
    get_base_delay_transmission,
)
import matplotlib.pyplot as plt
import json
import glob
import os
from collections import defaultdict

topo_type_ori = "_topo-pl-x_"
lr = 10
rotation_angle = 30
np.random.seed(0)

# the relationship between estimation error and flow size for flowSim
dir_input_synthetic = "/data2/lichenni/perflow_path"
dir_output = "/data2/lichenni/output_perflow"
program_name_list = [
    "fct_path_200000_shard1000_nflows1_nhosts3_nsamples1_lr10Gbps",
]
version_id_list = [[0]]
title_str_list = ["flowSim", "m4"]
main_title_list = ["test", "empirical"]
metric_label_list = ["L1 Loss", "Relative error (%)"]

fig_index = 0
for version_id_test in [1]:
    res_total_per_flow = []
    fsize_total_per_flow = []
    res_save = []
    for program_name_idx, program_name in enumerate(program_name_list):
        for version_id in version_id_list[program_name_idx]:
            dir_train = f"{dir_output}/{program_name}/version_{version_id}/"
            print(f"dir_train: {dir_train}")
            if version_id_test == 0:
                # f = open(f"{dir_train}/data_list.json", "r")
                # data_list = json.loads(f.read())
                # data_list_test=data_list["test"]
                # dir_input=dir_input_synthetic
                data_list_test = []
                dir_input = dir_input_synthetic
                for shard in np.arange(1000):
                    for n_flows in [2000]:
                        for n_hosts in [3, 5, 7]:
                            src_dst_pair_target_str = "_".join(
                                [str(x) for x in [0, n_hosts - 1]]
                            )
                            topo_type_cur = topo_type_ori.replace("-x_", f"-{n_hosts}_")
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                            # statss = np.load(f'{dir_input}/{spec}/stats.npy', allow_pickle=True)
                            # print(statss.item())
                            # size_distribution_list=["cachefollower-all","hadoop-all","webserver-all"]
                            # if statss.item().get("size_dist_candidate") != 'webserver-all': continue
                            # if float(statss.item().get("load_bottleneck_target")) > 0.8: continue
                            # if float(statss.item().get("ias_sigma_candidate")) > 1.5: continue

                            for sample in [0]:
                                data_list_test.append(
                                    (
                                        spec,
                                        (0, n_hosts - 1),
                                        topo_type_cur + f"s{sample}_i0",
                                        None,
                                        None,
                                    )
                                )
            else:
                data_list_test = []
                dir_input = dir_input_synthetic + "_empirical"
                for shard in np.arange(100):
                    for n_flows in [2000]:
                        for n_hosts in [3]:
                            src_dst_pair_target_str = "_".join(
                                [str(x) for x in [0, n_hosts - 1]]
                            )
                            topo_type_cur = topo_type_ori.replace("-x_", f"-{n_hosts}_")
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"

                            statss = np.load(
                                f"{dir_input}/{spec}/stats.npy", allow_pickle=True
                            )
                            # print(statss.item())
                            # size_distribution_list=["cachefollower-all","hadoop-all","webserver-all"]
                            # if statss.item().get("size_dist_candidate") != 'webserver-all': continue
                            # if float(statss.item().get("load_bottleneck_target")) < 0.5: continue
                            # if float(statss.item().get("ias_sigma_candidate")) > 1.5: continue

                            for sample in [0]:
                                data_list_test.append(
                                    (
                                        spec,
                                        (0, n_hosts - 1),
                                        topo_type_cur + f"s{sample}_i0",
                                        None,
                                        None,
                                    )
                                )

                                # pattern=os.path.join(
                                #         dir_train,
                                #         'test',
                                #         f'version_{version_id_test}',
                                #         f'{spec}{topo_type_cur}s{sample}_i0_{src_dst_pair_target_str}_seg*'
                                #     )
                                # matching_directories = [d for d in glob.glob(pattern) if os.path.isdir(d)]
                                # for matching_directory in matching_directories:
                                #     segment_id=matching_directory.split("_seg")[-1]
                                #     data_list_test.append(
                                #         (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}_i0", int(segment_id), None)
                                #     )

            len_tracks = len(data_list_test)
            print(f"{program_name} loads {len_tracks} tracks")

            res_per_flow = []
            fsize_per_flow = []
            res_save_per_scenario = []
            for spec, src_dst_pair_target, topo_type, segment_id, _ in data_list_test:
                dir_input_tmp = f"{dir_input}/{spec}"

                fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
                # busy_periods=np.load(f"{dir_input_tmp}/period{topo_type}.npy", allow_pickle=True)
                # fid=np.array(busy_periods[segment_id])

                sizes = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
                fsd = np.load(f"{dir_input_tmp}/fsd.npy")[fid]
                fid_ori = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
                fid_idx = np.where(np.isin(fid_ori, fid))[0]
                if program_name_idx == 0:
                    fcts_flowsim = np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid]
                    n_links_passed = abs(fsd[:, 0] - fsd[:, 1]) + 2
                    base_delay = get_base_delay_path(sizes, n_links_passed, lr)
                    i_fcts_flowsim = get_base_delay_transmission(sizes, lr) + base_delay
                    fcts_flowsim += base_delay
                    est = np.divide(fcts_flowsim, i_fcts_flowsim)
                else:
                    data = np.load(
                        f"{dir_train}/test/version_{version_id_test}/{spec}{topo_type}_{src_dst_pair_target_str}/res.npz"
                    )
                    # gt = data['output'].flatten()
                    est = data["est"].flatten()
                fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")[fid_idx]
                fcts_i = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")[fid_idx]
                gt = np.divide(fcts, fcts_i)

                # idx_sldn_one=np.where(gt<2.0)[0]

                # gt=gt[~idx_sldn_one]
                # sizes=sizes[~idx_sldn_one]
                # est=est[~idx_sldn_one]

                if len(gt) == 0:
                    print(f"empty gt: {spec}{topo_type}_{src_dst_pair_target_str}")
                    continue
                tmp = np.abs(gt - est) / gt * 100
                res_per_flow.extend(tmp)
                fsize_per_flow.extend(sizes)
                res_save_per_scenario.append(tuple([est, sizes, gt]))

            res_total_per_flow.append(np.array(res_per_flow))
            fsize_total_per_flow.append(np.array(fsize_per_flow))
            res_save.append(res_save_per_scenario)
    # res_save=np.array(res_save).squeeze()
    # print(f"res_save: {np.array(res_save).shape}", dtype=object)
    res_save = np.array(res_save, dtype=object)
    np.save(f"./res/flowsim_path_{main_title_list[version_id_test]}.npy", res_save)
    fsize_total_per_flow = [
        np.digitize(x, balance_size_bins) for x in fsize_total_per_flow
    ]

    # plt.figure(fig_index,figsize=(5, 3))
    # plt.title(f"per-flow perf. on {main_title_list[version_id_test]} set", fontsize="x-large")
    # for j in range(len(program_name_list)):
    #     plt.scatter(fsize_total_per_flow[j],res_total_per_flow[j],label=title_str_list[j],s=1)
    # plt.xticks(ticks=np.arange(len(balance_size_bins_label)), labels=balance_size_bins_label,rotation=rotation_angle)
    # plt.legend()
    # # plt.xscale('log')
    # plt.xlabel("Flow size")
    # plt.ylabel(f"{metric_label_list[1]}")
    # fig_index+=1

    plt.figure(fig_index, figsize=(5, 3))
    plt.title(
        f"per-flow perf. on {main_title_list[version_id_test]} set", fontsize="x-large"
    )
    x = []
    for j in range(len(program_name_list)):
        for i in range(len(balance_size_bins_label)):
            tmp = res_total_per_flow[j][fsize_total_per_flow[j] == i]
            x.append([np.mean(tmp), np.percentile(tmp, 99)])
        x = np.array(x)
        plt.plot(x[:, 0], label=f"{title_str_list[0]}-mean")
        plt.plot(x[:, 1], label=f"{title_str_list[0]}-99th")
        plt.xticks(
            ticks=np.arange(len(balance_size_bins_label)),
            labels=balance_size_bins_label,
            rotation=rotation_angle,
        )
        plt.legend()

        plt.xlabel("Flow size")
        plt.ylabel(f"{metric_label_list[1]}")
        plt.axvline(x=6, color="r", linestyle="--")
        plt.axvline(x=7, color="b", linestyle="--")
        plt.axhline(y=50, color="k", linestyle="--")
        plt.text(0, 50, "50%", color="k")
        plt.axhline(y=10, color="k", linestyle="--")
        plt.text(0, 10, "10%", color="k")
        plt.yscale("log")
    fig_index += 1
