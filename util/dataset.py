from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import json
import logging
import os
from .consts import (
    PLACEHOLDER,
    balance_len_bins,
    get_base_delay_transmission,
    get_base_delay_link,
    get_base_delay_path,
    # P99_PERCENTILE_LIST,
    # PERCENTILE_METHOD,
    N_BACKGROUND,
)


def collate_fn_link(batch):
    inputs, outputs, specs, src_dst_pairs, edge_indices = zip(*batch)

    # Get lengths of each sequence in the batch
    lengths = np.array([x.shape[0] for x in inputs]).astype(np.int64)

    # Pad sequences
    max_len = max(lengths)
    padded_inputs = np.zeros(
        (len(inputs), max_len, inputs[0].shape[1]), dtype=np.float32
    )
    padded_outputs = (
        np.ones((len(outputs), max_len, outputs[0].shape[1]), dtype=np.float32)
        * PLACEHOLDER
    )

    for i, (input, output) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, : input.shape[0], :] = input
        padded_outputs[i, : output.shape[0], :] = output

    # Determine the maximum number of edges
    max_num_edges = max(edge_index.shape[1] for edge_index in edge_indices)

    # Pad edge indices
    padded_edge_indices = []
    edge_indices_len = []
    for edge_index in edge_indices:
        padded_edge_index = np.full((2, max_num_edges), 0, dtype=edge_index.dtype)
        padded_edge_index[:, : edge_index.shape[1]] = edge_index
        padded_edge_indices.append(torch.tensor(padded_edge_index, dtype=torch.long))
        edge_indices_len.append(edge_index.shape[1])

    padded_edge_indices = torch.stack(padded_edge_indices)

    return (
        torch.tensor(padded_inputs),
        torch.tensor(padded_outputs),
        lengths,
        specs,
        src_dst_pairs,
        padded_edge_indices,
        np.array(edge_indices_len),
    )


def collate_fn_path(batch):
    inputs, outputs, specs, src_dst_pairs, edge_indices = zip(*batch)

    # Get lengths of each sequence in the batch
    lengths = np.array([x.shape[0] for x in inputs]).astype(np.int64)

    # Pad sequences
    max_len = max(lengths)
    padded_inputs = np.zeros(
        (len(inputs), max_len, inputs[0].shape[1]), dtype=np.float32
    )
    padded_outputs = (
        np.ones((len(outputs), max_len, outputs[0].shape[1]), dtype=np.float32)
        * PLACEHOLDER
    )

    for i, (input, output) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, : input.shape[0], :] = input
        padded_outputs[i, : output.shape[0], :] = output

    # Determine the maximum number of edges
    max_num_edges = max(edge_index.shape[1] for edge_index in edge_indices)

    # Pad edge indices
    padded_edge_indices = []
    edge_indices_len = []
    for edge_index in edge_indices:
        padded_edge_index = np.full((2, max_num_edges), 0, dtype=edge_index.dtype)
        padded_edge_index[:, : edge_index.shape[1]] = edge_index
        padded_edge_indices.append(torch.tensor(padded_edge_index, dtype=torch.long))
        edge_indices_len.append(edge_index.shape[1])

    padded_edge_indices = torch.stack(padded_edge_indices)
    return (
        torch.tensor(padded_inputs),
        torch.tensor(padded_outputs),
        lengths,
        specs,
        src_dst_pairs,
        padded_edge_indices,
        np.array(edge_indices_len),
    )


class DataModulePerFlow(LightningDataModule):
    def __init__(
        self,
        dir_input,
        shard_list,
        n_flows_list,
        n_hosts_list,
        sample_list,
        batch_size,
        num_workers,
        train_frac,
        dir_output,
        lr,
        topo_type="",
        output_type="queueLen",
        mode="train",
        enable_segmentation=False,
        enable_positional_encoding=False,
        flow_size_threshold=100000,
        enable_gnn=False,
        enable_abstime=False,
        segments_per_seq=200,
        sampling_method="uniform",  # uniform, weighted, balanced
        enable_path=False,
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
        self.segments_per_seq = segments_per_seq
        self.sampling_method = sampling_method
        self.enable_path = enable_path
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_gnn = enable_gnn
        self.enable_abstime = enable_abstime
        logging.info(
            f"call DataModulePerFlow: dir_input={dir_input}, dir_output={dir_output}, lr={lr}, topo_type={topo_type}, enable_segmentation={enable_segmentation}, segments_per_seq={segments_per_seq}, sampling_method={sampling_method}, enable_path={enable_path}"
        )
        data_list = []
        if mode == "train":
            if enable_segmentation:
                len_per_period_all = []
            for shard in shard_list:
                for n_flows in n_flows_list:
                    for n_hosts in n_hosts_list:
                        topo_type_cur = topo_type.replace("-x_", f"-{n_hosts}_")
                        spec = (
                            f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                        )
                        for sample in sample_list:
                            # qfeat=np.load(f"{dir_input}/{spec}/qfeat{topo_type_cur}s{sample}.npy")
                            # flow_id_list=qfeat[:,0]
                            # fsize=np.load(f"{dir_input}/{spec}/fsize.npy")

                            # statss = np.load(f'{dir_input}/{spec}/stats.npy', allow_pickle=True)
                            # if float(statss.item().get("load_bottleneck_target")) > 0.8: continue

                            file_suffix = f"s{sample}_i0"
                            fid = np.load(
                                f"{dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy"
                            )
                            if (
                                len(fid) == len(set(fid))
                                and np.all(fid[:-1] <= fid[1:])
                                and len(fid) % n_flows == 0
                            ):
                                if enable_segmentation:
                                    busy_periods = np.load(
                                        f"{dir_input}/{spec}/period{topo_type_cur}{file_suffix}_t{flow_size_threshold}.npy",
                                        allow_pickle=True,
                                    )
                                    # if self.enable_path:
                                    #     busy_periods = []
                                    #     for period in busy_periods_ori:
                                    #         if len(period) < 5000:
                                    #             busy_periods.append(period)
                                    # else:
                                    #     busy_periods = busy_periods_ori

                                    # len_per_period = [int(period[1])-int(period[0])+1 for period in busy_periods]

                                    len_per_period = [
                                        len(period) for period in busy_periods
                                    ]

                                    if np.sum(len_per_period) > 0:
                                        data_list_per_period = [
                                            (
                                                spec,
                                                (0, n_hosts - 1),
                                                topo_type_cur + file_suffix,
                                                int(segment_id),
                                                (
                                                    int(busy_periods[segment_id][0]),
                                                    int(busy_periods[segment_id][0]),
                                                ),
                                            )
                                            for segment_id in range(len(busy_periods))
                                        ]

                                        sample_indices = np.random.choice(
                                            len(len_per_period),
                                            segments_per_seq * 5,
                                            replace=True,
                                        )
                                        # sample_indices = np.arange(len(len_per_period))

                                        len_per_period_all.extend(
                                            [len_per_period[i] for i in sample_indices]
                                        )
                                        data_list.extend(
                                            [
                                                data_list_per_period[i]
                                                for i in sample_indices
                                            ]
                                        )

                                else:
                                    data_list.append(
                                        (
                                            spec,
                                            (0, n_hosts - 1),
                                            topo_type_cur + f"s{sample}",
                                        )
                                    )

            if enable_segmentation:
                len_per_period_all = np.array(len_per_period_all)
                n_samples = (
                    len(shard_list)
                    * len(n_flows_list)
                    * len(n_hosts_list)
                    * len(sample_list)
                    * segments_per_seq
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
                    print(
                        f"num of unique_lengths: {len(unique_lengths)}, num of counts: {counts}"
                    )
                    # Assign equal weight to each length category
                    length_weights = 1.0 / unique_lengths.size
                    # Calculate the weight for each period
                    weights = np.zeros(len(binned_lengths))
                    for length, count in zip(unique_lengths, counts):
                        period_indices = np.where(binned_lengths == length)[0]
                        weights[period_indices] = length_weights / count
                else:
                    raise ValueError(f"Unsupported sampling method: {sampling_method}")

                weights = weights / np.sum(weights)
                # sample_indices = np.random.choice(len(weights), n_samples, replace=True, p=weights)
                sample_indices = np.random.choice(
                    len(weights), min(n_samples, len(weights)), replace=False, p=weights
                )

                data_list = [data_list[i] for i in sample_indices]
                n_mean = np.mean([len_per_period_all[i] for i in sample_indices])
                n_max = np.max([len_per_period_all[i] for i in sample_indices])
                logging.info(
                    f"mean num of flows per busy period: mean-{n_mean}, max-{n_max}"
                )
            np.random.shuffle(data_list)
        self.data_list = data_list
        self.test_on_train = test_on_train
        self.test_on_empirical = test_on_empirical
        self.sampling_method = sampling_method
        self.test_on_manual = test_on_manual
        self.output_type = output_type

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
                self.dir_input,
                self.enable_positional_encoding,
                self.flow_size_threshold,
                self.enable_gnn,
                self.enable_abstime,
            )
            self.val = self.__create_dataset(
                self.val_list,
                self.dir_input,
                self.enable_positional_encoding,
                self.flow_size_threshold,
                self.enable_gnn,
                self.enable_abstime,
            )

            self.__dump_data_list(self.dir_output)

        if stage == "test":
            if self.test_on_manual:
                data_list_test = []
                for shard in np.arange(0, 3000):
                    for n_flows in [30000]:
                        for n_hosts in [2, 3, 4, 5, 6, 7, 8]:
                            topo_type_cur = self.topo_type.replace(
                                "x-x", f"{n_hosts}-{n_hosts}"
                            )
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                            dir_input_tmp = f"{self.dir_input}/{spec}"
                            if not os.path.exists(f"{dir_input_tmp}/flow_src_dst.npy"):
                                continue
                            flow_src_dst = np.load(f"{dir_input_tmp}/flow_src_dst.npy")
                            stats = np.load(
                                f"{dir_input_tmp}/{spec}/stats.npy", allow_pickle=True
                            )

                            n_flows_total = stats["n_flows"]
                            if len(flow_src_dst) == n_flows_total:
                                target_idx = stats["host_pair_list"].index(
                                    (0, n_hosts - 1)
                                )
                                size_dist = stats["size_dist_candidates"][
                                    target_idx
                                ].decode("utf-8")
                                if size_dist != "gaussian":
                                    continue
                                data_list_test.append(
                                    (spec, (0, n_hosts - 1), topo_type_cur)
                                )
            else:
                if self.test_on_empirical:
                    data_list_test = []

                    if self.enable_path:
                        shard_list = np.arange(0, 200)
                        n_hosts_list = [5]
                    else:
                        shard_list = np.arange(0, 100)
                        n_hosts_list = [21]

                    n_flows_list = [2000]
                    sample_list = [0]
                    if self.enable_segmentation:
                        len_per_period_all = []

                    for shard in shard_list:
                        for n_flows in n_flows_list:
                            for n_hosts in n_hosts_list:
                                topo_type_cur = self.topo_type.replace(
                                    "-x_", f"-{n_hosts}_"
                                )
                                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                                for sample in sample_list:
                                    # statss = np.load(f'{self.dir_input}/{spec}/stats.npy', allow_pickle=True)
                                    # if float(statss.item().get("load_bottleneck_target")) > 0.8: continue

                                    file_suffix = f"s{sample}_i0"
                                    fid = np.load(
                                        f"{self.dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy"
                                    )
                                    if (
                                        len(fid) == len(set(fid))
                                        and np.all(fid[:-1] <= fid[1:])
                                        and len(fid) % n_flows == 0
                                    ):
                                        if self.enable_segmentation:
                                            busy_periods = np.load(
                                                f"{self.dir_input}/{spec}/period{topo_type_cur}{file_suffix}_t{self.flow_size_threshold}.npy",
                                                allow_pickle=True,
                                            )

                                            # len_per_period = [int(period[1])-int(period[0])+1 for period in busy_periods]
                                            len_per_period = [
                                                len(period) for period in busy_periods
                                            ]

                                            if np.sum(len_per_period) > 0:
                                                data_list_per_period = [
                                                    (
                                                        spec,
                                                        (0, n_hosts - 1),
                                                        topo_type_cur + file_suffix,
                                                        int(segment_id),
                                                        (
                                                            int(
                                                                busy_periods[
                                                                    segment_id
                                                                ][0]
                                                            ),
                                                            int(
                                                                busy_periods[
                                                                    segment_id
                                                                ][0]
                                                            ),
                                                        ),
                                                    )
                                                    for segment_id in range(
                                                        len(busy_periods)
                                                    )
                                                ]
                                                sample_indices = np.random.choice(
                                                    len(len_per_period),
                                                    self.segments_per_seq * 5,
                                                    replace=True,
                                                )
                                                # sample_indices = np.arange(
                                                #     len(len_per_period)
                                                # )

                                                len_per_period_all.extend(
                                                    [
                                                        len_per_period[i]
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
                                        else:
                                            data_list_test.append(
                                                (
                                                    spec,
                                                    (0, n_hosts - 1),
                                                    topo_type_cur + f"s{sample}",
                                                )
                                            )
                    if self.enable_segmentation:
                        len_per_period_all = np.array(len_per_period_all)
                        n_samples = (
                            len(shard_list)
                            * len(n_flows_list)
                            * len(n_hosts_list)
                            * len(sample_list)
                            * self.segments_per_seq
                        )

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
                            print(
                                f"num of unique_lengths: {len(unique_lengths)}, num of counts: {counts}"
                            )
                            # Assign equal weight to each length category
                            length_weights = 1.0 / unique_lengths.size
                            # Calculate the weight for each period
                            weights = np.zeros(len(binned_lengths))
                            for length, count in zip(unique_lengths, counts):
                                period_indices = np.where(binned_lengths == length)[0]
                                weights[period_indices] = length_weights / count

                        weights = weights / np.sum(weights)
                        # sample_indices = np.random.choice(
                        #     len(weights),
                        #     n_samples,
                        #     replace=True,
                        #     p=weights,
                        # )
                        sample_indices = np.random.choice(
                            len(weights),
                            min(n_samples, len(weights)),
                            replace=False,
                            p=weights,
                        )

                        data_list_test = [data_list_test[i] for i in sample_indices]

                        n_mean = np.mean(
                            [len_per_period_all[i] for i in sample_indices]
                        )
                        n_max = np.max([len_per_period_all[i] for i in sample_indices])
                        logging.info(
                            f"mean num of flows per busy period: mean-{n_mean}, max-{n_max}"
                        )
                else:
                    data_list = self.__read_data_list(self.dir_output)
                    if self.test_on_train:
                        data_list_test = data_list["train"]
                    else:
                        data_list_test = data_list["test"]
            self.test = self.__create_dataset(
                data_list_test,
                self.dir_input,
                self.enable_positional_encoding,
                self.flow_size_threshold,
                self.enable_gnn,
                self.enable_abstime,
            )
            logging.info(f"#tracks: test-{len(data_list_test)}")

    def switch_to_other_epochs_logic(self):
        self.train.use_first_epoch_logic = False

    def train_dataloader(self):
        """
        Returns a PyTorch DataLoader for the training data.

        :return: A PyTorch DataLoader object.
        :rtype: torch.utils.data.DataLoader
        """

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn_path if self.enable_path else collate_fn_link,
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
            collate_fn=collate_fn_path if self.enable_path else collate_fn_link,
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
            collate_fn=collate_fn_path if self.enable_path else collate_fn_link,
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
        dir_input,
        enable_positional_encoding,
        flow_size_threshold,
        enable_gnn,
        enable_abstime,
    ):
        if self.enable_segmentation:
            if self.enable_path:
                return PathFctSldnSegment(
                    data_list,
                    dir_input,
                    enable_positional_encoding,
                    flow_size_threshold,
                    enable_gnn,
                )
            else:
                return LinkFctSldnSegment(
                    data_list,
                    dir_input,
                    enable_positional_encoding,
                    flow_size_threshold,
                    enable_gnn,
                    enable_abstime,
                )
        else:
            if self.output_type == "queueLen":
                return LinkQueueLen(
                    data_list,
                    dir_input,
                )
            elif self.output_type == "fctSldn":
                return LinkFctSldn(
                    data_list,
                    dir_input,
                )
            else:
                assert "output_type not supported"

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


class LinkQueueLen(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        logging.info(f"call LinkQueueLen: data_list={len(data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])

        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"

        # fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
        sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
        fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
        fats_ia_flowsim = np.diff(fats_flowsim)
        fats_ia_flowsim = np.insert(fats_ia_flowsim, 0, 0).astype(np.float32)
        sizes_flowsim = sizes_flowsim.astype(np.float32)
        input = np.concatenate(
            (sizes_flowsim[:, None], fats_ia_flowsim[:, None]), axis=1
        )

        qfeat = np.load(f"{dir_input_tmp}/qfeat{topo_type}.npy")
        if len(qfeat) != len(sizes_flowsim):
            print(f"qfeat shape mismatch: {len(qfeat)} vs {len(sizes_flowsim)}")
            assert False
        queue_lengths_dict = {qfeat[i, 0]: qfeat[i, 2] for i in range(len(qfeat))}
        output = (
            np.array(
                [queue_lengths_dict[flow_id] for flow_id in range(len(sizes_flowsim))]
            )
            .reshape(-1, 1)
            .astype(np.float32)
        )
        return (
            input,
            output,
            spec + topo_type,
            src_dst_pair_target_str,
        )


class LinkFctSldn(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        logging.info(f"call LinkFctSldn: data_list={len(data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])

        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"

        fid = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
        sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
        fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
        sizes_flowsim = sizes_flowsim[fid]
        fats_flowsim = fats_flowsim[fid]

        # Calculate inter-arrival times and adjust the first element
        fats_ia_flowsim = np.diff(fats_flowsim)
        fats_ia_flowsim = np.insert(fats_ia_flowsim, 0, 0)

        # Combine flow sizes and inter-arrival times into the input tensor
        input_data = np.column_stack((sizes_flowsim, fats_ia_flowsim)).astype(
            np.float32
        )

        fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
        i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
        output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)

        return input_data, output_data, spec + topo_type, src_dst_pair_target_str


class LinkFctSldnSegment(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
        enable_positional_encoding,
        flow_size_threshold,
        enable_gnn,
        enable_abstime,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        self.lr = 10.0
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_gnn = enable_gnn
        self.enable_abstime = enable_abstime
        logging.info(
            f"call LinkFctSldnSegment: data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}, enable_positional_encoding={enable_positional_encoding}, flow_size_threshold={flow_size_threshold}, enable_gnn={enable_gnn},enable_abstime={enable_abstime}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id, _ = self.data_list[idx]
        src_dst_pair_target_str = (
            "_".join([str(x) for x in src_dst_pair_target]) + f"_seg{segment_id}"
        )

        dir_input_tmp = f"{self.dir_input}/{spec}"
        feat_path = f"{dir_input_tmp}/feat{topo_type}_seg{segment_id}.npz"

        if not os.path.exists(feat_path) or self.use_first_epoch_logic:
            busy_periods = np.load(
                f"{dir_input_tmp}/period{topo_type}_t{self.flow_size_threshold}.npy",
                allow_pickle=True,
            )
            busy_periods_time = np.load(
                f"{dir_input_tmp}/period_time{topo_type}_t{self.flow_size_threshold}.npy"
            )
            assert len(busy_periods) == len(busy_periods_time)

            fid = np.array(busy_periods[segment_id])
            period_start_time, period_end_time = busy_periods_time[segment_id]
            assert np.all(fid[:-1] <= fid[1:])

            # fid = np.arange(busy_period[0], busy_period[1] + 1)
            sizes = np.load(f"{dir_input_tmp}/fsize.npy")[fid]
            fats = np.load(f"{dir_input_tmp}/fat.npy")[fid]
            fcts_flowsim = np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid]

            n_links_passed = np.ones_like(fcts_flowsim) * 2
            base_delay = get_base_delay_link(sizes, n_links_passed, self.lr)
            i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
            fcts_flowsim += base_delay
            sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)

            # Calculate inter-arrival times and adjust the first element
            if self.enable_abstime:
                fats_ia = fats - fats[0]
            else:
                fats_ia = np.diff(fats)
                fats_ia = np.insert(fats_ia, 0, 0)

            # seq_len=np.full((len(fid), 1), len(fid))
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")[fid]
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")[fid]
            output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
            assert (output_data >= 1.0).all()

            flag_from_last_period = np.array(fats < period_start_time)
            flag_flow_incomplete = np.array(fats + fcts > period_end_time)
            assert not flag_flow_incomplete.all()

            sizes = np.log1p(sizes)
            fats_ia = np.log1p(fats_ia)
            sldn_flowsim[flag_flow_incomplete] = 0
            output_data[flag_flow_incomplete] = PLACEHOLDER
            # Generate positional encoding
            if self.enable_positional_encoding:
                positional_encodings = self.get_positional_encoding(len(fid), 3)
                input_data = np.column_stack(
                    (
                        sizes,
                        fats_ia,
                        sldn_flowsim,
                        flag_from_last_period,
                        positional_encodings,
                    )
                ).astype(np.float32)
            else:
                input_data = np.column_stack(
                    (sizes, fats_ia, sldn_flowsim, flag_from_last_period)
                ).astype(np.float32)
            # input_data = np.column_stack((input_data, seq_len)).astype(np.float32)

            # Compute the adjacency matrix for the bipartite graph
            edge_index = self.compute_edge_index(fid)

            # assert (input_data >= 0.0).all()
            # Optionally save features for future epochs
            # np.savez(feat_path, input_data=input_data, output_data=output_data, adj_matrix=adj_matrix)
        else:
            feat = np.load(feat_path)
            input_data = feat["input_data"]
            output_data = feat["output_data"]

        return (
            input_data,
            output_data,
            spec + topo_type,
            src_dst_pair_target_str,
            edge_index,
        )

    def get_positional_encoding(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term[: d_model // 2])
        pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
        return pe

    def compute_edge_index(self, fid):
        edge_index = []
        n_flows = len(fid)
        for i in range(len(fid)):
            flow_node_idx = i
            edge_index.append([n_flows, flow_node_idx])
            edge_index.append([flow_node_idx, n_flows])

        edge_index = np.array(edge_index).T
        return edge_index


class PathFctSldnSegment(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
        enable_positional_encoding,
        flow_size_threshold,
        enable_gnn,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        self.lr = 10.0
        self.enable_positional_encoding = enable_positional_encoding
        self.flow_size_threshold = flow_size_threshold
        self.enable_gnn = enable_gnn
        logging.info(
            f"call PathFctSldnSegment: data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}, enable_positional_encoding={enable_positional_encoding}, flow_size_threshold={flow_size_threshold}, enable_gnn={enable_gnn}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id, _ = self.data_list[idx]
        src_dst_pair_target_str = (
            "_".join([str(x) for x in src_dst_pair_target]) + f"_seg{segment_id}"
        )

        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"
        feat_path = f"{dir_input_tmp}/feat{topo_type}_seg{segment_id}.npz"

        if not os.path.exists(feat_path) or self.use_first_epoch_logic:
            n_hosts = int(spec.split("_")[2][6:])

            busy_periods = np.load(
                f"{dir_input_tmp}/period{topo_type}_t{self.flow_size_threshold}.npy",
                allow_pickle=True,
            )
            busy_periods_time = np.load(
                f"{dir_input_tmp}/period_time{topo_type}_t{self.flow_size_threshold}.npy"
            )
            assert len(busy_periods) == len(busy_periods_time)

            fid_period = np.array(busy_periods[segment_id]).astype(int)
            fid_period = np.sort(fid_period)
            period_start_time, period_end_time = busy_periods_time[segment_id]

            # get all previous flows
            fid_ori = np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
            sizes = np.load(f"{dir_input_tmp}/fsize.npy")[fid_ori]
            fats = np.load(f"{dir_input_tmp}/fat.npy")[fid_ori]
            fsd = np.load(f"{dir_input_tmp}/fsd.npy")[fid_ori]
            fcts_flowsim = np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid_ori]
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")

            fidx_prev = fid_ori <= np.max(fid_period)

            fid_prev = fid_ori[fidx_prev]
            sizes = sizes[fidx_prev]
            fats = fats[fidx_prev]
            fsd = fsd[fidx_prev]
            fcts = fcts[fidx_prev]
            i_fcts = i_fcts[fidx_prev]
            fcts_flowsim = fcts_flowsim[fidx_prev]

            # compute propagation delay
            n_links_passed = abs(fsd[:, 0] - fsd[:, 1]) + 2
            base_delay = get_base_delay_path(
                sizes=sizes,
                n_links_passed=n_links_passed,
                lr_bottleneck=self.lr,
            )
            i_fcts_flowsim = get_base_delay_transmission(sizes, self.lr) + base_delay
            fcts_flowsim += base_delay
            sldn_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)

            fid_period_idx = np.array(
                [
                    np.where(fid_prev == ele)[0][0] if ele in fid_prev else -1
                    for ele in fid_period
                ]
            )

            flowsim_dist = np.zeros((len(fid_period), N_BACKGROUND))
            for flow_idx, flow_id in enumerate(fid_period_idx):
                flow_id_target = np.logical_and(
                    np.logical_and(
                        fsd[:, 0] == fsd[flow_id, 0], fsd[:, 1] == fsd[flow_id, 1]
                    ),
                    fats + fcts < fats[flow_id] + fcts[flow_id],
                )
                sldn_flowsim_tmp = sldn_flowsim[flow_id_target]
                n_tmp = min(len(sldn_flowsim_tmp), N_BACKGROUND)
                flowsim_dist[flow_idx, :n_tmp] = sldn_flowsim_tmp[-n_tmp:]

            sizes = sizes[fid_period_idx]
            fats = fats[fid_period_idx]
            sldn_flowsim = sldn_flowsim[fid_period_idx]
            fsd = fsd[fid_period_idx]
            n_links_passed = n_links_passed[fid_period_idx]
            fcts = fcts[fid_period_idx]
            i_fcts = i_fcts[fid_period_idx]

            output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
            assert (output_data >= 1.0).all()

            # Calculate inter-arrival times and adjust the first element
            fats_ia = np.diff(fats)
            fats_ia = np.insert(fats_ia, 0, 0)
            assert (fats_ia >= 0).all()
            # fats_ia[fats_ia < 0] = 0
            # fats_ia = fats - np.min(fats)

            # output_data[sizes > self.flow_size_threshold] = PLACEHOLDER

            sizes = np.log1p(sizes)
            fats_ia = np.log1p(fats_ia)
            flag_from_last_period = np.array(fats < period_start_time)
            flag_flow_incomplete = np.array(fats + fcts > period_end_time)
            assert not flag_flow_incomplete.all()

            sldn_flowsim[flag_flow_incomplete] = 0
            flowsim_dist[flag_flow_incomplete] = 0
            output_data[flag_flow_incomplete] = PLACEHOLDER

            # Generate positional encoding
            if self.enable_positional_encoding:
                positional_encodings = self.get_positional_encoding(len(fid_period), 4)
                input_data = np.column_stack(
                    (
                        fats_ia,
                        sizes,
                        n_links_passed,
                        sldn_flowsim,
                        flowsim_dist,
                        flag_from_last_period,
                        positional_encodings,
                    )
                ).astype(np.float32)
            else:
                input_data = np.column_stack(
                    (
                        fats_ia,
                        sizes,
                        n_links_passed,
                        sldn_flowsim,
                        flowsim_dist,
                        flag_from_last_period,
                    )
                ).astype(np.float32)
            # input_data = np.log1p(input_data)

            # Compute the adjacency matrix for the bipartite graph
            edge_index = self.compute_edge_index(n_hosts, fsd)

            # np.savez(feat_path, input_data=input_data, output_data=output_data,edge_index=edge_index)
        else:
            feat = np.load(feat_path)
            input_data = feat["input_data"]
            output_data = feat["output_data"]
            edge_index = feat["edge_index"]
        return (
            input_data,
            output_data,
            spec + topo_type,
            src_dst_pair_target_str,
            edge_index,
        )

    def get_positional_encoding(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term[: d_model // 2])
        pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
        return pe

    def compute_edge_index(self, n_hosts, fsd_flowsim):
        edge_index = []
        n_flows = len(fsd_flowsim)
        n_links = 2 * n_hosts - 1
        for i in range(n_flows):
            src = fsd_flowsim[i, 0]
            dst = fsd_flowsim[i, 1]
            flow_node_idx = i
            assert src < dst
            edge_index.append([n_flows + src, flow_node_idx])
            edge_index.append([flow_node_idx, n_flows + src])
            edge_index.append([n_flows + n_links + dst, flow_node_idx])
            edge_index.append([flow_node_idx, n_flows + n_links + dst])

            for link_idx in range(src, dst):
                edge_index.append([n_flows + n_hosts + link_idx, flow_node_idx])
                edge_index.append([flow_node_idx, n_flows + n_hosts + link_idx])
        # edge_index.append([n_flows, n_flows + n_hosts])
        # for link_idx in range(1, n_hosts - 1):
        #     edge_index.append([n_flows + link_idx, n_flows + n_hosts + link_idx])
        #     edge_index.append(
        #         [n_flows + n_hosts + link_idx - 1, n_flows + n_links + link_idx]
        #     )
        # edge_index.append([n_flows + 2 * n_hosts - 2, n_flows + n_links + n_hosts - 1])
        edge_index = np.array(edge_index).T
        return edge_index
