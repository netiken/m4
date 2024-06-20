from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
import torch
from .consts import (
    P99_PERCENTILE_LIST,
    PERCENTILE_METHOD,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
    BDP_DICT,
    LINK_TO_DELAY_DICT,
    get_size_bucket_list,
    get_size_bucket_list_output,
    get_base_delay_pmn,
)
from .func import decode_dict
import json
import logging
import os

# Custom collate function to handle variable sequence lengths
def collate_fn_per_flow(batch):
    inputs, outputs, specs, src_dst_pairs = zip(*batch)
    
    # Get lengths of each sequence in the batch
    lengths = np.array([x.shape[0] for x in inputs]).astype(np.int64)
    
    # Pad sequences
    max_len = max(lengths)
    padded_inputs = np.zeros((len(inputs), max_len, inputs[0].shape[1]), dtype=np.float32)
    padded_outputs = np.ones((len(outputs), max_len, outputs[0].shape[1]), dtype=np.float32)
    
    for i, (input, output) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, :input.shape[0], :] = input
        padded_outputs[i, :output.shape[0], :] = output
    
    return torch.tensor(padded_inputs), torch.tensor(padded_outputs), lengths, specs, src_dst_pairs
class PathDataModulePerFlow(LightningDataModule):
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
        data_list = []
        if mode == "train":
            for shard in shard_list:
                for n_flows in n_flows_list:
                    for n_hosts in n_hosts_list:
                        topo_type_cur = topo_type.replace(
                            "-x_", f"-{n_hosts}_"
                        )
                        spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                        for sample in sample_list:
                            # qfeat=np.load(f"{dir_input}/{spec}/qfeat{topo_type_cur}s{sample}.npy")
                            # flow_id_list=qfeat[:,0]
                            # fsize=np.load(f"{dir_input}/{spec}/fsize.npy")
                            fid = np.load(f"{dir_input}/{spec}/fid{topo_type_cur}s{sample}.npy")
                            if len(fid)==len(set(fid)) and np.all(fid[:-1] <= fid[1:]):
                                data_list.append(
                                    (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
                                )
                            
            np.random.shuffle(data_list)
        self.data_list = data_list
        self.test_on_train = test_on_train
        self.test_on_empirical = test_on_empirical
        self.test_on_manual = test_on_manual
        self.output_type=output_type
        
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
            )
            self.val = self.__create_dataset(
                self.val_list,
                self.dir_input,
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
                            stats = decode_dict(
                                np.load(
                                    f"{dir_input_tmp}/stats.npy",
                                    allow_pickle=True,
                                    encoding="bytes",
                                ).item()
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
                    for shard in np.arange(0, 2000):
                        for n_flows in [1000]:
                            for n_hosts in [3]:
                                topo_type_cur = self.topo_type.replace(
                                    "-x_", f"-{n_hosts}_"
                                )
                                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                                for sample in [0]:
                                    fid = np.load(f"{self.dir_input}/{spec}/fid{topo_type_cur}s{sample}.npy")
                                    if len(fid)==len(set(fid)) and np.all(fid[:-1] <= fid[1:]):
                                        data_list_test.append(
                                            (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
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
            collate_fn=collate_fn_per_flow,
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
            collate_fn=collate_fn_per_flow,
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
            collate_fn=collate_fn_per_flow,
        )

    def __random_split_list(self, lst, percentage):
        length = len(lst)
        split_index = int(length * percentage / self.batch_size) * self.batch_size

        train_part = lst[:split_index]
        test_part = lst[split_index:]

        return train_part, test_part

    def __create_dataset(self, data_list, dir_input):
        if self.output_type=="queueLen":
            return PathDatasetQueueLen(
                data_list,
                dir_input,
            )
        elif self.output_type=="fctSldn":
            return PathDatasetFctSldn(
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


class PathDatasetQueueLen(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        logging.info(
            f"call PathDatasetQueueLen: data_list={len(data_list)}"
        )

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
        fats_ia_flowsim=np.diff(fats_flowsim)
        fats_ia_flowsim=np.insert(fats_ia_flowsim, 0, 0).astype(np.float32)
        sizes_flowsim=sizes_flowsim.astype(np.float32)
        input=np.concatenate((sizes_flowsim[:,None],fats_ia_flowsim[:,None]),axis=1)
        
        qfeat=np.load(f"{dir_input_tmp}/qfeat{topo_type}.npy")
        if len(qfeat)!=len(sizes_flowsim):
            print(f"qfeat shape mismatch: {len(qfeat)} vs {len(sizes_flowsim)}")
            assert False
        queue_lengths_dict = {qfeat[i,0]: qfeat[i,2] for i in range(len(qfeat))}
        output=np.array([queue_lengths_dict[flow_id] for flow_id in range(len(sizes_flowsim))]).reshape(-1, 1).astype(np.float32)
        return (
            input,
            output,
            spec+topo_type,
            src_dst_pair_target_str,
        )
        
class PathDatasetFctSldn(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        logging.info(
            f"call PathDatasetFctSldn: data_list={len(data_list)}"
        )

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])
        
        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"
        
        fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
        sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
        fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
        sizes_flowsim=sizes_flowsim[fid]
        fats_flowsim=fats_flowsim[fid]
        
        # Calculate inter-arrival times and adjust the first element
        fats_ia_flowsim=np.diff(fats_flowsim)
        fats_ia_flowsim=np.insert(fats_ia_flowsim, 0, 0)
        
        # Combine flow sizes and inter-arrival times into the input tensor
        input_data = np.column_stack((sizes_flowsim, fats_ia_flowsim)).astype(np.float32)
        
        fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
        i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
        output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)
        
        return input_data, output_data, spec + topo_type, src_dst_pair_target_str
