from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch
from .consts import (
    PLACEHOLDER,
    balance_bins,
    get_base_delay_link,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
)
from .func import decode_dict
import json
import logging
import os

# Custom collate function to handle variable sequence lengths
def collate_fn_per_flow(batch):
    inputs, outputs, specs, src_dst_pairs, adj_matrices = zip(*batch)
    
    # Get lengths of each sequence in the batch
    lengths = np.array([x.shape[0] for x in inputs]).astype(np.int64)
    
    # Pad sequences
    max_len = max(lengths)
    padded_inputs = np.zeros((len(inputs), max_len, inputs[0].shape[1]), dtype=np.float32)
    padded_outputs = np.ones((len(outputs), max_len, outputs[0].shape[1]), dtype=np.float32)*PLACEHOLDER
    
    for i, (input, output) in enumerate(zip(inputs, outputs)):
        padded_inputs[i, :input.shape[0], :] = input
        padded_outputs[i, :output.shape[0], :] = output
    
    # Pad adjacency matrices to the maximum size in the batch
    max_num_flows = max([adj.shape[0] for adj in adj_matrices])
    max_num_links = max([adj.shape[1] for adj in adj_matrices])
    padded_adj_matrices = np.zeros((len(adj_matrices), max_num_flows, max_num_links), dtype=np.float32)
    
    for i, adj in enumerate(adj_matrices):
        padded_adj_matrices[i, :adj.shape[0], :adj.shape[1]] = adj
    return torch.tensor(padded_inputs), torch.tensor(padded_outputs), lengths, specs, src_dst_pairs, torch.tensor(padded_adj_matrices)

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
        segments_per_seq=200,
        sampling_method="uniform", # uniform, weighted, balanced
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
        self.enable_segmentation=enable_segmentation
        self.segments_per_seq=segments_per_seq
        self.sampling_method=sampling_method
        self.enable_path=enable_path
        logging.info(
            f"call DataModulePerFlow: dir_input={dir_input}, dir_output={dir_output}, lr={lr}, topo_type={topo_type}, enable_segmentation={enable_segmentation}, segments_per_seq={segments_per_seq}, sampling_method={sampling_method}, enable_path={enable_path}"
        )
        data_list = []
        if mode == "train":
            if enable_segmentation:
                len_per_period_all=[]
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
                            
                            # statss = np.load(f'{dir_input}/{spec}/stats.npy', allow_pickle=True)
                            # if float(statss.item().get("load_bottleneck_target")) > 0.8: continue
                            
                            file_suffix=f"s{sample}_i0"
                            fid = np.load(f"{dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy")
                            if len(fid)==len(set(fid)) and np.all(fid[:-1] <= fid[1:]) and len(fid)%n_flows==0:
                                if enable_segmentation:
                                    busy_periods=np.load(f"{dir_input}/{spec}/period{topo_type_cur}{file_suffix}.npy", allow_pickle=True)
                                    
                                    len_per_period = [int(period[1])-int(period[0])+1 for period in busy_periods]
                                    
                                    if np.sum(len_per_period)>0:
                                        data_list_per_period=[(spec, (0, n_hosts - 1), topo_type_cur+file_suffix, int(segment_id), (int(busy_periods[segment_id][0]), int(busy_periods[segment_id][1]))) for segment_id in range(len(busy_periods))]
                                        len_per_period_all.extend(len_per_period)
                                        data_list.extend(data_list_per_period)
                                        
                                    assert len(len_per_period_all)==len(data_list)
                                else:
                                    data_list.append(
                                        (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
                                    )
                                    
            if enable_segmentation:
                len_per_period_all=np.array(len_per_period_all)
                n_samples=len(shard_list)*len(n_flows_list)*len(n_hosts_list)*len(sample_list)*segments_per_seq
                 # Sample indices from the array based on the weights
                if sampling_method=="uniform":
                    weights = len_per_period_all > 0
                elif sampling_method=="weighted":
                    weights = len_per_period_all
                elif sampling_method=="balanced":
                    # Bin the lengths
                    binned_lengths = np.digitize(len_per_period_all, balance_bins)
                    
                    # Create a dictionary to count the number of periods for each length
                    unique_lengths, counts = np.unique(binned_lengths, return_counts=True)
                    print(f"num of unique_lengths: {len(unique_lengths)}, num of counts: {counts}")
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
                sample_indices = np.random.choice(len(weights), min(n_samples, len(weights)), replace=False, p=weights)
                
                data_list = [data_list[i] for i in sample_indices]
                n_mean = np.mean([len_per_period_all[i] for i in sample_indices])  
                logging.info(f"mean num of flows per busy period: {n_mean}")  
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
                    shard_list=np.arange(0, 200)
                    n_flows_list=[2000]
                    n_hosts_list=[21]
                    sample_list=[0]
                    if self.enable_segmentation:
                        len_per_period_all=[]
                        
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
                                    
                                    file_suffix=f"s{sample}_i0"
                                    fid = np.load(f"{self.dir_input}/{spec}/fid{topo_type_cur}{file_suffix}.npy")
                                    if len(fid)==len(set(fid)) and np.all(fid[:-1] <= fid[1:]) and len(fid)%n_flows==0:
                                        if self.enable_segmentation:
                                            busy_periods=np.load(f"{self.dir_input}/{spec}/period{topo_type_cur}{file_suffix}.npy", allow_pickle=True)

                                            len_per_period = [int(period[1])-int(period[0])+1 for period in busy_periods]
                                            
                                            if np.sum(len_per_period)>0:
                                                data_list_per_period=[(spec, (0, n_hosts - 1), topo_type_cur+file_suffix, int(segment_id), (int(busy_periods[segment_id][0]), int(busy_periods[segment_id][1]))) for segment_id in range(len(busy_periods))]
                                                len_per_period_all.extend(len_per_period)
                                                data_list_test.extend(data_list_per_period)
                                            assert len(len_per_period_all)==len(data_list_test)
                                        else:
                                            data_list_test.append(
                                                (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
                                            )
                    if self.enable_segmentation:
                        len_per_period_all=np.array(len_per_period_all)
                        n_samples=len(shard_list)*len(n_flows_list)*len(n_hosts_list)*len(sample_list)*self.segments_per_seq
                        
                        # weights = len_per_period_all > 0
                        
                        # Bin the lengths
                        binned_lengths = np.digitize(len_per_period_all, balance_bins)
                        
                        # Create a dictionary to count the number of periods for each length
                        unique_lengths, counts = np.unique(binned_lengths, return_counts=True)
                        print(f"num of unique_lengths: {len(unique_lengths)}, num of counts: {counts}")
                        # Assign equal weight to each length category
                        length_weights = 1.0 / unique_lengths.size
                        # Calculate the weight for each period
                        weights = np.zeros(len(binned_lengths))
                        for length, count in zip(unique_lengths, counts):
                            period_indices = np.where(binned_lengths == length)[0]
                            weights[period_indices] = length_weights / count
                
                        weights = weights / np.sum(weights)        
                        sample_indices = np.random.choice(len(weights), min(n_samples, len(weights)), replace=False, p=weights)
                            
                        data_list_test = [data_list_test[i] for i in sample_indices]
                        n_mean = np.mean([len_per_period_all[i] for i in sample_indices])  
                        logging.info(f"mean num of flows per busy period: {n_mean}")  
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
        if self.enable_segmentation:
            if self.enable_path:
                return PathFctSldnSegment(
                    data_list,
                    dir_input,
                )
            else:
                return LinkFctSldnSegment(
                    data_list,
                    dir_input,
                )
        else:
            if self.output_type=="queueLen":
                return LinkQueueLen(
                    data_list,
                    dir_input,
                )
            elif self.output_type=="fctSldn":
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
        logging.info(
            f"call LinkQueueLen: data_list={len(data_list)}"
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
        
class LinkFctSldn(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        logging.info(
            f"call LinkFctSldn: data_list={len(data_list)}"
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

class LinkFctSldnSegment(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        logging.info(
            f"call LinkFctSldnSegment: data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}"
        )
        self.lr=10.0

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id, busy_period = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])+f'_seg{segment_id}'
        
        dir_input_tmp = f"{self.dir_input}/{spec}"
        feat_path=f"{dir_input_tmp}/feat{topo_type}_seg{segment_id}.npz"
        
        if not os.path.exists(feat_path) or self.use_first_epoch_logic:
            # busy_periods=np.load(f"{dir_input_tmp}/period{topo_type}.npy", allow_pickle=True)
            # fid=[int(flow_id) for flow_id in busy_periods[segment_id]]
            # busy_period=busy_periods[segment_id]
            fid=np.arange(busy_period[0], busy_period[1]+1)
            # fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
            sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
            fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
            fcts_flowsim = np.load(f"{dir_input_tmp}/fct_flowsim.npy")[fid]
            
            sizes_flowsim=sizes_flowsim[fid]
            fats_flowsim=fats_flowsim[fid]
            
            n_links_passed=np.ones_like(fcts_flowsim)*2
            base_delay=get_base_delay_link(sizes_flowsim,n_links_passed,self.lr)
            i_fcts_flowsim = (sizes_flowsim + np.ceil(sizes_flowsim / MTU) * HEADER_SIZE) * BYTE_TO_BIT / self.lr + base_delay
            fcts_flowsim += base_delay
            
            sldn_flowsim=np.divide(fcts_flowsim, i_fcts_flowsim)
            
            # Calculate inter-arrival times and adjust the first element
            fats_ia_flowsim=np.diff(fats_flowsim)
            fats_ia_flowsim=np.insert(fats_ia_flowsim, 0, 0)
            
            # Combine flow sizes and inter-arrival times into the input tensor
            input_data = np.column_stack((sizes_flowsim, fats_ia_flowsim, sldn_flowsim)).astype(np.float32)
            assert (input_data>=0.0).all()
            
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
            output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)[fid]
            assert (output_data>=1.0).all()
            
            # Compute the adjacency matrix for the bipartite graph
            adj_matrix = self.compute_adjacency_matrix(fid)
            
            # np.savez(feat_path, input_data=input_data, output_data=output_data)
        else:
            feat=np.load(feat_path)
            input_data=feat["input_data"]
            output_data=feat["output_data"]
            
        return input_data, output_data, spec + topo_type, src_dst_pair_target_str, adj_matrix
    
    def compute_adjacency_matrix(self, flow_ids):
        num_flows = len(flow_ids)
        num_links = 1  # Only one link in this case
        adj_matrix = np.zeros((num_flows, num_links), dtype=np.float32)
        adj_matrix[:, 0] = 1  # Connect each flow to the single link
        
        return adj_matrix
    
class PathFctSldnSegment(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
    ):
        self.data_list = data_list
        self.dir_input = dir_input
        self.use_first_epoch_logic = True
        logging.info(
            f"call PathFctSldnSegment: data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}"
        )

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type, segment_id, busy_period = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])+f'_seg{segment_id}'
        
        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"
        
        # load feat
        feat_path=f"{dir_input_tmp}/feat{topo_type}_seg{segment_id}.npz"
        
        if not os.path.exists(feat_path) or self.use_first_epoch_logic:
            # busy_periods=np.load(f"{dir_input_tmp}/period{topo_type}.npy", allow_pickle=True)
            # busy_period=busy_periods[segment_id]
            fid=np.arange(busy_period[0], busy_period[1]+1)
            # fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
            sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
            fats_flowsim = np.load(f"{dir_input_tmp}/fat.npy")
            fsd_flowsim=np.load(f"{dir_input_tmp}/fsd.npy")
            
            sizes_flowsim=sizes_flowsim[fid]
            fats_flowsim=fats_flowsim[fid]
            fsd_flowsim=fsd_flowsim[fid]
            
            flows_fg =np.logical_and(fsd_flowsim[:,0]==src_dst_pair_target[0], fsd_flowsim[:,1]==src_dst_pair_target[1])
            assert np.sum(flows_fg)>0
            # Calculate inter-arrival times and adjust the first element
            fats_ia_flowsim=np.diff(fats_flowsim)
            fats_ia_flowsim=np.insert(fats_ia_flowsim, 0, 0)
            
            link_padding = np.zeros((len(fid), 6))
            for flow_idx in range(len(fid)):
                link_padding[flow_idx, fsd_flowsim[flow_idx,0]:fsd_flowsim[flow_idx,1]] = 1
            # Combine flow sizes and inter-arrival times into the input tensor
            input_data = np.column_stack((sizes_flowsim, fats_ia_flowsim, link_padding)).astype(np.float32)
            assert (input_data>=0.0).all()
            
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
            output_data = np.divide(fcts, i_fcts).reshape(-1, 1).astype(np.float32)[fid]
            assert (output_data>=1.0).all()
            output_data[~flows_fg] = PLACEHOLDER
            
            # np.savez(feat_path, input_data=input_data, output_data=output_data)
        else:
            feat=np.load(feat_path)
            input_data=feat["input_data"]
            output_data=feat["output_data"]
            
        return input_data, output_data, spec + topo_type, src_dst_pair_target_str