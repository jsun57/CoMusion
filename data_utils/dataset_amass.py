import os
import numpy as np
import pandas as pd
from .skeleton import SkeletonAMASS
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import hashlib
import math
import json
import zarr


class DatasetAMASS(Dataset):
    def __init__(self, mode, t_his, t_pred, segments_path=None, actions='all', augmentation=0, stride=1, transform=None, dtype='float32', *args, **kwargs):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        # dtype: use float32 as default
        assert dtype.lower() in ["float64", "float32"], "Only dtypes=float64/32 supported in this project."
        self.dtype = np.float64 if dtype.lower() == 'float64' else np.float32
        # folders if precomputed
        self.stats_folder = './auxiliar/datasets/amass/'
        self.precomputed_folder = './data/amass/precomputed/'
        self.segments_path = None if mode == 'train' else os.path.join(self.stats_folder, 'segments_test.csv')
        # augmentation and stride: belfusion style
        self.augmentation = augmentation
        self.stride = stride
        # meta info
        self.dict_indices = {} # dict_indices[dataset][file_idx] indicated idx where dataset-file_idx annotations start.
        self.metadata_class_idx = 0 # 0: dataset, 1: filename --> dataset is the class used for metrics computation
        if mode == 'train':
            self.idx_to_class = ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'EKUT','EyesJapanDataset',
                                 'KIT', 'PosePrior', 'TCDHands', 'TotalCapture', 'HumanEva', 'HDM05', 'SFU', 'MoSh']
        else:
            self.idx_to_class = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        # raw dataset
        self._prepare_data()     # just the skeleton
        self.annotations = self._read_annotations() # self.dict_indices and self.clip_idx_to_metadata avaliable
        self._prepare_segments() # self.segments and self.segment_idx_to_metadata available
        # transform: for normalization and augmentation
        self.transform = transform
        # generate dataset stat id
        if mode == 'train':
            self.stat_id, self.hased_id = self._get_hash_str()
        # mean_motion_per_class for CMD: keep info for test partition
        self.mean_motion_per_class = [0.00427221349493322, 0.008289838197001043, 0.0016145416139357026, 
                                      0.004560201420525195, 0.007548907591298325, 0.0052984837093390524, 
                                      0.007567679516515443]

    def _prepare_data(self):
        """
        amass prepare_data: for AMASS, only prepare skeleton
        """
        self.data_file = None
        self.datasets_split = {
            'train': ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'EKUT','EyesJapanDataset',
                      'KIT', 'PosePrior', 'TCDHands', 'TotalCapture', 'HumanEva', 'HDM05', 'SFU', 'MoSh'],
            'test': ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        }
        self.datasets = [x for x in self.datasets_split[self.mode]]
        self.num_actions = len(self.datasets)
        self.skeleton = SkeletonAMASS(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                                      joints_left=[1, 4, 7, 10, 13, 16, 18, 20],
                                      joints_right=[2, 5, 8, 11, 14, 17, 19, 21])
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(22) if x not in self.removed_joints]) # 22
        self.skeleton.remove_joints(self.removed_joints)

    def _read_annotations(self):
        """
        convert data dic to annotation list: []
        """
        anns_all = []
        self.dict_indices = {}          # dic[dataset] := dic[file_idx] := index
        self.clip_idx_to_metadata = []  # [(dataset, file_idx)]
        counter = 0

        for dataset in self.datasets:
            self.dict_indices[dataset] = {}

            z_poses = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses.zarr'), mode='r')
            z_trans = zarr.open(os.path.join(self.precomputed_folder, dataset, 'trans.zarr'), mode='r')
            z_index = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses_index.zarr'), mode='r')

            for file_idx in range(z_index.shape[0]):
                self.dict_indices[dataset][file_idx] = counter
                i0, i = z_index[file_idx]
                seq = z_poses[i0:i]
                seq[:, 1:] -= seq[:, :1]
                seq = seq[:, self.kept_joints, :]
                self.dict_indices[dataset][file_idx] = counter
                self.clip_idx_to_metadata.append((dataset, file_idx))
                counter += 1
                anns_all.append(seq.astype(self.dtype))
        return anns_all 

    def _generate_segments(self):
        """
        important: ensure each epoch visits all samples from the dataset.
        """
        assert self.clip_idx_to_metadata is not None, "idx_to_metadata must be initialized before generating segments"
        both = [((idx, init, init+self.t_total-1), self.clip_idx_to_metadata[idx]) for idx in range(len(self.annotations)) 
                                                        for init in range(0, self.annotations[idx].shape[0]-self.t_total+1)]
        # unzip to separate the segments from the metadata
        segments, segment_idx_to_metadata = list(zip(*both))
        return segments, segment_idx_to_metadata

    def _prepare_segments(self):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path)
            self.stride = 1             # next sample is one frame after
            self.augmentation = 0       # no left/right random shifting
        else:
            self.segments, self.segment_idx_to_metadata = self._generate_segments()

    def _load_annotations_and_segments(self, segments_path):
        """
        if segment path exists, can direct load it
        """
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        datasets, file_idces = list(df["dataset"].unique()), list(df["file_idx"].unique())
        
        segments = [(self.dict_indices[row["dataset"]][row["file_idx"]], 
                    row["pred_init"] - self.t_his, 
                    row["pred_init"] + self.t_pred - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["dataset"], row["file_idx"]) for i, row in df.iterrows()]

        return segments, segment_idx_to_metadata

    def _get_segment(self, i, init, end):
        """
        return shape [t_total, n_joints, n_dim]
        """
        assert init >= 0, "init point for segment must be >= 0"
        traj = self.annotations[i][init:end+1]
        return traj

    def _get_hash_str(self):
        trans = []
        base = [str(self.t_his), str(self.t_pred), str(self.stride), str(self.augmentation)]
        if self.transform is not None:
            trans = [str(self.transform.transforms[0].flip_prob), str(self.transform.transforms[0].mirror_prob), str(self.transform.transforms[0].rotate_prob)]
        to_hash = "_".join(tuple(base + trans)).replace('.', '')
        return to_hash, str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())

    def __len__(self):
        return len(self.segments) // self.stride 

    def __getitem__(self, index):
        segment_idx = int(self.stride * index + self.augmentation)
        if self.augmentation != 0:
            offset = np.random.randint(-self.augmentation, self.augmentation + 1)
            final_idx = max(0, min(segment_idx + offset, len(self.segments) - 1))
            (i, init, end) = self.segments[final_idx]
        else:
            (i, init, end) = self.segments[segment_idx]

        traj = self._get_segment(i, init, end)
        metadata = self.segment_idx_to_metadata[segment_idx]

        extra = {
            "sample_idx": index,
            "segment_idx": segment_idx,
            "clip_idx": i,
            "init": init,
            "end": end,
            "metadata": metadata,
            "act": self.class_to_idx[metadata[0]],
            "raw_traj": traj.copy()
        }

        if self.transform:
            traj = self.transform(traj)

        return traj, extra

    def iter_generator(self, step=25):
        """
        [1, t_all, n, c]; [1, 1]: just iterate on raw data
        """
        for sub, data_s in self.data.items():
            for act, seq in data_s.items():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj


def generate_amass_loss_weights(t_length, scale=10):
    """
    Structure-aware loss for amass
    """
    assert scale > 0 and scale != 1

    weights = [1,5,9,2,6,10,3,7,11,4,8,12,14,18,13,15,19,16,20,17,21]
    weights = np.repeat(weights, 3)  # 21x3=63

    chain = [[0], 
            [128.13, 38.05, 41.24, 13.91],              # left leg
            [128.68, 38.06, 40.78, 14.27],              # right leg
            [112.41, 114.3, 5.85, 21.91, 9.84],         # head
            [14.06, 11.88, 25.8, 25.78],                # left arm
            [14.51, 11.43, 25.87, 26.24]]               # right arm 

    new_chain = []
    for x in chain:
        s = sum(x)
        if s == 0:
            new_chain.append([0])
            continue
        new_x = []
        for i in range(len(x)):
            new_x.append((i+1)/(len(x))*math.log(sum(x[:i])+scale))
        new_chain.append(new_x)

    new_chain = [item for sublist in new_chain for item in sublist]

    s_weight = []
    for i in range(len(weights)):
        s_weight.append(new_chain[weights[i]])

    s_weight = np.asarray(s_weight)
    t_weight = np.ones(t_length)

    ret = s_weight[None,:] * t_weight[:,None]
    ret = ret / np.sum(ret) * 63 * t_length

    return torch.from_numpy(ret)
