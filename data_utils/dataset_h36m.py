import os
import numpy as np
import pandas as pd
from .skeleton import SkeletonH36M
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import hashlib
import math
import json


class DatasetH36M(Dataset):
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
        self.stats_folder = './auxiliar/datasets/h36m/'
        self.segments_path = None if mode == 'train' else os.path.join(self.stats_folder, 'segments_test.csv')
        # augmentation and stride: belfusion style
        self.augmentation = augmentation
        self.stride = stride
        # meta info
        self.dict_indices = {} # dict_indices[subject][action] indicated idx where subject-action annotations start.
        self.metadata_class_idx = 1 # 0: subject, 1: action --> action is the class used for metrics computation
        self.idx_to_class = ['Directions', 'Discussion', 'Eating', 'Greeting', 
                            'Phoning', 'Posing', 'Purchases', 'Sitting', 
                            'SittingDown', 'Smoking', 'Photo', 'Waiting', 
                            'Walking', 'WalkDog', 'WalkTogether']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        # raw dataset
        self._prepare_data()     # self.data: dic[subject][action]: entire sequence
        self.annotations = self._read_annotations() # self.dict_indices and self.clip_idx_to_metadata avaliable
        self._prepare_segments() # self.segments and self.segment_idx_to_metadata available
        # transform: for normalization and augmentation
        self.transform = transform
        # generate dataset stat id
        if mode == 'train':
            self.stat_id, self.hash_id = self._get_hash_str() 
        # mean_motion_per_class for CMD
        self.mean_motion_per_class = [0.004528946212615328,  0.005068199383505345,  0.003978791804673771, 
                                      0.005921345536787865,  0.003595039379111546,  0.004192961478268034, 
                                      0.005664689143238568,  0.0024945400286369122, 0.003543066357658834, 
                                      0.0035990843311130487, 0.004356865838457266,  0.004219841185066826, 
                                      0.007528046315984569,  0.007054820734533077,  0.006751761745020258]

    def _prepare_data(self):
        self.data_file = os.path.join('./data', 'data_3d_h36m.npz')
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [9, 11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = SkeletonH36M(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                              16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                     joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                     joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self._process_data()

    def _process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            self.num_actions = len(self.actions)
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        else:
            self.num_actions = 15
        for subject in data_f.keys():
            for action in data_f[subject].keys():
                seq = data_f[subject][action][:, self.kept_joints, :]
                seq[:, 1:] -= seq[:, :1]
                data_f[subject][action] = seq
        self.data = data_f

    def _read_annotations(self):
        """
        convert data dic to annotation list: []
        """
        anns_all = []
        self.dict_indices = {}          # dic[subject] := dic[action] := index
        self.dict_indices_inv = {}
        self.clip_idx_to_metadata = []  # [(subject, action)]
        counter = 0
        for subject in self.data:
            self.dict_indices[subject] = {}
            for action in self.data[subject]:
                self.dict_indices[subject][action] = counter
                self.dict_indices_inv[counter] = (subject, action)
                self.clip_idx_to_metadata.append((subject, action.split(" ")[0]))       # only action raw name 
                counter += 1
                anns_all.append(self.data[subject][action].astype(self.dtype)) #  [seq_all_length, joint_num, 3]
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
            # only for testing
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
        subjects, actions = list(df["subject"].unique()), list(df["action"].unique())
        
        segments = [(self.dict_indices[row["subject"]][row["action"]], 
                    row["pred_init"] - self.t_his, 
                    row["pred_init"] + self.t_pred - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["subject"], row["action"].split(" ")[0]) for i, row in df.iterrows()]

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
            trans = [str(self.transform.transforms[0].rotate_prob)]
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
            "act": self.class_to_idx[metadata[1]],
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


def generate_h36_loss_weights(t_length, scale=10):
    """
    Structure-aware loss for h36m
    """
    assert scale > 0 and scale != 1

    weights = np.arange(16) + 1
    weights = np.repeat(weights, 3)

    chain = [[0], 
            [118.9, 45.17, 44.7],           # right leg
            [134.79, 45.17, 44.7],          # left leg
            [114.7, 25.55, 11.71, 11.5],    # head
            [15.05, 28.2, 24.99],           # left arm
            [15.05, 28.2, 24.99]]           # right arm

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
    ret = ret / np.sum(ret) * 48 * t_length

    return torch.from_numpy(ret)
