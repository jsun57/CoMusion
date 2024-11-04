import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R


class DataAugmentation:
    def __init__(self, rotate_prob=0.0):
        self.rotate_prob = rotate_prob
                    
    def rotating(self, traj, prob=1):
        """
        traj: [t_all, J, 3]: apply random rotations with probability 1
        """
        rotation_axes = ['z'] # 'x' and 'y' not used because the person could be upside down
        for a in rotation_axes:
            if np.random.rand() < prob:
                degrees = np.random.randint(0, 360)
                r = R.from_euler(a, degrees, degrees=True).as_matrix().astype(np.float32)
                traj = (r @ traj.reshape((-1, 3)).T).T.reshape(traj.shape)
        return traj

    def __call__(self, sequence):
        """
        seq: [seq_len, J, 3]
        """
        sequence = self.rotating(sequence, prob=self.rotate_prob)

        return sequence


def calculate_stats(dataset, batch_size=2048, num_workers=0):
    """
    calculate mean, std, min, max of augmented dataset: calling once should be sufficient
    """
    stats_folder = dataset.stats_folder
    statistics_folder = os.path.join(stats_folder, "stats", dataset.stat_id)
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    mean_file = os.path.join(statistics_folder, 'mean.npy')
    std_file = os.path.join(statistics_folder, 'std.npy')
    min_file = os.path.join(statistics_folder, 'min.npy')
    max_file = os.path.join(statistics_folder, 'max.npy')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # calculate stats: should be done only once per config...
    n, c = dataset[0][0].shape[-2:]
    mean, std = torch.zeros((n, c)), torch.zeros((n, c))
    minv, maxv = torch.full((n, c), float('inf')), torch.full((n, c), float('-inf'))
    total_sum = torch.zeros((n, c))
    total_sum_sq = torch.zeros((n, c))
    count = 0
    for i, (seq, extra) in enumerate(dataloader):
        flat_seq = seq.reshape((-1, n, c))
        total_sum += torch.sum(flat_seq, dim=0)
        total_sum_sq += torch.sum(flat_seq ** 2 , dim=0) 
        batch_min = torch.min(flat_seq, dim=0)[0]
        batch_max = torch.max(flat_seq, dim=0)[0]
        minv = torch.minimum(minv, batch_min)
        maxv = torch.maximum(maxv, batch_max)
        count += flat_seq.shape[0]
    mean = total_sum / count
    std = torch.sqrt((total_sum_sq / count) - (mean ** 2))
    # save to file
    mean.numpy().dump(mean_file)
    std.numpy().dump(std_file)
    minv.numpy().dump(min_file)
    maxv.numpy().dump(max_file)
    return


def load_stats(dataset):
    mean = np.load(os.path.join(dataset.stats_folder, "stats", dataset.stat_id, 'mean.npy'), allow_pickle=True)
    std = np.load(os.path.join(dataset.stats_folder, "stats", dataset.stat_id, 'std.npy'), allow_pickle=True)
    minv = np.load(os.path.join(dataset.stats_folder, "stats", dataset.stat_id, 'min.npy'), allow_pickle=True)
    maxv = np.load(os.path.join(dataset.stats_folder, "stats", dataset.stat_id, 'max.npy'), allow_pickle=True)
    return mean, std, minv, maxv
