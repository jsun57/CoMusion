import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import math
from .fid import FID
from einops import rearrange

def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def APD(pred, target, *args, t0=0, t=-1):
    """
    pred: [b, n_sample, t_pred, NC]; target: [b, t_pred, NC]; 
    """
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1: # only one sample => no APD possible
        return torch.tensor([0] * batch_size, device=pred.device)

    arr = pred.reshape(batch_size, n_samples, -1) # (batch_size, num_samples, others)
    dist = torch.cdist(arr, arr)
    dist_shape = dist.shape[-1]

    iu = np.triu_indices(dist_shape, 1) # symmetric matrix, only keep the upper ones, diagonal not considered
    pdist_shape = iu[0].shape[0]
    bs_indices = np.expand_dims(np.array([i for i in range(batch_size) for j in range(pdist_shape)]), 0) # we expand it to all batch_size
    values_mask = np.tile(iu, batch_size)
    final_mask = np.concatenate((bs_indices, values_mask), axis=0)

    # we filter only upper triangular values
    results = dist[final_mask].reshape((batch_size, pdist_shape)).mean(axis=-1)

    return results


def ADE(pred, target, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
    return dist.min(axis=-1).values


def FDE(pred, target, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))
    
    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1)[..., -1]
    return dist.min(axis=-1).values


def MMADE(pred, target, gt_multi, *args, t0=0, t=-1): # memory efficient version
    """
    pred: [b, sample_num, t_pred, NC]
    target[b, t_pred, NC] 
    """
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size): # gt_multi[i] [num_similar, t_pred, nc];  pred[i]: [num_sample, t_pred, nc]
        n_gts = gt_multi[i].shape[0]
        if n_gts == 1:
            results[i] = float('nan')
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)   # [1, num_sample, t_pred, nc]
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1) # [num_similar, 1, t_pred, nc]

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
        results[i] = dist.min(axis=-1).values.mean()

    return results


def MMFDE(pred, target, gt_multi, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size, ))
    for i in range(batch_size):
        n_gts = gt_multi[i].shape[0]
        if n_gts == 1:
            results[i] = float('nan')
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1)[..., -1]
        results[i] = dist.min(axis=-1).values.mean()

    return results


def APDE(curr_apds, gt_apds):
    """
    input: current batch apds [b, ]
    input: gt apds [b, ]: if zero, ignore
    return: [b, ], none indicating gt apd is 0
    """
    nonzero_idxs = torch.nonzero(gt_apds)
    zero_idxs = torch.nonzero(gt_apds.eq(0))
    ret = abs(curr_apds - gt_apds)
    ret[zero_idxs] = float('nan')
    return ret


def CMD(val_per_frame, val_ref):
    T = len(val_per_frame) + 1
    return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])


def CMD_helper(pred, extra, histogram_data, all_obs_classes):
    """
    pred: [b, num_s, t_pred, NC] -> [batch, num_s, t_pred, joint, 3]
    """
    pred_flat = rearrange(pred, '... (n c) -> ... n c', c=3)
    motion = (torch.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1)).mean(axis=1).mean(axis=-1)    

    histogram_data.append(motion.cpu().detach().numpy())    
    classes = extra['act'].numpy()
    all_obs_classes.append(classes)
    
    return


def CMD_pose(dataset, histogram_data, all_obs_classes):
    """
    TODO: validate this function
    """
    ret = 0
    obs_classes = np.concatenate(all_obs_classes, axis=0)
    motion_data = np.concatenate(histogram_data, axis=0)
    motion_data_mean = motion_data.mean(axis=0)

    motion_per_class = np.zeros((dataset.num_actions, motion_data.shape[1]))
    # CMD weighted by class
    for i, (name, class_val_ref) in enumerate(zip(dataset.idx_to_class, dataset.mean_motion_per_class)):
        mask = obs_classes == i
        if mask.sum() == 0:
            continue
        motion_data_mean = motion_data[mask].mean(axis=0)
        motion_per_class[i] = motion_data_mean
        ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / obs_classes.shape[0])
    return ret


def FID_helper(pred, gt, classifier_for_fid, all_pred_activations, all_gt_activations, all_pred_classes, all_gt_classes):
    """
    pred: [b, sample_num, t_pred, NC]
    gt: [b, t_pred, NC]
    """
    b, s = pred.shape[0], pred.shape[1]

    pred_ = rearrange(pred, 'b s t d -> (b s) d t') # [bs, nc, t_pred])
    gt_ = rearrange(gt, 'b t d -> b d t') # [b, nc, t_pred]

    pred_activations = classifier_for_fid.get_fid_features(motion_sequence=pred_).cpu().data.numpy()
    gt_activations = classifier_for_fid.get_fid_features(motion_sequence=gt_).cpu().data.numpy()

    all_pred_activations.append(pred_activations)
    all_gt_activations.append(gt_activations)
    
    pred_classes = classifier_for_fid(motion_sequence=pred_.float()).cpu().data.numpy().argmax(axis=1)
    # recover the batch size and samples dimension
    pred_classes = pred_classes.reshape([b, s])
    gt_classes = classifier_for_fid(motion_sequence=gt_.float()).cpu().data.numpy().argmax(axis=1)
    # append to the list
    all_pred_classes.append(pred_classes)
    all_gt_classes.append(gt_classes)

    return


def FID_pose(all_gt_activations, all_pred_activations):
    ret = 0
    ret = FID(np.concatenate(all_gt_activations, axis=0), np.concatenate(all_pred_activations, axis=0))
    return ret
