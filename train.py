import os
import sys
import csv
import math
import time
import argparse
import numpy as np
import pandas as pd
from einops import rearrange
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform

sys.path.append(os.getcwd())
from utils import *
from models.load_models import get_model
from models.fid_classifier import ClassifierForFID
from models.GaussianDiffusion import GaussianDiffusion
from utils.metrics import CMD_helper, CMD_pose
from utils.metrics import FID_helper, FID_pose
from utils.metrics import APD, APDE, ADE, FDE, MMADE, MMFDE
from data_utils.transforms import calculate_stats, load_stats
from data_utils.dataset_h36m import DatasetH36M, generate_h36_loss_weights
from data_utils.dataset_amass import DatasetAMASS, generate_amass_loss_weights
from data_utils.transforms import DataAugmentation


def generate_loss_weight(cfg):
    if cfg.dataset == 'h36m':
        gen_weight = generate_h36_loss_weights
    else:
        gen_weight = generate_amass_loss_weights
    # history recon weight
    in_weights = gen_weight(cfg.t_his, scale=cfg.loss_weight_scale)
    # future prediction weight
    out_weights = gen_weight(cfg.t_pred, scale=cfg.loss_weight_scale)
    loss_weights = torch.cat((in_weights, out_weights), dim=0)
    return loss_weights


class Trainer(object):
    def __init__(
        self,
        dataset,
        diffusion_model,
        cfg,
        train_batch_size=16,
        train_lr=1e-4,
        weight_decay=0,
        actions='all',
    ):
        super().__init__()

        self.model = diffusion_model
        self.device = next(self.model.parameters()).device

        self.cfg = cfg

        self.batch_size = train_batch_size
        self.input_n = cfg.t_his
        self.output_n = cfg.t_pred
        self.dtype = torch.float32 if self.cfg.dtype == 'float32' else torch.float64
        
        # dataset and dataloader initialization
        transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
        test_transform = None
        stat_dataset = dataset('train', self.input_n, self.output_n, augmentation=cfg.augmentation, stride=cfg.stride, transform=transform, dtype=cfg.dtype)
       
        # info saving
        stats_folder = os.path.join('auxiliar/datasets/', cfg.dataset, "stats", stat_dataset.stat_id)
        mmapd_path = os.path.join('auxiliar/datasets/', cfg.dataset, 'mmapd_GT.csv')

        if not os.path.exists(stats_folder) or len(os.listdir(stats_folder)) == 0:
            print('Calculating stats...')
            calculate_stats(stat_dataset)
        else:
            print('Stats precomputed.')
        print('Loading stats...')
        
        self.mean, self.std, self.minv, self.maxv = load_stats(stat_dataset)
        self.mean_torch = torch.from_numpy(self.mean[1:,:]).to(self.device).to(self.dtype)
        self.std_torch = torch.from_numpy(self.std[1:,:]).to(self.device).to(self.dtype)
        self.minv_torch = torch.from_numpy(self.minv[1:,:]).to(self.device).to(self.dtype)
        self.maxv_torch = torch.from_numpy(self.maxv[1:,:]).to(self.device).to(self.dtype)

        print('Preparing datasets...')
        self.train_dataset = dataset('train', self.input_n, self.output_n, augmentation=cfg.augmentation, stride=cfg.stride, transform=transform, dtype=cfg.dtype)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # NOTE: test partition: can be seen only once
        self.eval_dataset = dataset('test', self.input_n, self.output_n, augmentation=0, stride=1, transform=None, dtype=cfg.dtype)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        # multimodal GT
        print('Calculating mmGT...')
        self.multimodal_traj = self.get_multimodal_gt()

        # APDE computation loading
        self.mmapd = self.get_mmapd(mmapd_path)

        # FID classifier
        if self.cfg.dataset == 'h36m':
            print('Loading FID classifier...')
            self.classifier_for_fid = self.get_classifier()
        else:
            print('No FID classifier available...')
            self.classifier_for_fid = None            

        # optimizer
        self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=(0.9, 0.99), weight_decay=weight_decay)    # weight_decay is 0, same as Adam Pytorch Implementation
        self.scheduler = get_scheduler(self.opt, policy=cfg.sched_policy, nepoch_fix=cfg.num_epoch_fix_lr, nepoch=cfg.train_epoch)

        # epoch counter state
        self.epoch = 0
        self.train_loss_list = []
        print('Trainer initialization done.')

    def save(self, to_save_path):
        data = {
            'epoch': self.epoch,
            'train_loss_list': self.train_loss_list,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.scheduler.state_dict(),
        }
        torch.save(data, to_save_path)
        return

    def load(self, to_load_path):
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.epoch = data['epoch']
        self.train_loss_list = data['train_loss_list']
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['sched'])
        print(">>> finish loading model ckpt from path '{}'".format(to_load_path))
        return

    def train(self):
        self.model.train()
        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_loss_info = {}
        for traj_np, extra in self.train_dataloader:
            traj = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1).to(self.device).to(self.dtype)   # [b, t_total, 16x3]
            loss, loss_info = self.model(traj, None, div_k=self.cfg.div_k, uncond=True, mmgt=None)
            for key, value in loss_info.items():
                if key not in epoch_loss_info:
                    epoch_loss_info[key] = value
                else:
                    epoch_loss_info[key] += value
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            epoch_iter += 1

        self.scheduler.step()
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_loss_info.items():
            epoch_loss_info[key] /= epoch_iter
        lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return lr, epoch_loss, epoch_loss_info, dt 

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        all_data = []
        for i, (data, extra) in enumerate(self.eval_dataloader):    # [batch_size, t_all, num_joints, dim]
            data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        all_start_pose = all_data[:,self.input_n-1,:]
        pd = squareform(pdist(all_start_pose))
        traj_gt_arr = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(pd[i] < self.cfg.multimodal_threshold)
            traj_gt_arr.append(torch.from_numpy(all_data[ind][:, self.input_n:, :]).to(self.dtype))
        return traj_gt_arr

    def get_mmapd(self, mmapd_path):
        df = pd.read_csv(mmapd_path)
        mmapds = torch.as_tensor(list(df["gt_APD"]))
        return mmapds

    def get_classifier(self):
        classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
            output_size=15, use_noise=None, device=self.device, dtype=self.dtype).to(self.device)                    
        classifier_path = os.path.join("./auxiliar", "h36m_classifier.pth")
        classifier_state = torch.load(classifier_path, map_location=self.device)
        classifier_for_fid.load_state_dict(classifier_state["model"])
        classifier_for_fid.eval()
        return classifier_for_fid

    def get_prediction(self, data, act, sample_num, uncond, use_ema=True, concat_hist=False):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        traj = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1).to(self.device).to(self.dtype)    # [b, t_total, 16x3]

        # process x_0_history: [b*sample_num, t_pred, nc]
        x_0_history = torch.repeat_interleave(traj[:,:-self.output_n,:], sample_num, dim=0)
        total_sample_num = x_0_history.shape[0]
        Y = self.model.sample(x_0_history, None, batch_size=total_sample_num, clip_denoised=False, uncond=uncond)       # [b*sample_num, t_pred, nc]

        if concat_hist:
            Y = torch.cat((x_0_history, Y), dim=1)
        Y = Y.contiguous()

        return Y

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()

        def get_gt(data, input_n):
            gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            return gt[:, input_n:, :]

        # all quantitative results in paper
        stats_func = {'APD': APD, 'APDE': APDE, 'ADE': ADE,'FDE': FDE, 'MMADE': MMADE, 'MMFDE': MMFDE, 'FID': None, 'CMD': None}
        stats_names = list(stats_func.keys())
        stats_meter = {x: AverageMeterTorch() for x in stats_names}

        histogram_data = []
        all_pred_activations = [] # for FID. We need to compute the activations of the predictions
        all_gt_activations = [] # for FID. We need to compute the activations of the GT
        all_pred_classes = []
        all_gt_classes = []
        all_obs_classes = []

        counter = 0
        t = time.time()
        for i, (data, extra) in enumerate(self.eval_dataloader):   
            gt = get_gt(data, self.input_n).to(self.device).to(self.dtype)                  # [batch_size, t_pred, NC]
            pred = self.get_prediction(data, extra['act'], sample_num=self.cfg.eval_sample_num, uncond=True, concat_hist=False).detach()

            pred = rearrange(pred, '(b s) ... -> b s ...', b=gt.shape[0])
            gt_multi = self.multimodal_traj[counter:counter+gt.shape[0]]
            gt_multi = [t.to(self.device) for t in gt_multi]
            gt_apd = self.mmapd[counter:counter+gt.shape[0]].to(self.device)
            
            for stats in stats_names:
                if stats not in ('APDE', 'FID', 'CMD'):
                    val = stats_func[stats](pred, gt, gt_multi)
                    stats_meter[stats].update(val)

            # calculate APDE
            apde = stats_func['APDE'](stats_meter['APD'].raw_val, gt_apd)
            stats_meter['APDE'].update(apde)

            # calculate CMD, FID
            CMD_helper(pred, extra, histogram_data, all_obs_classes)
            if self.cfg.dataset == 'h36m':
                FID_helper(pred, gt, self.classifier_for_fid, all_pred_activations, all_gt_activations, all_pred_classes, all_gt_classes)

            counter += data.shape[0]

            print('-' * 80)
            print('Num in multi_GT: ', gt_multi[0].shape[0])
            for stats in stats_names:
                str_stats = f'{counter-data.shape[0]:04d} {stats:<6}: ' + f'({stats_meter[stats].avg:.4f})'
                print(str_stats)

        # postprocess CMD, FID
        cmd_val = CMD_pose(self.eval_dataset, histogram_data, all_obs_classes)
        stats_meter['CMD'].direct_set_avg(cmd_val)

        if self.cfg.dataset == 'h36m':
            fid_val = FID_pose(all_gt_activations, all_pred_activations)
            stats_meter['FID'].direct_set_avg(fid_val)

        return stats_meter

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        return {x: y.avg for x, y in stats_dic.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    cfg = Config(args.cfg, test=args.test)
    set_global_seed(args.seed)
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    """parameter"""
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    node_n = cfg.node_n

    """data"""
    if cfg.dataset == 'h36m':
        dataset_cls = DatasetH36M
    else:
        dataset_cls = DatasetAMASS
    action = 'all'

    """loss weight"""
    loss_weights = generate_loss_weight(cfg)    # [t_all/r_pred, NxC]

    """model"""
    model = get_model(cfg).to(dtype).to(device)
    diffuser = GaussianDiffusion(
        model=model,
        cfg=cfg,
        future_motion_size=(t_pred, node_n), # [T_pred, N*C=num_nodes]
        timesteps=cfg.diffuse_steps,
        loss_type=cfg.loss_type,
        objective=cfg.objective,
        beta_schedule=cfg.beta_schedule,
        history_weight=cfg.history_weight,
        future_weight=cfg.future_weight,
        st_loss_weight=loss_weights,
    ).to(dtype).to(device)

    """trainer"""
    trainer = Trainer(
        dataset=dataset_cls,
        diffusion_model=diffuser,
        train_batch_size=cfg.batch_size,
        train_lr=cfg.train_lr,
        weight_decay=cfg.weight_decay,
        actions=action,
        cfg=cfg,
    )

    start_epoch = 0
    print(">>> model on:", device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # For testing only
    if args.test:
        to_load_path = os.path.join(cfg.model_path, cfg.model_id + '_mdl.pth')
        data = torch.load(to_load_path, map_location=device)
        trainer.model.load_state_dict(data['model'])
    else:
        # For continuous training
        if args.load:
            file_name = 'ckpt_' + cfg.id + '.pth.tar'        
            trainer.load(os.path.join(cfg.model_dir, file_name))
            start_epoch = trainer.epoch

        # Training
        for epoch in range(start_epoch, cfg.train_epoch):
            ret_log = np.array([epoch + 1])
            head = np.array(['epoch'])
            lr, epoch_loss, epoch_loss_info, dt = trainer.train()

            print(">>> epoch: ", epoch)

            ret_log = np.append(ret_log, [lr, dt, epoch_loss])
            head = np.append(head, ['lr', 'dt', 't_l'])

            for key, value in epoch_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            # update log file and save checkpoint
            is_create = False
            if not args.load:
                if epoch == 0:
                    is_create = True
            save_csv_log(cfg, head, ret_log, is_create, file_name=cfg.id + '_log')
            
            # checkpoint, info saving
            file_name = 'ckpt_' + cfg.id + '.pth.tar'
            save_ckpt(cfg, trainer, file_name=file_name)


    print('Compute final stats...')
    stats = trainer.evaluation()
    print(stats)


    with open('%s/eval_stats.csv' % (cfg.result_dir), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    print('Done.')


if __name__ == '__main__':
    main()