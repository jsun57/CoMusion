import yaml
import os
import torch
import torch.nn as nn

class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = 'cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = '/tmp' if test else 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        self.vis_dir = '%s/vis' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # common
        self.model_type = cfg.get('model_type', 'CoMusion')
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.eval_sample_num = cfg['eval_sample_num']
        self.dtype = cfg.get('dtype', 'float32')

        # # Diffusion F_theta network
        self.model_specs = cfg.get('model_specs', dict())
        self.node_n = self.model_specs.get('node_n', 48)
        self.act = self.model_specs.get('act', nn.Tanh)
        self.dct_dim = self.model_specs.get('dct_dim', 125)
        self.gcn_dim = self.model_specs.get('gcn_dim', 256)
        self.gcn_drop = self.model_specs.get('gcn_drop', 0.5)
        self.inner_stage = self.model_specs.get('inner_stage', 2)
        self.outer_stage = self.model_specs.get('outer_stage', 3)
        self.trans_dim = self.model_specs.get('trans_dim', 512)
        self.trans_drop = self.model_specs.get('trans_drop', 0.1)
        self.trans_ff_dim = self.model_specs.get('trans_ff_dim', 1024)
        self.trans_num_heads = self.model_specs.get('trans_num_heads', 4)
        self.trans_num_layers = self.model_specs.get('trans_num_layers', 8)

        # # Diffuser 
        self.diff_specs = cfg.get('diff_specs', dict())
        self.diffuse_steps = self.diff_specs.get('diffuse_steps', 10)
        self.loss_type = self.diff_specs.get('loss_type', 'l1')
        self.objective = self.diff_specs.get('objective', 'pred_x0')
        self.beta_schedule = self.diff_specs.get('beta_schedule', 'ours')
        self.div_k = self.diff_specs.get('div_k', 2)
        
        # # Data
        self.data_specs = cfg.get('data_specs', dict())
        self.dataset = self.data_specs.get('dataset', 'h36m')
        self.actions = self.data_specs.get('actions', 'all')
        self.augmentation = self.data_specs.get('augmentation', 0)
        self.stride = self.data_specs.get('stride', 1)
        self.multimodal_threshold = self.data_specs.get('multimodal_threshold', 0.5)
        self.data_aug = self.data_specs.get('data_aug', True)
        self.rota_prob = self.data_specs.get('rota_prob', 1.0)

        # # Learning
        self.learn_specs = cfg.get('learn_specs', dict())
        self.train_lr = self.learn_specs.get('train_lr', 1e-3)
        self.weight_decay = self.learn_specs.get('weight_decay', 0)
        self.train_epoch = self.learn_specs.get('train_epoch', 500)
        self.sched_policy = self.learn_specs.get('sched_policy', 'lambda')
        self.num_epoch_fix_lr = self.learn_specs.get('num_epoch_fix_lr', 100)
        self.batch_size = self.learn_specs.get('batch_size', 16)

        # # structure-aware loss
        self.st_loss_specs = cfg.get('st_loss_specs', dict())
        self.loss_weight_scale = self.st_loss_specs.get('loss_weight_scale', 10)

        # # Loss weight
        self.loss_weight_specs = cfg.get('loss_weight_specs', dict())
        self.history_weight = self.loss_weight_specs.get('history_weight', 1)
        self.future_weight = self.loss_weight_specs.get('future_weight', 20)

        # # Logging
        self.logging_specs = cfg.get('logging_specs', dict())
        self.model_id = self.logging_specs.get('model_id', 'sample')
        self.model_path = self.logging_specs.get('model_path', './results/sample/models')