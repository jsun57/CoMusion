from __future__ import absolute_import
import logging
import os
import torch
import pandas as pd
import numpy as np

class AverageMeterTorch(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_val = 0 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        """
        val: [eval_batch_size], can contains nans
        """
        self.raw_val = val
        self.val = torch.nan_to_num(val)
        self.sum += sum(self.val).item()
        self.count += torch.count_nonzero(self.val).item()
        if self.count != 0:
            self.avg = self.sum / self.count 

    def direct_set_avg(self, val):
        self.raw_val = val 
        self.val = val
        self.avg = val
        self.sum = val
        self.count = 1


def save_csv_log(cfg, head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = cfg.log_dir + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(cfg, trainer, file_name='ckpt_CoMusion.pth.tar'):
    file_path = os.path.join(cfg.model_dir, file_name)
    trainer.save(file_path)

