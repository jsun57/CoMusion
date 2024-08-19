from models.CoMusion import CoMusion
import torch.nn as nn

def get_model(cfg):
    assert cfg.model_type == 'CoMusion'
    model = CoMusion(
        input_n=cfg.t_his,
        output_n=cfg.t_pred,
        dct_dim=cfg.dct_dim, 
        gcn_dim=cfg.gcn_dim,
        inner_stage=cfg.inner_stage, 
        outer_stage=cfg.outer_stage, 
        gcn_drop=cfg.gcn_drop, 
        node_n=cfg.node_n,
        act=eval(cfg.act),
        trans_dim=cfg.trans_dim,
        trans_drop=cfg.trans_drop,
        trans_ff_dim=cfg.trans_ff_dim,
        trans_num_heads=cfg.trans_num_heads, 
        trans_num_layers=cfg.trans_num_layers,
    )
    return model
