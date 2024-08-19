import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def exists(x):
    return x is not None

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(0))
        self.att.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """perform AXW: input [b, num_nodes, dim]"""
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC_Block(nn.Module):
    def __init__(self, in_features, out_features, p_dropout, bias=True, node_n=48, act=nn.Tanh):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.gc1 = GraphConvolution(in_features, out_features, node_n=node_n, bias=bias)
        self.gc2 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)

        self.bn1 = nn.BatchNorm1d(node_n * out_features)
        self.bn2 = nn.BatchNorm1d(node_n * out_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = act()

        self.res = nn.Conv1d(in_features, out_features, kernel_size=1) if in_features != out_features else nn.Identity()

    def forward(self, x):
        """
        in->GCB->GCB->in+out
        """
        y = x
        
        y = self.gc1(y)
        b, n, f = y.shape
        y = self.bn1(y.contiguous().view(b, -1)).contiguous().view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.contiguous().view(b, -1)).contiguous().view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + self.res(x.permute(0, 2, 1)).permute(0, 2, 1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DCTblock(nn.Module):
    def __init__(self, dct_dim, num_stages, t_dim, t_his_dim, hidden_dim, p_dropout=0.1, node_n=48, act=nn.Tanh, nc=True, gt_lead=False):
        """
        perform one time DCT in-between operation, consisting of <num_stages> GCN blocks: this is the <learning stage>
        """
        super(DCTblock, self).__init__()
        self.dct_dim = dct_dim
        self.num_stages = num_stages
        self.t_dim = t_dim
        self.t_his_dim = t_his_dim
        self.hidden_dim = hidden_dim
        self.gt_lead = gt_lead

        num_nodes = node_n if nc else node_n // 3

        ### graph based
        self.nc = nc
        if nc:  # after dct: [b, nc, dct_dim]
            self.encoder = GraphConvolution(dct_dim, hidden_dim, node_n=num_nodes)
        else:   # after dct: [b, n, c*dct_dim]
            self.encoder = GraphConvolution(3*dct_dim, hidden_dim, node_n=num_nodes)

        self.bn1 = nn.BatchNorm1d(num_nodes * hidden_dim)
        self.do = nn.Dropout(p_dropout)
        self.act_f = act()
        self.blocks = nn.ModuleList([GC_Block(
            hidden_dim, hidden_dim, p_dropout=p_dropout, node_n=num_nodes, act=act,
        ) for _ in range(num_stages)])

        if nc:
            self.decoder = GraphConvolution(hidden_dim, dct_dim, node_n=num_nodes)
        else:
            self.decoder = GraphConvolution(hidden_dim, 3*dct_dim, node_n=num_nodes)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        trans_m, itrans_m = get_dct_matrix(t_dim)
        trans_m = torch.from_numpy(trans_m).float()
        itrans_m = torch.from_numpy(itrans_m).float()
        register_buffer('trans_m', trans_m)
        register_buffer('itrans_m', itrans_m)

    def forward(self, x):
        """
        x: [B, T_in+T_pred, NC]
        return: [B, T_all, NC]
        """

        # dct processing
        x_pose_orig = x
        x = torch.matmul(self.trans_m[:self.dct_dim,:].unsqueeze(dim=0), x).permute(0, 2, 1)      # [B, NC, dct_dim]
        if not self.nc:
            x = rearrange(x, 'b (n c) d -> b n (c d)', c=3)     # [B, N, C*dct_dim]
        x_orig = x
        
        #########################################
        x = self.encoder(x)                                     # [B, NC, hid] / [B, N, hid]
        b, n, f = x.shape
        x = self.bn1(x.contiguous().view(b, -1)).contiguous().view(b, n, f)
        x = self.do(self.act_f(x))
        for i in range(self.num_stages):
            x = self.blocks[i](x) 
        x = self.decoder(x)                                     # [B, NC, dct_dim] / [B, N, C*dct_dim] 
        x = x + x_orig
        #########################################

        # back to pose space
        if not self.nc:
            x = rearrange(x, 'b n (c d) -> b (n c) d', c=3)     # [B, NC, dct_dim]

        x = x.permute(0, 2, 1)
        x = torch.matmul(self.itrans_m[:,:self.dct_dim].unsqueeze(dim=0), x)        # [B, t_all, NC]

        if self.gt_lead:
            x = torch.cat([x_pose_orig[:,:self.t_his_dim,:], x[:,self.t_his_dim:,:]], dim=1)

        return x

class MRN(nn.Module):
    def __init__(self, input_n, output_n, dct_dim, hidden_dim, inner_stage, outer_stage, p_dropout, \
        node_n=48, act=nn.Tanh, nc=True, gt_lead=False):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        NOTE: this is the motion refinement module
        """
        super(MRN, self).__init__()
        self.num_blocks = outer_stage
        self.input_n = input_n
        self.output_n = output_n

        self.blocks = []
        
        for i in range(self.num_blocks):
            self.blocks.append(
                DCTblock(
                    dct_dim=dct_dim, 
                    num_stages=inner_stage,
                    t_dim=input_n+output_n,
                    t_his_dim=input_n,
                    hidden_dim=hidden_dim, 
                    p_dropout=p_dropout, 
                    node_n=node_n, 
                    act=act,
                    nc=nc,
                    gt_lead=gt_lead,
                )
            )

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        """
        input: [B, T_in+T_out, N*C]
        output: [B, T_in+T_out, N*C]
        """
        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class CoMusion(nn.Module):
    def __init__(
        self,
        input_n,
        output_n,
        dct_dim, 
        gcn_dim,
        gcn_drop, 
        inner_stage, 
        outer_stage, 
        node_n,
        act=nn.Tanh,
        trans_dim=512,
        trans_drop=0.1,
        trans_ff_dim=1024,
        trans_num_heads=4, 
        trans_num_layers=8,
    ):
        super(CoMusion, self).__init__()

        self.input_process = nn.Linear(node_n, trans_dim)
        self.sequence_pos_encoder = PositionalEncoding(trans_dim, trans_drop)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(                      # NOTE: motion refinement module
            d_model=trans_dim,
            nhead=trans_num_heads,
            dim_feedforward=trans_ff_dim,
            dropout=trans_drop,
            activation='gelu',
            norm_first=False,
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=trans_num_layers)
        self.embed_timestep = TimestepEmbedder(trans_dim, self.sequence_pos_encoder)
        self.output_process = nn.Linear(trans_dim, node_n)
        self.MRN = MRN(input_n, output_n, dct_dim, gcn_dim, inner_stage, outer_stage, gcn_drop, node_n=node_n, act=act, nc=True, gt_lead=False) # NOTE: motion reconstruction module

    def forward(self, x, timesteps, x_0_history, x_self_cond=None, *args, **kwargs):
        # get time embedding
        time_emb = self.embed_timestep(timesteps)   # [1, bs, d] (ours, b 1 d)

        x = x.permute(1, 0, 2)                      # [t, b, d]
        x = self.input_process(x)                   # [t, b, d']

        # adding the timestep embed, transformer
        x = torch.cat((time_emb, x), axis=0)        # [t+1, b, d']
        x = self.sequence_pos_encoder(x)            # [t+1, b, d']
        x = self.seqTransEncoder(x)[1:]             # [t, b, d']
        x = self.output_process(x)                  # [t, b, d]
        
        # combine with x_0_history
        x = x.permute(1, 0, 2)
        x = torch.cat([x_0_history, x], dim=1)

        # GCN learning
        x = self.MRN(x)

        return x
