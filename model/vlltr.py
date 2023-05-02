import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        factory_kwargs = {'dtype': torch.float16}
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1q = nn.LayerNorm(dim, **factory_kwargs)
        self.norm1k = nn.LayerNorm(dim, **factory_kwargs)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, qx: torch.Tensor, kx: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # qx: [Bq, 1, C]    kx: [Bk, Nk, C]
        # key_padding_mask: [Bk, Nk] (mask==1 ==> '-inf')
        # output: [Bq, Bk, C]
        assert qx.shape[-1] == kx.shape[-1] and qx.shape[1] == 1
        Bq, _, C = qx.shape
        Bk, Nk, _ = kx.shape
        q = self.wq(self.norm1q(qx)).reshape(Bq, 1, self.num_heads, C //
                                             self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(self.norm1k(kx)).reshape(Bk, Nk, self.num_heads, C //
                                             self.num_heads).permute(0, 2, 1, 3)
        v = kx.unsqueeze(1)
        #  q: [Bq, num_heads,  1, C // num_heads]
        # kv: [Bk, num_heads, Nk, C // num_heads]
        # attn: [Bq, Bk, num_heads, Nk]
        attn = torch.einsum('qhoc,khnc->qkhn', q, k) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(2), float('-inf'),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('khnc,qkhn->qkhc', v, attn).reshape(Bq, Bk, C)

        return x


class VLLTR(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 op_type='two_branch', num_classes=0, use_constant_norm=False, v_detach=False):
        super().__init__()
        factory_kwargs = {'dtype': torch.float16}
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.op_type = op_type
        self.use_constant_norm = use_constant_norm
        self.v_detach = v_detach
        if self.op_type == 'concat':
            self.fc = nn.Linear(in_features=dim * 2, out_features=1, bias=True, **factory_kwargs)
        elif self.op_type == 'add':
            self.fc = nn.Linear(in_features=dim, out_features=1, bias=True, **factory_kwargs)
        elif self.op_type == 'cosine':
            self.fc = None
        elif self.op_type == 'two_branch':
            self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            self.visual_fc = nn.Sequential(
                nn.Linear(dim, 4 * dim, **factory_kwargs),
                nn.ReLU(),
                nn.Linear(4 * dim, num_classes, **factory_kwargs))
        else:
            self.fc = None

    def forward(self, qx: torch.Tensor, kx: torch.Tensor, key_padding_mask: torch.Tensor = None, logit_scale=None):
        # qx: [Bq, 1, C]    kx: [Bk, Nk, C]
        # v: [Bq, Bk, C]
        v = self.attn(qx, kx, key_padding_mask=key_padding_mask)
        if self.op_type == 'concat':
            x = qx.expand(qx.shape[0], kx.shape[0], qx.shape[-1])
            x = torch.cat((x, v), dim=-1)  # [Bq, Bk, 2*C]
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'cosine':
            if logit_scale is not None:
                qx_ = F.normalize(qx, p=2, dim=-1)
                if self.v_detach:
                    v_ = v / (v.norm(dim=-1, keepdim=True).detach())
                else:
                    v_ = F.normalize(v, p=2, dim=-1)
                x = torch.einsm('qkc,qoc->qk', v_, qx_) * logit_scale.exp()
            else:
                x = torch.einsum('qkc,qoc->qk', v, qx)
        elif self.op_type == 'add':
            x = qx.expand(qx.shape[0], kx.shape[0], qx.shape[-1]) + v
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'two_branch':
            x1 = self.visual_fc(qx.squeeze(1))

            if logit_scale is not None:
                if self.use_constant_norm:
                    qx_ = F.normalize(qx, p=2, dim=-1)
                    v_ = v / 21.1578
                    x2 = torch.einsum('qkc,qoc->qk', v_, qx_) * logit_scale.exp()
                else:
                    qx_ = F.normalize(qx, p=2, dim=-1)
                    if self.v_detach:
                        v_ = v / (v.norm(dim=-1, keepdim=True).detach())
                    else:
                        v_ = F.normalize(v, p=2, dim=-1)
                    x2 = torch.einsum('qkc,qoc->qk', v_, qx_) * logit_scale.exp()
            else:
                x2 = torch.einsum('qkc,qoc->qk', v, qx)

            return x1, x2

        return x.squeeze(-1)

