import numpy
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
import  torchvision
from modules import Encoder, LayerNorm
from einops import rearrange
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class HeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, num_tokens, input_dim, attn_dropout, ff_dropout, args):
        super().__init__()
        self.dim = args.d_model
        self.depth = args.num_hidden_layers
        self.heads = args.num_attention_heads
        self.dim_head = 2*args.num_attention_heads
        # self.embeds = nn.Embedding(num_tokens, args.embed_dim)
        self.layers = nn.ModuleList([])
        self.embeds = nn.Linear(input_dim, args.embed_dim)

        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(args.embed_dim, HeadAttention(args.embed_dim, heads=self.heads, dim_head=self.dim_head, dropout=attn_dropout))),
                Residual(PreNorm(args.embed_dim, FeedForward(args.embed_dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        # print(self.embeds)
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x
class DeformConv(nn.Module):
    def __init__(self, batch, channel, height, width):
        super(DeformConv, self).__init__()
        self.weight = torch.rand(1, channel, 3, 3).to(device)
        self.offset = torch.rand(batch, 18, height, width).to(device)

    def forward(self, input):
        input = input.unsqueeze(1)

        # print(self.offset.shape)
        input = torchvision.ops.deform_conv2d(input=input, weight=self.weight, offset=self.offset,padding=1)
        input = input.squeeze(1)
        return input

class DownConv(nn.Module):
    def __init__(self, c_in):
        super(DownConv, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.linear_out = nn.Linear(33,64)
        self.linear_out2 = nn.Linear(65, 128)

    def forward(self, x):

        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        l = x.size(-1)
        if(l==33):
            x = self.linear_out(x)
        else:
            x = self.linear_out2(x)
        x = x.transpose(1, 2)

        return x

class Time_DownConv(nn.Module):
    def __init__(self, c_in):
        super(Time_DownConv, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.linear_out = nn.Linear(33,64)
        self.linear_out2 = nn.Linear(65, 128)

    def forward(self, x):

        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        l = x.size(-1)
        if(l==33):
            x = self.linear_out(x)
        else:
            x = self.linear_out2(x)
        x = x.transpose(1, 2)

        return x