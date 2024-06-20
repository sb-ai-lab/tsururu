import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from tsururu.models.torch_based.layers.Embed import DataEmbedding
from tsururu.models.torch_based.layers.Conv_Block import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, (length - (self.seq_len + self.pred_len)), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class TimesNet_NN(nn.Module):
    def __init__(self, seq_len, pred_len, e_layers, enc_in, d_model, d_ff, num_kernels, top_k, c_out, dropout, embed, freq):
        super(TimesNet_NN, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k
        self.c_out = c_out
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(self.e_layers)])
        
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1) 

        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model)

        dec_out = self.projection(enc_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
