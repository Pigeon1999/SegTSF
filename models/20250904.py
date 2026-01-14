import torch 
import torch.nn as nn
import torch.nn.functional as F

# RevIN
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class BlockSharedLinear(nn.Module):
    def __init__(self, period, seq_len, pred_len):
        super(BlockSharedLinear, self).__init__()
        self.period = period
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.seg_num_x = self.seq_len // self.period
        self.seg_num_y = self.pred_len // self.period
        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(-1, self.seg_num_x, self.period).permute(0, 2, 1)
        y = self.linear(x)
        y = y.reshape(B, C, -1)
        return y
       
class BTLowRankHead(nn.Module):
    def __init__(self, L, H, k=24, r=48):
        super().__init__()
        # 채널 공유 conv: (B*C,1,L)로 바꿔 1채널 conv 적용 → 파라미터 k개
        self.conv = nn.Conv1d(1, 1, kernel_size=(k//2) * 2 + 1, padding=k//2, bias=False)
        self.p1 = nn.Linear(L, r, bias=False)
        self.p2 = nn.Linear(r, H, bias=True)
        nn.init.zeros_(self.p2.bias)

    def forward(self, x):              # x: (B,C,L)
        B, C, L = x.shape
        y = x.reshape(B*C, 1, L)
        y = self.conv(y).reshape(B, C, L)   # 근접(대각선) 보정
        return self.p2(self.p1(y))          # 전역(저랭크) 재조합

    
class Stream(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(Stream, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = 24

        self.fc1 = BlockSharedLinear(period=48, seq_len=self.seq_len*2, pred_len=self.seq_len*2)

        self.pool = nn.AvgPool1d(kernel_size=2)
        self.ln = nn.LayerNorm(seq_len)

        self.fc2 = BlockSharedLinear(period=self.period_len, seq_len=self.seq_len, pred_len=self.seq_len)
        self.get_output = BTLowRankHead(L=self.seq_len, H=self.pred_len)

    def forward(self, x):
        diff = x[:, :, 1:] - x[:, :, :-1]
        diff = F.pad(diff, (1, 0), mode='replicate')  # Δx
        
        output = torch.stack([x, diff], dim=-1).reshape(x.size(0), x.size(1), -1)
        output = self.fc1(output)

        output = self.pool(output)
        output = self.ln(output)

        output = self.fc2(output)
        output = self.get_output(output)

        return output
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Default 336
        self.pred_len = configs.pred_len # Default 96
        self.channels = configs.channels

        # Normalization
        self.revin = RevIN(self.channels, affine=True, subtract_last=False)

        # Stream
        self.stream = Stream(self.seq_len, self.pred_len, self.channels)

    def forward(self, x, *args):
        # Shape --> (Batch, Input, Channel)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)
        
        output = self.stream(x)
        
        output = output.permute(0, 2, 1)
        output = self.revin(output, "denorm")
        return output