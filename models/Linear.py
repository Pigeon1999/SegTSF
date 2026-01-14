import torch.nn as nn

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]