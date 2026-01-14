import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.input_size = config.seq_len
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.pred_len = config.pred_len
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.input_size * self.pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1, self.pred_len, self.input_size)
