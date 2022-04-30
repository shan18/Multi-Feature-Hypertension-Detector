import torch
from torch import nn


class HypertensionDetectorBiLSTM(nn.Module):
    def __init__(self, hidden_dim, seq_meta_len, n_layers, fc_dim, dropout, device):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.seq_meta_fc = nn.Linear(seq_meta_len, hidden_dim)
        self.rnn = nn.LSTM(1, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, seq, seq_meta):
        """Input shapes

        seq: [batch_size, seq_length]
        seq_meta: [batch_size, seq_meta_len]
        """

        batch_size, seq_len = seq.shape
        seq = seq.unsqueeze(-1).permute(1, 0, 2)  # [seq_len, batch_size, 1]

        seq_meta = self.seq_meta_fc(seq_meta)  # [batch_size, hidden_dim]
        seq_meta = seq_meta.unsqueeze(0).repeat(self.n_layers * 2, 1, 1)  # [n_layers * 2, batch_size, hidden_dim]

        _, (hidden, _) = self.rnn(
            seq, (
                seq_meta,
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device)
            )
        )  # [2 * num_layers, batch_size, hidden_dim]

        hidden = hidden.permute(1, 2, 0)  # [batch_size, hidden_dim, 2 * num_layers]
        hidden = self.global_pool(hidden).squeeze(-1)  # [batch_size, hidden_dim]

        output = self.fc1(hidden)  # [batch_size, fc_dim]
        output = self.fc2(output)  # [batch_size, 1]

        return output
