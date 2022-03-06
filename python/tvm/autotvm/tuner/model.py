import torch


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_layers=3,
        embed_size=752,
        num_heads=16,
        dropout=0.2,
        hidden_size=256,
        output_size=1,
    ):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout)
        self.encoder = torch.nn.TransformerEncoder(self.layer, num_layers)
        self.decoder = torch.nn.Linear(embed_size, output_size)
        self.src_mask = None

    def forward(self, inputs):
        encode = self.encoder(inputs, self.src_mask)
        output = self.decoder(encode)
        return output.squeeze()


class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(in_dim, hidden_dim, num_layers=3)
        self.decoder = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        output, (h, c) = self.lstm1(output)
        output = self.decoder(output).squeeze()
        return output


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        output = self.linear1(output)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output = self.linear3(output)
        return output.squeeze()
