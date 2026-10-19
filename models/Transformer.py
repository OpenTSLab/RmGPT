import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Model(nn.Module):
    def __init__(self, args, configs_list, pretrain=False):
        super(Model, self).__init__()
        
        self.configs_list = configs_list
        self.d_model = args.d_model  # Adding d_model definition

        # Model settings
        self.input_len = args.input_len
        self.num_class = configs_list[0][1]['num_class']
        self.channel_num = configs_list[0][1]['enc_in']
        
        self.embedding = nn.Linear(self.channel_num, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8), num_layers=args.e_layers)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_input_dim = self.d_model  # Since adaptive pool outputs (B, D, 1)

        self.classifier = nn.Linear(self.fc_input_dim, self.num_class)
        self.rul_predictor = nn.Linear(self.fc_input_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, task_id=None, task_name=None, enable_mask=None):
        # x_enc [B, L, D]
        x = self.embedding(x_enc)  # [B, L, D]
        x = self.pos_encoder(x)  # [B, L, D]
        x = x.permute(1, 0, 2)  # [L, B, D]
        out = self.transformer_encoder(x)  # [L, B, D]
        out = out.permute(1, 2, 0)  # [B, D, L]
        out = self.adaptive_pool(out)  # [B, D, 1]
        out = out.view(out.size(0), -1)  # Flatten the output tensor to [B, D]

        if 'classification' in task_name:
            dec_out = self.classifier(out)
            return dec_out  # [B, N]
        if 'RUL' in task_name:
            dec_out = self.rul_predictor(out)
            return dec_out  # [B, 1]
        return None


# Testing the model
class Args:
    def __init__(self):
        self.d_model = 512
        self.stride = 1
        self.patch_len = 1
        self.input_len = 2048
        self.e_layers = 6  # Setting the number of Transformer layers

args = Args()
configs_list = [(0, {'enc_in': 3, 'num_class': 10})]

model = Model(args, configs_list)
x_enc = torch.randn(10, 2048, 3)
x_mark_enc = torch.randn(10, 2048, 3)
task_name = 'classification'
output = model(x_enc, x_mark_enc, task_name=task_name)
print(output.shape)
