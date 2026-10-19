import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, configs_list, pretrain=False):
        super(Model, self).__init__()

        self.configs_list = configs_list
        self.d_model = args.d_model  # Adding d_model definition

        # Model settings
        self.input_len = args.input_len
        self.num_class = configs_list[0][1]['num_class']
        self.channel_num = configs_list[0][1]['enc_in']
        self.hidden_dim = args.d_model
        self.num_layers = args.e_layers

        # CNN layers
        self.conv1 = nn.Conv1d(self.channel_num, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM layer
        self.lstm = nn.LSTM(128, self.hidden_dim, self.num_layers, batch_first=True)

        # Fully connected layers
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)
        self.rul_predictor = nn.Linear(self.hidden_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, task_id=None, task_name=None, enable_mask=None):
        # x_enc [B, L, D]
        x = x_enc.permute(0, 2, 1)  # Adjusting to match input size [B, D, L]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Preparing for LSTM layer
        x = x.permute(0, 2, 1)  # [B, L, D]

        # LSTM
        lstm_out, (hn, cn) = self.lstm(x)

        # Use the last hidden state as the feature representation
        out = lstm_out[:, -1, :]  # [B, hidden_dim]

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
        self.e_layers = 2  # Number of LSTM layers

args = Args()
configs_list = [(0, {'enc_in': 3, 'num_class': 10})]

model = Model(args, configs_list)
x_enc = torch.randn(10, 2048, 3)
x_mark_enc = torch.randn(10, 1024, 3)
task_name = 'classification'
output = model(x_enc, x_mark_enc, task_name=task_name)
print(output.shape)
