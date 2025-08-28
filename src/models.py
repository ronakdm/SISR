import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
import scipy

def create_output_layer(in_features, task, n_classes):
    if task == "classification":
        out = nn.Linear(in_features, n_classes, bias=False)
    elif task == "regression":
        out = nn.Linear(in_features, 1, bias=False)
    else:
        raise NotImplementedError
    return out
    
class FourierMLP(nn.Module):
    def __init__(self, task, n_channels, n_samples, n_classes=None, hidden_dims=[]):
        super().__init__()
        assert n_samples % 2 == 0
        self.layers = nn.ModuleList()
        if len(hidden_dims) > 0:
            dims = [n_channels * (n_samples // 2 + 1)] + hidden_dims
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out =  create_output_layer(dims[-1], task, n_classes)
        else:
            self.out =  create_output_layer(n_channels * (n_samples // 2 + 1), task, n_classes)
        self.bn = nn.BatchNorm1d(n_channels * (n_samples // 2 + 1), affine=False)
        self.task = task

    def extract_features(self, x_batch):
        psd = torch.square(torch.fft.rfft(x_batch, norm="ortho")).mean(axis=1)
        h = torch.log(psd).reshape(len(psd), -1)
        h = self.bn(h)
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        if len(self.layers):
            h = self.layers[-1](h)
        return h

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        pred = self.out(h)
        if self.task == "classification":
            loss = F.cross_entropy(pred, y_batch, reduction='mean')
        elif self.task == "regression":
            loss = F.mse_loss(pred, y_batch, reduction='mean')
        return loss, pred
    
class CNN(nn.Module):
    def __init__(
            self, 
            task,
            n_channels,
            n_samples,
            n_classes=None,
            hidden_dim=32, 
            n_layers=0,
            kernel_size=5, 
            dilation=1, 
            stride=1,
        ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(n_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(n_channels)

        out_samples1 = math.floor((n_samples + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        out_samples2 = math.floor((out_samples1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [n_channels * out_samples2] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out =  create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(n_channels * out_samples2, task, n_classes)
        self.task = task
    
    def extract_features(self, x_batch):
        # convolution
        h = F.relu(self.bn1(self.conv1(x_batch)))
        h = F.relu(self.bn2(self.conv2(h)) + x_batch)
        h = torch.flatten(h, start_dim=1)

        # fully connected
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        if len(self.layers):
            h = self.layers[-1](h)
        return h
    
    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        pred = self.out(h)
        if self.task == "classification":
            loss = F.cross_entropy(pred, y_batch.long(), reduction='mean')
        elif self.task == "regression":
            loss = F.mse_loss(pred[:, 0], y_batch, reduction='mean')
        return loss, pred
    
class SpectrogramMLP(nn.Module):
    def __init__(
            self, 
            task,
            n_channels,
            n_freq,
            n_windows,
            n_classes=None,
            win_len=50,
            n_layers=0,
            hidden_dim=64,
        ):
        super().__init__()
        self.out = nn.Linear(n_channels * n_windows * n_freq, 1)
        self.window = torch.from_numpy(scipy.signal.windows.hamming(win_len)).float()
        self.win_len = 50

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [n_channels * n_windows * n_freq] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(n_channels * n_windows * n_freq, task, n_classes)
        self.task = task
    
    def extract_features(self, x_batch):
        # spectrogram
        h = torch.abs(torchaudio.functional.spectrogram(
            x_batch, 
            0, 
            self.window, 
            50, 
            self.win_len // 2, 
            self.win_len, 
            None, 
            True
        ))
        h = torch.flatten(h, start_dim=1)

        # fully connected
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        if len(self.layers) == 1:
            h = F.relu(self.layers[-1](h))
        elif len(self.layers) > 1:
            h = self.layers[-1](h)
        return h
    
    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        pred = self.out(h)
        if self.task == "classification":
            loss = F.cross_entropy(pred, y_batch.long(), reduction='mean')
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction='mean')
        return loss, pred

class MeanNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_batch, y_batch):
        y_pred = x_batch.mean(dim=-1)
        loss = F.mse_loss(y_pred, y_batch, reduction='mean')
        return loss, y_pred
