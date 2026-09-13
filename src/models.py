import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
import numpy as np


def create_output_layer(in_features, task, n_classes):
    if task == "classification":
        out = nn.Linear(in_features, n_classes, bias=False)
    elif task == "regression":
        # n_classes is repurposed here as the output dimension for
        # sequence/vector regression (e.g. predicting a per-timestep
        # presence indicator instead of a single scalar); defaults to a
        # plain scalar output when not provided, matching prior behavior.
        out = nn.Linear(in_features, n_classes if n_classes else 1, bias=False)
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
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(
                n_channels * (n_samples // 2 + 1), task, n_classes
            )
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
            loss = F.cross_entropy(pred, y_batch, reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred, y_batch, reduction="mean")
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
        self.conv1 = nn.Conv1d(
            n_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(
            hidden_dim,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_channels)

        out_samples1 = math.floor(
            (n_samples + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        out_samples2 = math.floor(
            (out_samples1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [n_channels * out_samples2] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
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
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred[:, 0], y_batch, reduction="mean")
        return loss, pred


class TimeSeriesMLP(nn.Module):
    def __init__(
        self,
        task,
        n_channels,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        in_dim = n_channels * n_timepoints

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [in_dim] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(in_dim, task, n_classes)

        self.task = task

    def extract_features(self, x_batch):
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)
        if x_batch.dim() != 3:
            raise ValueError(
                f"Expected input with shape (batch, channels, time) or (batch, time), got {tuple(x_batch.shape)}."
            )
        if x_batch.shape[1] != self.n_channels and x_batch.shape[2] == self.n_channels:
            x_batch = x_batch.transpose(1, 2)

        h = torch.flatten(x_batch, start_dim=1)
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        if len(self.layers):
            h = self.layers[-1](h)
        return h

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        pred = self.out(h)
        if self.task == "classification":
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction="mean")
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            pred = self.out(h)
        return pred


class SpectrogramMLP(nn.Module):
    def __init__(
        self,
        task,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        time_halfbandwidth_product=2,
        baseline_duration=0.5,
        multitaper_method="kramer_mtpr",
        multitaper_n_jobs=1,
        multitaper_output="power",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.bottleneck = bottleneck
        self.freq_cutoff = freq_cutoff
        self.sample_rate = sample_rate
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.time_window_duration = self.window_length / self.sample_rate
        self.time_window_step = self.hop_length / self.sample_rate
        self.baseline_duration = baseline_duration
        self.multitaper_method = multitaper_method
        self.multitaper_n_jobs = multitaper_n_jobs
        self.multitaper_output = multitaper_output
        self.window = torch.hamming_window(self.window_length, periodic=True).float()
        self.baseline_frames = max(1, int(np.ceil(self.baseline_duration * self.sample_rate / self.hop_length)))

        probe = torch.ones((n_channels, n_timepoints), dtype=torch.float32)
        probe_tf = self._spectrogram(probe)
        self.frequencies = np.linspace(0, self.sample_rate / 2, probe_tf.shape[1])

        if self.bottleneck:
            nyquist = self.sample_rate / 2
            assert freq_cutoff <= nyquist
            self.freq_mask = self.frequencies <= freq_cutoff
            if not np.any(self.freq_mask):
                raise ValueError(
                    f"freq_cutoff={freq_cutoff} removes all multitaper frequencies."
                )
            self.n_freq_keep = int(np.sum(self.freq_mask))
        else:
            self.freq_mask = None
            self.n_freq_keep = int(probe_tf.shape[1])

        self.n_time = int(probe_tf.shape[-1])
        probe_feature_map = probe_tf[:, self.freq_mask, :] if self.bottleneck else probe_tf
        in_dim = int(np.prod(probe_feature_map.shape))
        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [in_dim] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(in_dim, task, n_classes)

        self.task = task

    def _spectrogram(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)

        if x_sample.dim() != 2:
            raise ValueError(
                f"Expected a single trial with shape (channels, time), got {tuple(x_sample.shape)}."
            )
        if x_sample.shape[0] != self.n_channels and x_sample.shape[1] == self.n_channels:
            x_sample = x_sample.T

        # ensure signal length is at least n_fft to avoid reflect-padding errors
        signal_len = x_sample.shape[-1]
        if signal_len < self.n_fft:
            pad_amount = self.n_fft - signal_len
            x_sample = F.pad(x_sample, (0, pad_amount), mode="constant", value=0.0)

        window = self.window.to(device=x_sample.device, dtype=x_sample.dtype)
        tf = torch.abs(
            torchaudio.functional.spectrogram(
                x_sample,
                pad=0,
                window=window,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_length,
                power=None,
                normalized=False,
                center=False,
                pad_mode="constant",
                onesided=True,
            )
        )
        baseline = tf[..., : self.baseline_frames].mean(dim=-1, keepdim=True)
        tf = tf / torch.clamp(baseline, min=torch.finfo(tf.dtype).tiny)
        tf = torch.log10(torch.clamp(tf, min=torch.finfo(tf.dtype).tiny))
        return tf

    def _compute_multitaper_feature_map(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        tf = self._spectrogram(x_sample)
        if self.bottleneck:
            tf = tf[:, self.freq_mask, :]
        return tf

    def extract_features(self, x_batch):
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)

        feature_maps = [self._compute_multitaper_feature_map(sample) for sample in x_batch]
        h = torch.stack(feature_maps)
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
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction="mean")
        return loss, pred
    
    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            pred = self.out(h)
        return pred


class BandMLP(nn.Module):
    def __init__(
        self,
        task,
        n_channels,
        n_fft=256,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        band_ranges=None,
        time_halfbandwidth_product=2,
        baseline_duration=0.5,
        multitaper_method="kramer_mtpr",
        multitaper_n_jobs=1,
        multitaper_output="power",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.bottleneck = bottleneck
        self.freq_cutoff = freq_cutoff
        self.sample_rate = sample_rate
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.time_window_duration = self.window_length / self.sample_rate
        self.time_window_step = self.hop_length / self.sample_rate
        self.baseline_duration = baseline_duration
        self.multitaper_method = multitaper_method
        self.multitaper_n_jobs = multitaper_n_jobs
        self.multitaper_output = multitaper_output
        self.window = torch.hamming_window(self.window_length, periodic=True).float()
        self.baseline_frames = max(
            1, int(np.ceil(self.baseline_duration * self.sample_rate / self.hop_length))
        )

        self.band_ranges = (
            band_ranges
            if band_ranges is not None
            else ((0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100))
        )

        probe = torch.ones((n_channels, n_timepoints), dtype=torch.float32)
        probe_tf = self._spectrogram(probe)
        self.frequencies = np.linspace(0, self.sample_rate / 2, probe_tf.shape[1])

        if self.bottleneck:
            nyquist = self.sample_rate / 2
            assert freq_cutoff <= nyquist
            self.freq_mask = self.frequencies <= freq_cutoff
            if not np.any(self.freq_mask):
                raise ValueError(
                    f"freq_cutoff={freq_cutoff} removes all frequencies available to BandMLP."
                )
        else:
            self.freq_mask = None

        probe_band_map = self._band_aggregate(probe_tf)
        in_dim = int(np.prod(probe_band_map.shape))

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [in_dim] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(in_dim, task, n_classes)

        self.task = task

    def _spectrogram(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)

        if x_sample.dim() != 2:
            raise ValueError(
                f"Expected a single trial with shape (channels, time), got {tuple(x_sample.shape)}."
            )
        if x_sample.shape[0] != self.n_channels and x_sample.shape[1] == self.n_channels:
            x_sample = x_sample.T

        # ensure signal length is at least n_fft to avoid reflect-padding errors
        signal_len = x_sample.shape[-1]
        if signal_len < self.n_fft:
            pad_amount = self.n_fft - signal_len
            x_sample = F.pad(x_sample, (0, pad_amount), mode="constant", value=0.0)

        window = self.window.to(device=x_sample.device, dtype=x_sample.dtype)
        tf = torch.abs(
            torchaudio.functional.spectrogram(
                x_sample,
                pad=0,
                window=window,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_length,
                power=None,
                normalized=False,
                center=False,
                pad_mode="constant",
                onesided=True,
            )
        )
        baseline = tf[..., : self.baseline_frames].mean(dim=-1, keepdim=True)
        tf = tf / torch.clamp(baseline, min=torch.finfo(tf.dtype).tiny)
        tf = torch.log10(torch.clamp(tf, min=torch.finfo(tf.dtype).tiny))
        return tf

    def _band_aggregate(self, tf):
        if self.bottleneck:
            tf = tf[:, self.freq_mask, :]
            frequencies = self.frequencies[self.freq_mask]
        else:
            frequencies = self.frequencies

        band_maps = []
        for low, high in self.band_ranges:
            band_mask = (frequencies >= low) & (frequencies < high)
            if not np.any(band_mask):
                raise ValueError(
                    f"Band {low}-{high} Hz contains no frequencies for BandMLP."
                )
            band_mask = torch.as_tensor(band_mask, device=tf.device, dtype=torch.bool)
            band_maps.append(tf[:, band_mask, :].mean(dim=1, keepdim=True))
        return torch.cat(band_maps, dim=1)

    def _compute_band_feature_map(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        tf = self._spectrogram(x_sample)
        return self._band_aggregate(tf)

    def extract_features(self, x_batch):
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)

        feature_maps = [self._compute_band_feature_map(sample) for sample in x_batch]
        h = torch.stack(feature_maps)
        h = torch.flatten(h, start_dim=1)

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
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction="mean")
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            pred = self.out(h)
        return pred


class SpectrogramLogVarMLP(nn.Module):
    """Like SpectrogramMLP, but each per-window spectral feature vector is
    augmented with the log-variance of the raw signal within that same
    window (the feature CSP-LDA classifies on), and is meant to predict one
    output per time bin (e.g. per-timestep signal presence) rather than a
    single per-trial scalar."""

    def __init__(
        self,
        task,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        baseline_duration=0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.bottleneck = bottleneck
        self.freq_cutoff = freq_cutoff
        self.sample_rate = sample_rate
        self.baseline_duration = baseline_duration
        self.window = torch.hamming_window(self.window_length, periodic=True).float()
        self.baseline_frames = max(1, int(np.ceil(self.baseline_duration * self.sample_rate / self.hop_length)))

        probe = torch.ones((n_channels, n_timepoints), dtype=torch.float32)
        probe_tf = self._spectrogram(probe)
        self.frequencies = np.linspace(0, self.sample_rate / 2, probe_tf.shape[1])

        if self.bottleneck:
            nyquist = self.sample_rate / 2
            assert freq_cutoff <= nyquist
            self.freq_mask = self.frequencies <= freq_cutoff
            if not np.any(self.freq_mask):
                raise ValueError(
                    f"freq_cutoff={freq_cutoff} removes all frequencies available to SpectrogramLogVarMLP."
                )
        else:
            self.freq_mask = None

        self.n_time = int(probe_tf.shape[-1])
        probe_feature_map = self._append_log_variance(probe_tf, probe)
        in_dim = int(np.prod(probe_feature_map.shape))

        self.layers = nn.ModuleList()
        if n_layers > 0:
            dims = [in_dim] + [hidden_dim] * n_layers
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.out = create_output_layer(dims[-1], task, n_classes)
        else:
            self.out = create_output_layer(in_dim, task, n_classes)

        self.task = task

    def _normalize_sample(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        if x_sample.dim() != 2:
            raise ValueError(
                f"Expected a single trial with shape (channels, time), got {tuple(x_sample.shape)}."
            )
        if x_sample.shape[0] != self.n_channels and x_sample.shape[1] == self.n_channels:
            x_sample = x_sample.T

        # ensure signal length is at least n_fft to avoid reflect-padding errors
        signal_len = x_sample.shape[-1]
        if signal_len < self.n_fft:
            pad_amount = self.n_fft - signal_len
            x_sample = F.pad(x_sample, (0, pad_amount), mode="constant", value=0.0)
        return x_sample

    def _spectrogram(self, x_sample):
        x_sample = self._normalize_sample(x_sample)
        window = self.window.to(device=x_sample.device, dtype=x_sample.dtype)
        tf = torch.abs(
            torchaudio.functional.spectrogram(
                x_sample,
                pad=0,
                window=window,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_length,
                power=None,
                normalized=False,
                center=False,
                pad_mode="constant",
                onesided=True,
            )
        )
        baseline = tf[..., : self.baseline_frames].mean(dim=-1, keepdim=True)
        tf = tf / torch.clamp(baseline, min=torch.finfo(tf.dtype).tiny)
        tf = torch.log10(torch.clamp(tf, min=torch.finfo(tf.dtype).tiny))
        return tf

    def _windowed_log_variance(self, x_sample):
        # CSP-LDA-style feature: log-variance of the raw (unfiltered by
        # frequency) signal within each of the same windows the spectrogram
        # is computed over, giving one scalar per channel per time bin.
        x_sample = self._normalize_sample(x_sample)
        frames = x_sample.unfold(-1, self.n_fft, self.hop_length)  # (channels, n_time, n_fft)
        var = frames.var(dim=-1, unbiased=False)
        logvar = torch.log(torch.clamp(var, min=torch.finfo(var.dtype).tiny))
        return logvar.unsqueeze(1)  # (channels, 1, n_time)

    def _append_log_variance(self, tf, x_sample):
        if self.bottleneck:
            tf = tf[:, self.freq_mask, :]
        logvar = self._windowed_log_variance(x_sample)
        return torch.cat([tf, logvar], dim=1)

    def _compute_feature_map(self, x_sample):
        tf = self._spectrogram(x_sample)
        return self._append_log_variance(tf, x_sample)

    def extract_features(self, x_batch):
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)

        feature_maps = [self._compute_feature_map(sample) for sample in x_batch]
        h = torch.stack(feature_maps)
        h = torch.flatten(h, start_dim=1)

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
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction="mean")
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            pred = self.out(h)
        return pred


class ClampedLinear(nn.Module):
    """Linear layer (no bias) whose effective weights are hard-clamped to
    [-max_abs_weight, max_abs_weight] on every forward pass, so a single
    noise-correlated input feature can't dominate the output via an
    unbounded coefficient."""

    def __init__(self, in_features, out_features, max_abs_weight=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.max_abs_weight = max_abs_weight

    def forward(self, x):
        w = torch.clamp(self.weight, -self.max_abs_weight, self.max_abs_weight)
        return F.linear(x, w)


class SpectrogramPresenceMLP(SpectrogramMLP):
    """Like SpectrogramMLP (spectral signature only, no log-variance
    feature), but predicts a per-timestep binary presence indicator via
    independent per-bin binary classification (BCE loss) instead of a
    scalar regression target, and constrains the output layer's weights to
    [-max_abs_weight, max_abs_weight] to reduce sensitivity to picking up
    noise via large coefficients."""

    def __init__(
        self,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        max_abs_weight=1.0,
    ):
        super().__init__(
            "regression", n_channels, n_fft=n_fft, window_length=window_length,
            hop_length=hop_length, n_timepoints=n_timepoints, n_classes=n_classes,
            n_layers=n_layers, hidden_dim=hidden_dim, bottleneck=bottleneck,
            freq_cutoff=freq_cutoff, sample_rate=sample_rate,
        )
        in_features = self.out.weight.shape[1]
        out_features = n_classes if n_classes else 1
        self.out = ClampedLinear(in_features, out_features, max_abs_weight=max_abs_weight)
        self.task = "presence_classification"

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        logits = self.out(h)
        loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="mean")
        pred = torch.sigmoid(logits)
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            logits = self.out(h)
            pred = torch.sigmoid(logits)
        return pred


class SpectrogramSharedBinPresenceMLP(SpectrogramMLP):
    """Like SpectrogramPresenceMLP, but instead of flattening the whole
    time-frequency map and predicting every output time bin from a weight
    row that can draw on the entire spectrogram, applies ONE shared
    linear-classification weight vector (over frequency only) to each time
    bin's own frequency column independently -- the same weights reused at
    every time bin, so a bin's presence prediction can only depend on that
    bin's own spectral content, not the rest of the trial. Weights
    hard-clamped to [-max_abs_weight, max_abs_weight].

    Uses a multitaper (DPSS) power spectral estimate per window instead of
    a single Hamming-windowed periodogram: K orthogonal tapers are applied
    to each window and their periodograms averaged, reducing the variance
    of each bin's power estimate (a single-window periodogram is noisy
    even when the true underlying oscillation is clearly visible in the
    raw signal). K = 2*time_halfbandwidth_product - 1 tapers are used,
    chosen to keep the mainlobe width close to what a single Hamming
    window already achieves at window_length=n_fft, so this trades
    variance down without materially widening frequency resolution."""

    def __init__(
        self,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        max_abs_weight=1.0,
        time_halfbandwidth_product=2.0,
        l1_weight=0.0,
        pos_weight=None,
        log_power=True,
        zscore_freq=False,
    ):
        # Must be set before super().__init__(), which probes self._spectrogram()
        # during its own __init__ to determine n_freq/n_time; plain (non-Tensor,
        # non-Module) attributes are safe to set pre-init, same reasoning as
        # _get_tapers' use of self.__dict__ directly below.
        self.log_power = log_power
        # Per-trial, per-frequency-bin z-score (mean/std computed across
        # that trial's own time-bin axis), applied after the log_power
        # transform (or directly to linear power if log_power=False).
        # Needed when log_power=False: raw multitaper power spans several
        # orders of magnitude across frequency bins and trials, which blows
        # up the classifier's logits (and, via the lam-scaled supervised
        # gradient, the unmixing matrix update) regardless of lr_model.
        # z-scoring brings every bin/trial onto a comparable O(1) scale
        # without reintroducing the log compression.
        self.zscore_freq = zscore_freq
        super().__init__(
            "regression", n_channels, n_fft=n_fft, window_length=window_length,
            hop_length=hop_length, n_timepoints=n_timepoints, n_classes=1,
            n_layers=0, bottleneck=bottleneck, freq_cutoff=freq_cutoff,
            sample_rate=sample_rate, time_halfbandwidth_product=time_halfbandwidth_product,
        )
        n_freq = self.n_freq_keep
        self.out = ClampedLinear(n_freq, 1, max_abs_weight=max_abs_weight)
        self.task = "presence_classification_shared"
        # L1 penalty on the shared per-bin logistic-regression weight
        # vector, on top of AdamW's existing L2 weight_decay -- encourages
        # the classifier to concentrate weight on the informative frequency
        # bins and drive marginal/incidental bins to exactly zero, rather
        # than just shrinking every bin's weight proportionally.
        self.l1_weight = l1_weight
        # Positive-class weight for the BCE loss, to counteract label
        # imbalance (e.g. a short presence window inside a long trial):
        # None (default) leaves the loss unweighted, a float applies a
        # fixed weight, and "auto" recomputes neg/pos from each batch's own
        # y_batch, so a missed positive costs as much as its true rarity in
        # that batch warrants rather than being swamped by the majority
        # negative class.
        self.pos_weight = pos_weight

    def _get_tapers(self, device, dtype):
        # Lazily computed and cached via __dict__ directly (not a
        # registered buffer) so this works regardless of nn.Module
        # init-order, since it's first needed during the parent class's
        # own __init__-time probe call.
        cache = self.__dict__.setdefault("_tapers_cache", {})
        key = (str(device), str(dtype))
        if key not in cache:
            from scipy.signal.windows import dpss
            n_tapers = max(1, int(2 * self.time_halfbandwidth_product) - 1)
            tapers_np = dpss(self.window_length, self.time_halfbandwidth_product, Kmax=n_tapers)
            cache[key] = torch.as_tensor(tapers_np.copy(), dtype=dtype, device=device)
        return cache[key]

    def _spectrogram(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        if x_sample.dim() != 2:
            raise ValueError(
                f"Expected a single trial with shape (channels, time), got {tuple(x_sample.shape)}."
            )
        if x_sample.shape[0] != self.n_channels and x_sample.shape[1] == self.n_channels:
            x_sample = x_sample.T

        signal_len = x_sample.shape[-1]
        if signal_len < self.n_fft:
            x_sample = F.pad(x_sample, (0, self.n_fft - signal_len), mode="constant", value=0.0)

        frames = x_sample.unfold(-1, self.n_fft, self.hop_length)  # (channels, n_time, n_fft)
        windowed = frames[..., : self.window_length]  # (channels, n_time, window_length)

        tapers = self._get_tapers(device=x_sample.device, dtype=x_sample.dtype)  # (K, window_length)
        tapered = windowed.unsqueeze(-2) * tapers  # (channels, n_time, K, window_length)
        if self.n_fft > self.window_length:
            tapered = F.pad(tapered, (0, self.n_fft - self.window_length))
        spec = torch.fft.rfft(tapered, n=self.n_fft, dim=-1)  # (channels, n_time, K, n_freq)
        mt_power = (spec.real ** 2 + spec.imag ** 2).mean(dim=-2)  # average over tapers
        tf = mt_power.transpose(-1, -2)  # (channels, n_freq, n_time), matching existing convention

        # log10(power) by default (self.log_power=True), no baseline-ratio
        # division. The original baseline-ratio+log feature discarded
        # absolute magnitude (a tiny bin doubling from baseline looked
        # identical to a huge bin doubling from baseline); plain raw
        # (linear) power preserved magnitude but put every bin on a very
        # wide, unbounded numeric scale. log-compressing the raw power
        # (without normalizing it away against a baseline first) keeps the
        # magnitude ordering between bins intact while still giving a more
        # tractable dynamic range for the classifier's clamped weights to
        # operate over. Clamping to a machine-tiny floor avoids log(0) and
        # is harmless when log_power=False since real EEG power is always
        # far above that floor.
        tf = torch.clamp(tf, min=torch.finfo(tf.dtype).tiny)
        if self.log_power:
            tf = torch.log10(tf)
        if self.zscore_freq:
            # Per-(channel, freq)-bin z-score across this trial's own time
            # axis (dim=-1): no persistent/running statistics, so identical
            # at train and eval time and independent of batch composition.
            mean = tf.mean(dim=-1, keepdim=True)
            std = tf.std(dim=-1, keepdim=True)
            tf = (tf - mean) / (std + 1e-8)
        return tf

    def extract_features(self, x_batch):
        # Raw (channel, freq, time) feature map, NOT flattened, so the
        # shared per-bin classifier can be applied along the time axis.
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)
        feature_maps = [self._compute_multitaper_feature_map(sample) for sample in x_batch]
        h = torch.stack(feature_maps)  # (batch, n_channels, n_freq, n_time)
        return h

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        h = h.squeeze(1).transpose(1, 2)  # (batch, n_time, n_freq)
        logits = self.out(h).squeeze(-1)  # (batch, n_time) — shared weights across time
        if self.pos_weight == "auto":
            n_pos = y_batch.sum()
            n_neg = y_batch.numel() - n_pos
            pw = (n_neg / n_pos.clamp(min=1)).detach()
        elif self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        else:
            pw = None
        loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="mean", pos_weight=pw)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.out.weight.abs().sum()
        pred = torch.sigmoid(logits)
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            h = h.squeeze(1).transpose(1, 2)
            logits = self.out(h).squeeze(-1)
            pred = torch.sigmoid(logits)
        return pred


class MultiChannelSharedBinPresenceMLP(SpectrogramSharedBinPresenceMLP):
    """Like SpectrogramSharedBinPresenceMLP, but the shared per-time-bin
    classifier reads ALL channels at once: at each time bin, every
    channel's spectral column (n_freq_keep values) is concatenated into a
    single (n_channels * n_freq_keep) input vector, and one shared weight
    vector over that concatenated space is applied at every time bin (same
    weights reused across time, exactly like the parent class -- only the
    per-bin input width changes, from one channel's frequencies to every
    channel's). Meant for training directly on raw multi-channel data
    (e.g. all-channel LFP/EEG), with no upstream unmixing matrix -- the
    parent class hardcodes n_channels=1 in forward()/predict() (a bare
    `.squeeze(1)`), which silently breaks for n_channels>1; this class
    fixes that by reshaping instead of squeezing."""

    def __init__(self, n_channels, *args, max_abs_weight=1.0, **kwargs):
        super().__init__(n_channels, *args, max_abs_weight=max_abs_weight, **kwargs)
        # Parent __init__ already built self.out sized for ONE channel's
        # frequencies (n_freq_keep); replace it with the correct
        # (n_channels * n_freq_keep)-input version.
        self.out = ClampedLinear(n_channels * self.n_freq_keep, 1, max_abs_weight=max_abs_weight)

    def _reshape_features(self, x_batch):
        h = self.extract_features(x_batch)  # (batch, n_channels, n_freq, n_time)
        batch, n_channels, n_freq, n_time = h.shape
        return h.permute(0, 3, 1, 2).reshape(batch, n_time, n_channels * n_freq)  # (batch, n_time, n_channels*n_freq)

    def forward(self, x_batch, y_batch):
        h = self._reshape_features(x_batch)
        logits = self.out(h).squeeze(-1)  # (batch, n_time) — shared weights across time
        if self.pos_weight == "auto":
            n_pos = y_batch.sum()
            n_neg = y_batch.numel() - n_pos
            pw = (n_neg / n_pos.clamp(min=1)).detach()
        elif self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        else:
            pw = None
        loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="mean", pos_weight=pw)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.out.weight.abs().sum()
        pred = torch.sigmoid(logits)
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self._reshape_features(x_batch)
            logits = self.out(h).squeeze(-1)
            pred = torch.sigmoid(logits)
        return pred


class SpectrogramSharedBinClassifierMLP(SpectrogramSharedBinPresenceMLP):
    """Like SpectrogramSharedBinPresenceMLP, but with a genuine two-class
    ("no presence" / "presence") softmax classification readout per time
    bin, trained with cross-entropy loss instead of a single-logit sigmoid +
    BCE. predict() returns the hard argmax 0/1 class label directly, rather
    than a continuous probability that a caller has to threshold themselves.
    Reuses the parent class's multitaper spectrogram/feature extraction
    unchanged; only the output layer, loss, and prediction are different."""

    def __init__(
        self,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        max_abs_weight=1.0,
        time_halfbandwidth_product=2.0,
        l1_weight=0.0,
        class_weight=None,
        log_power=True,
        zscore_freq=False,
    ):
        super().__init__(
            n_channels, n_fft=n_fft, window_length=window_length, hop_length=hop_length,
            n_timepoints=n_timepoints, bottleneck=bottleneck, freq_cutoff=freq_cutoff,
            sample_rate=sample_rate, max_abs_weight=max_abs_weight,
            time_halfbandwidth_product=time_halfbandwidth_product, l1_weight=l1_weight,
            log_power=log_power, zscore_freq=zscore_freq,
        )
        n_freq = self.n_freq_keep
        self.out = ClampedLinear(n_freq, 2, max_abs_weight=max_abs_weight)
        self.task = "presence_classification_shared_ce"
        # Per-class weight for cross-entropy, analogous to pos_weight for
        # BCE: None (default) leaves the loss unweighted, a 2-element
        # sequence applies fixed [weight_no_presence, weight_presence], and
        # "auto" recomputes both from each batch's own y_batch so the rarer
        # class isn't swamped by the majority class during training.
        self.class_weight = class_weight

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        h = h.squeeze(1).transpose(1, 2)  # (batch, n_time, n_freq)
        logits = self.out(h)  # (batch, n_time, 2) — shared weights across time
        target = y_batch.long()
        if self.class_weight == "auto":
            n_pos = target.sum()
            n_neg = target.numel() - n_pos
            weight = torch.stack([
                target.numel() / (2.0 * n_neg.clamp(min=1)),
                target.numel() / (2.0 * n_pos.clamp(min=1)),
            ]).to(logits.dtype).detach()
        elif self.class_weight is not None:
            weight = torch.as_tensor(self.class_weight, dtype=logits.dtype, device=logits.device)
        else:
            weight = None
        # F.cross_entropy expects the class dimension at index 1.
        loss = F.cross_entropy(logits.transpose(1, 2), target, weight=weight, reduction="mean")
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.out.weight.abs().sum()
        pred = torch.argmax(logits, dim=-1).float()  # (batch, n_time): hard 0/1 label
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            h = self.extract_features(x_batch)
            h = h.squeeze(1).transpose(1, 2)
            logits = self.out(h)
            pred = torch.argmax(logits, dim=-1).float()
        return pred


class SpectrogramMLPL1(SpectrogramMLP):
    def __init__(
        self,
        task,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        n_classes=None,
        n_layers=0,
        hidden_dim=64,
        bottleneck=False,
        freq_cutoff=250,
        sample_rate=1000,
        l1_lambda=1e-4,
        l1_on_all_parameters=False,
    ):
        super().__init__(
            task,
            n_channels,
            n_fft=n_fft,
            window_length=window_length,
            hop_length=hop_length,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            bottleneck=bottleneck,
            freq_cutoff=freq_cutoff,
            sample_rate=sample_rate,
        )
        self.l1_lambda = l1_lambda
        self.l1_on_all_parameters = l1_on_all_parameters

    def l1_penalty(self):
        if self.l1_on_all_parameters:
            parameters = [p for p in self.parameters() if p.requires_grad]
        else:
            parameters = [self.out.weight]
            parameters.extend(layer.weight for layer in self.layers)
        return sum(parameter.abs().sum() for parameter in parameters)

    def forward(self, x_batch, y_batch):
        h = self.extract_features(x_batch)
        pred = self.out(h)
        if self.task == "classification":
            loss = F.cross_entropy(pred, y_batch.long(), reduction="mean")
        elif self.task == "regression":
            loss = F.mse_loss(pred.squeeze(1), y_batch, reduction="mean")
        loss = loss + self.l1_lambda * self.l1_penalty()
        return loss, pred


class MeanNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_batch, y_batch):
        y_pred = x_batch.mean(dim=-1)
        loss = F.mse_loss(y_pred, y_batch, reduction="mean")
        return loss, y_pred


class LogVarianceSharedBinClassifierMLP(nn.Module):
    """Differentiable LDA-equivalent: a single log-variance feature per time
    bin (same window_length/hop_length/n_fft sliding-window framing as the
    Spectrogram* models, so n_time bins line up with existing presence
    labels), fed through one shared 2-class linear head (identical weights
    reused at every time bin) trained with cross-entropy via ordinary
    backprop.

    This exists because a real closed-form sklearn LDA (fit via class
    means/pooled covariance) has no gradient w.r.t. its input, so it cannot
    supply the supervised gradient sisr's W update needs -- it can
    only be applied post-hoc to an already-fixed W (see
    neural_moab_baseline_presence_lda_comparison.py). This class trades the
    closed-form guarantee for differentiability, letting an LDA-style
    (log-variance + linear discriminant) objective actually drive W during
    training, at the cost of no longer being literally LDA (gradient
    descent on a linear classifier is logistic regression, not LDA, though
    the two coincide asymptotically under Gaussian equal-covariance
    class-conditionals).

    Unlike the multitaper spectrogram classifiers, there is no frequency
    axis: CSP's per-component log-variance feature is reduced to a single
    scalar per bin, since here "the component" is just W's row, already
    produced by the unmixing matrix."""

    def __init__(
        self,
        n_channels,
        n_fft=50,
        window_length=50,
        hop_length=25,
        n_timepoints=2000,
        sample_rate=1000,
        class_weight=None,
        eps=1e-10,
        max_abs_weight=1.0,
    ):
        super().__init__()
        assert n_channels == 1, "LogVarianceSharedBinClassifierMLP expects a single (already-unmixed) source channel"
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.class_weight = class_weight
        self.eps = eps
        self.task = "log_variance_classification_shared"
        # Weight (not bias) hard-clamped to [-max_abs_weight, max_abs_weight],
        # matching ClampedLinear's role for the other Spectrogram* classifiers:
        # without it, the weight can grow unboundedly during training and,
        # via the lam-scaled supervised gradient, blow up the unmixing matrix
        # update (empirically diverged to NaN around ~1000 iterations
        # without this). Bias is left unclamped since it can't cause the
        # same multiplicative blowup and LDA's own decision boundary is
        # affine (needs an intercept term).
        self.max_abs_weight = max_abs_weight

        probe = torch.ones((n_channels, n_timepoints), dtype=torch.float32)
        probe_feat = self._log_variance(probe)
        self.n_time = int(probe_feat.shape[-1])

        self.out = nn.Linear(1, 2, bias=True)  # shared across time bins

    def _log_variance(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        signal_len = x_sample.shape[-1]
        if signal_len < self.n_fft:
            x_sample = F.pad(x_sample, (0, self.n_fft - signal_len), mode="constant", value=0.0)
        frames = x_sample.unfold(-1, self.n_fft, self.hop_length)  # (channels, n_time, n_fft)
        windowed = frames[..., : self.window_length]  # (channels, n_time, window_length)
        # unbiased=False (population variance, ddof=0) to match np.var's
        # default used in neural_moab_baseline_presence_lda_comparison.py.
        var = windowed.var(dim=-1, unbiased=False)  # (channels, n_time)
        log_var = torch.log(var + self.eps)
        # Per-trial z-score across this trial's own time axis. Weight
        # clamping alone (see max_abs_weight) was not sufficient to prevent
        # NaN divergence around ~1000-2000 iterations: unmixing-matrix scale
        # drift during training can still push raw log-variance far enough
        # from its typical range to blow up the lam-scaled supervised
        # gradient. This mirrors the fix used for
        # SpectrogramSharedBinPresenceMLP's analogous linear-power
        # divergence (zscore_freq=True).
        mean = log_var.mean(dim=-1, keepdim=True)
        std = log_var.std(dim=-1, keepdim=True)
        return (log_var - mean) / (std + 1e-8)

    def extract_features(self, x_batch):
        # sisr's training loop calls model(W[c] @ x_batch, ...) with a
        # 2-D (batch, T) tensor (no explicit channel dim), while evaluation
        # code (compute_loss) passes 3-D (batch, 1, T); normalize both to
        # 3-D, matching SpectrogramSharedBinPresenceMLP's convention.
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)
        feats = torch.stack([self._log_variance(sample) for sample in x_batch])  # (batch, n_channels, n_time)
        return feats

    def _clamped_out(self, feats):
        w = torch.clamp(self.out.weight, -self.max_abs_weight, self.max_abs_weight)
        return F.linear(feats, w, self.out.bias)

    def forward(self, x_batch, y_batch):
        feats = self.extract_features(x_batch).squeeze(1).unsqueeze(-1)  # (batch, n_time, 1)
        logits = self._clamped_out(feats)  # (batch, n_time, 2) -- shared weights across time
        target = y_batch.long()
        if self.class_weight == "auto":
            n_pos = target.sum()
            n_neg = target.numel() - n_pos
            weight = torch.stack([
                target.numel() / (2.0 * n_neg.clamp(min=1)),
                target.numel() / (2.0 * n_pos.clamp(min=1)),
            ]).to(logits.dtype).detach()
        elif self.class_weight is not None:
            weight = torch.as_tensor(self.class_weight, dtype=logits.dtype, device=logits.device)
        else:
            weight = None
        loss = F.cross_entropy(logits.transpose(1, 2), target, weight=weight, reduction="mean")
        pred = torch.argmax(logits, dim=-1).float()  # (batch, n_time): hard 0/1 label
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            feats = self.extract_features(x_batch).squeeze(1).unsqueeze(-1)
            logits = self._clamped_out(feats)
            pred = torch.argmax(logits, dim=-1).float()
        return pred


class LogVarianceTrialClassifierMLP(nn.Module):
    """Trial-level (not per-time-bin) LDA-equivalent: a single log-variance
    feature computed over one fixed analysis window per trial (by default,
    the motor-imagery portion after skip_samples leading samples are
    dropped, matching CSP+LDA's own MI-only analysis window), fed through a
    2-class linear head trained with cross-entropy -- one prediction per
    trial. This mirrors how CSP+LDA actually predicts (a single decision
    per trial from a single log-variance feature per component), unlike
    LogVarianceSharedBinClassifierMLP's per-time-bin presence-over-time
    signal. Labels are accordingly a plain per-trial scalar (1 if this
    trial's class matches this source, else 0), not a presence vector.

    Weight (not bias) is hard-clamped to [-max_abs_weight, max_abs_weight]
    and the feature is z-scored across the batch of trials, for the same
    numerical-stability reasons as LogVarianceSharedBinClassifierMLP."""

    def __init__(
        self,
        n_channels,
        skip_samples=0,
        sample_rate=1000,
        class_weight=None,
        eps=1e-10,
        max_abs_weight=1.0,
    ):
        super().__init__()
        assert n_channels == 1, "LogVarianceTrialClassifierMLP expects a single (already-unmixed) source channel"
        self.n_channels = n_channels
        self.skip_samples = skip_samples
        self.sample_rate = sample_rate
        self.class_weight = class_weight
        self.eps = eps
        self.max_abs_weight = max_abs_weight
        self.task = "log_variance_trial_classification"
        self.n_time = 1  # one prediction per trial, not per bin

        self.out = nn.Linear(1, 2, bias=True)

    def _log_variance(self, x_sample):
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        seg = x_sample[:, self.skip_samples:]
        var = seg.var(dim=-1, unbiased=False)  # (channels,)
        return torch.log(var + self.eps)

    def extract_features(self, x_batch):
        # Normalize 2-D (batch, T) [sisr's training-loop call
        # convention] vs. 3-D (batch, 1, T) [evaluation convention] the same
        # way LogVarianceSharedBinClassifierMLP does.
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)
        feats = torch.stack([self._log_variance(sample) for sample in x_batch])  # (batch, n_channels)
        return feats

    def _clamped_out(self, feats):
        w = torch.clamp(self.out.weight, -self.max_abs_weight, self.max_abs_weight)
        return F.linear(feats, w, self.out.bias)

    def _zscored(self, feats):
        # z-score across the trial (batch) axis -- the natural axis of
        # variation here, since there is only one feature value per trial
        # (unlike the shared-bin model, which z-scores across time within
        # a single trial).
        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True)
        return (feats - mean) / (std + 1e-8)

    def forward(self, x_batch, y_batch):
        feats = self._zscored(self.extract_features(x_batch))  # (batch, 1)
        logits = self._clamped_out(feats)  # (batch, 2)
        target = y_batch.long().reshape(-1)
        if self.class_weight == "auto":
            n_pos = target.sum()
            n_neg = target.numel() - n_pos
            weight = torch.stack([
                target.numel() / (2.0 * n_neg.clamp(min=1)),
                target.numel() / (2.0 * n_pos.clamp(min=1)),
            ]).to(logits.dtype).detach()
        elif self.class_weight is not None:
            weight = torch.as_tensor(self.class_weight, dtype=logits.dtype, device=logits.device)
        else:
            weight = None
        loss = F.cross_entropy(logits, target, weight=weight, reduction="mean")
        pred = torch.argmax(logits, dim=-1).float()  # (batch,): hard 0/1 label, one per trial
        return loss, pred

    def predict(self, x_batch):
        with torch.no_grad():
            feats = self._zscored(self.extract_features(x_batch))
            logits = self._clamped_out(feats)
            pred = torch.argmax(logits, dim=-1).float()
        return pred
