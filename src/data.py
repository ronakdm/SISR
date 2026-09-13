import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import scipy
from scipy.linalg import hilbert
from scipy.stats import boxcox
from scipy.signal.windows import tukey
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

####################################
# Auxiliary functions
####################################


def create_hilbert_matrix(n, alpha):

    # Get eigenvectors of Hilbert matrix
    H = hilbert(n)
    _, Q = np.linalg.eigh(H)  # Q is orthonormal

    # Create exponentially spaced eigenvalues: lambda_i = exp(alpha * (i - 1)/(n - 1))
    lambdas = np.exp(alpha * np.linspace(0, 1, n))
    Lambda = np.diag(lambdas)

    # Construct matrix with controlled condition number ~ exp(alpha)
    A_alpha = Q @ Lambda @ Q.T
    cond_number = np.linalg.cond(A_alpha)

    return A_alpha, cond_number


def extract_spectrogram_features(x_batch, n_fft=50, win_len=50):
    window = torch.from_numpy(scipy.signal.windows.hamming(win_len)).float()
    abs_spectrogram = torch.abs(
        torchaudio.functional.spectrogram(
            x_batch, 0, window, n_fft, win_len // 2, win_len, None, True
        )
    )
    return abs_spectrogram


####################################
# Simulated data
####################################


def generate_laplace(
    dataset_name="mini", train_size=0.8, noise_scale=0.0, generator=None
):
    rng = np.random.default_rng(generator)
    if dataset_name == "mini":
        N, C, T = 100, 10, 10**3
    elif dataset_name == "wide":
        N, C, T = 100, 30, 10**5
    elif dataset_name == "long":
        N, C, T = 100, 10, 10**6
    else:
        raise ValueError

    s = rng.laplace(size=(N, C, T))
    mixing_mat = rng.standard_normal(size=(C, C))
    x = np.einsum(
        "ij,njt->nit", mixing_mat + noise_scale * rng.standard_normal(size=(C, C)), s
    )
    x = x - np.mean(x, axis=-1, keepdims=True)

    x_train, x_test = train_test_split(x, train_size=train_size, random_state=generator)

    return x_train, x_test, mixing_mat


def generate_multioutput_simulated_1(
    dataset_name="mini",
    M=5,
    noise_scale=0.0,
    cond_number=5,
    train_size=0.8,
    generator=None,
):
    rng = np.random.default_rng(generator)
    if dataset_name == "mini":
        N, C, T = 1000, 10, 2 * 10**3
    elif dataset_name == "tall":
        N, C, T = 10**4, 10, 2 * 10**3
    else:
        raise ValueError

    assert M <= C
    sampling_rate = 10**3

    mixing_mat, _ = create_hilbert_matrix(C, cond_number)
    task_start_index = int(0.25 * T)
    task_completion_limit = int(0.9 * T)

    # Initialize sources array
    sources = np.zeros((N, C, T)).astype(np.float32)
    labels = np.zeros((N, M)).astype(np.float32)

    # Generate sources for each sample
    signals = []

    frequencies = np.linspace(20, 100, M, dtype=np.int32)
    for i in range(N):
        # Unsupervised sources
        sources[i, : C - M] = rng.laplace(size=(C - M, T))

        # Supervised sources
        for j in range(M):
            task_completed_index = rng.integers(task_start_index, task_completion_limit)
            labels[i, j] = frequencies[j] * (task_completed_index - task_start_index) / sampling_rate
            ratio_to_complete = (task_completed_index - task_start_index) / T
            signal_duration = np.arange(
                int(ratio_to_complete * task_start_index), task_start_index
            )
            
            sources[i, C - M + j, signal_duration] = np.sin(
                2 * np.pi * frequencies[j] * signal_duration / sampling_rate
            )

        signals.append(
            (mixing_mat + noise_scale * rng.standard_normal(size=(C, C))) @ sources[i]
        )

    x = np.stack(signals)

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, random_state=generator
    )

    # To get the original sources again, just use the same seeds as used in the experiment setting.
    metadata = {"mixing_mat": mixing_mat, "labels": labels}

    return x_train, y_train, x_test, y_test, metadata

def normalize_energy(x: np.ndarray) -> np.ndarray:
    return x / (np.sqrt(np.mean(x**2)))


def generate_multioutput_simulated_2(
    dataset_name="mini",
    M=5,
    noise_scale=0.0,
    snr=1,
    cond_number=5,
    train_size=0.8,
    freq=10,
    generator=None,
):
    rng = np.random.default_rng(generator)
    if dataset_name == "mini":
        N, C, T = 1000, 10, 2 * 10**3
    elif dataset_name == "tall":
        N, C, T = 10**4, 10, 2 * 10**3
    else:
        raise ValueError

    assert M <= C
    sampling_rate = 10**3

    mixing_mat, _ = create_hilbert_matrix(C, cond_number)
    task_start_index = int(0.25 * T)
    task_completion_limit = int(0.9 * T)

    # Signal-to-noise ratio
    scale = np.sqrt(snr)

    # Initialize sources array
    sources = np.zeros((N, C, T)).astype(np.float32)
    labels = np.zeros((N, M)).astype(np.int32)

    # Generate sources for each sample
    signals = []

    for i in range(N):
        # Unsupervised sources
        background_signal = rng.laplace(size=(C - M, T))
        for c in range(C-M):
            sources[i, c] = normalize_energy(background_signal[c])

        # Supervised sources
        labels[i, :] = rng.choice([0, 1], size=M)
        for j in range(M): 
            if labels[i, j] == 1:
                signal_duration = np.arange(
                    task_start_index, task_completion_limit 
                )
                
                supervised_signal = np.sin(
                    2 * np.pi * (j+1) * freq * signal_duration / sampling_rate
                )
                sources[i, C - M + j, signal_duration] = scale * normalize_energy(supervised_signal)
            else:
                background_signal = rng.laplace(size=(1, T))
                sources[i, C - M + j, :] = normalize_energy(background_signal)

        signals.append(
            (mixing_mat + noise_scale * rng.standard_normal(size=(C, C))) @ sources[i]
        )

    x = np.stack(signals)
    labels = labels.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, random_state=generator
    )

    # To get the original sources again, just use the same seeds as used in the experiment setting.
    metadata = {"mixing_mat": mixing_mat, "labels": labels}

    return x_train, y_train, x_test, y_test, metadata


def generate_multioutput_simulated_3(
    dataset_name="mini",
    M=2,
    noise_scale=0.0,
    signal_scale = 10,
    snr=1,
    cond_number=5,
    train_size=0.8,
    generator=None,
):
    """Like simulated_2 but the supervised step pulse and gamma wave live on
    different latent sources before mixing, alongside 6 additional
    unlabeled oscillatory "distractor" sources (each at its own hardcoded
    frequency) and 2 pure-noise (Laplace) background sources -- 10 total
    sources mixed into the observed data via a square, invertible C=10
    mixing matrix, of which only pulse and gamma are behaviorally labeled
    (M=2, hardcoded). All 8 oscillatory sources (labeled and unlabeled
    alike) have an independently-sampled random duration AND random start
    time each trial -- including pulse, which previously always started at
    a fixed 500ms -- and are Tukey-tapered (alpha=0.5) rather than gated
    on/off abruptly, to avoid the spectral leakage a hard rectangular
    envelope would introduce.
    """
    TAPER_ALPHA = 0.5  # Tukey window: fraction of each burst spent ramping up/down
    rng = np.random.default_rng(generator)
    if dataset_name == "mini":
        N, C, T = 1000, 10, 2 * 10**3
    elif dataset_name == "tall":
        N, C, T = 10**4, 10, 2 * 10**3
    else:
        raise ValueError

    assert M == 2, "this generator hardcodes exactly 2 labeled sources (pulse, gamma)"
    sampling_rate = 10**3

    mixing_mat = rng.standard_normal(size=(C, C))

    scale = signal_scale * np.sqrt(snr)
    min_duration, max_duration = 100, 500

    sources = np.zeros((N, C, T), dtype=np.float32)
    labels = np.zeros((N, M), dtype=np.float32)
    pulse_lengths = np.zeros((N, M), dtype=np.int32)
    gamma_lengths = np.zeros((N, M), dtype=np.int32)
    pulse_starts = np.zeros(N, dtype=np.int32)
    gamma_starts = np.zeros(N, dtype=np.int32)

    step_freq = 10.0
    gamma_freq = 90.0
    # 6 unlabeled oscillatory "distractor" sources, each at its own
    # hardcoded frequency (distinct from step_freq/gamma_freq) -- present
    # to make source separation harder/more realistic than plain Laplace
    # noise, but carry no behavioral label.
    distractor_freqs = [7.0, 16.0, 24.0, 45.0, 75.0, 120.0]
    assert len(distractor_freqs) == C - 4  # C minus 2 noise channels minus 2 labeled bursts

    def make_burst(freq, length_):
        burst_time = np.arange(length_)
        signal = np.sin(2 * np.pi * freq * burst_time / sampling_rate)
        return signal * tukey(length_, alpha=TAPER_ALPHA)

    signals = []
    for i in range(N):
        # 2 pure-noise (Laplace) background sources -- "noise as is".
        background = rng.laplace(size=(2, T))
        for c in range(2):
            sources[i, c] = normalize_energy(background[c])

        # 6 unlabeled oscillatory distractor sources: hardcoded frequency,
        # independently random duration and start time each trial.
        for j, freq in enumerate(distractor_freqs):
            length_ = rng.integers(min_duration, max_duration + 1)
            start = rng.integers(0, T - min_duration)
            end = min(start + length_, T)
            burst = make_burst(freq, end - start)
            sources[i, 2 + j, start:end] = scale * normalize_energy(burst)

        # Supervised sources: one pulse source and one gamma source on
        # separate latent channels, both randomly timed and durationed,
        # later mixed into the observed data.
        pulse_length = rng.integers(min_duration, max_duration + 1)
        gamma_length = rng.integers(min_duration, max_duration + 1)
        pulse_start = rng.integers(0, T - min_duration)
        gamma_start = rng.integers(0, T - min_duration)

        labels[i, 0] = pulse_length
        labels[i, 1] = gamma_length
        pulse_lengths[i, 0] = pulse_length
        gamma_lengths[i, 1] = gamma_length
        pulse_starts[i] = pulse_start
        gamma_starts[i] = gamma_start

        pulse_end = min(pulse_start + pulse_length, T)
        gamma_end = min(gamma_start + gamma_length, T)

        # 10 Hz sine on supervised source 0. Tapered (rather than gated
        # on/off abruptly) so the burst's envelope has no hard
        # discontinuity: an abrupt rectangular on/off is itself broadband,
        # and convolves the sinusoid's clean frequency line with the
        # window's sinc-shaped spectrum, leaking energy into neighboring
        # frequency bins.
        step_signal = make_burst(step_freq, pulse_end - pulse_start)
        sources[i, C - 1, pulse_start:pulse_end] = scale * normalize_energy(step_signal)

        # Gamma wave on supervised source 1, independently timed from the pulse.
        gamma_signal = make_burst(gamma_freq, gamma_end - gamma_start)
        sources[i, C - 2, gamma_start:gamma_end] = scale * normalize_energy(gamma_signal)

        signals.append(
            (mixing_mat + noise_scale * rng.standard_normal(size=(C, C))) @ sources[i]
        )

    x = np.stack(signals)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, random_state=generator
    )
    # split sources the same way as the observed data, so that we can later check if the model is able to recover the supervised sources.
    sources_train, sources_test = train_test_split(
        sources, train_size=train_size, random_state=generator
    )

    metadata = {
        "mixing_mat": mixing_mat,
        "labels": labels,
        "pulse_lengths": pulse_lengths,
        "gamma_lengths": gamma_lengths,
        "pulse_starts": pulse_starts,
        "gamma_starts": gamma_starts,
        "distractor_freqs": distractor_freqs,
    }

    return x_train, y_train, x_test, y_test, sources_train, sources_test, metadata


def generate_multioutput_simulated_4(
    N=6000,
    C=10,
    T=1000,
    kappa=5,
    sampling_rate=1000,
    theta_freq=6.0,
    gamma_freq=40.0,
    aperiodic_exponent=1.5,
    min_duration=100,
    max_duration=500,
    burst_scale=5.0,
    train_size=0.8,
    generator=None,
    theta_idx=None,
    gamma_idx=None,
):
    """Fig. 1's data generator (originally one-off code in
    analysis scripts/divergence_neural_like_bursts.py), reproduced here as a
    reusable function.

    Unlike generate_multioutput_simulated_3, ALL C sources -- not just the 2
    supervised ones -- carry an independent 1/f^exponent aperiodic ("neural
    background") profile, built via spectral synthesis (Laplace-seeded white
    noise, colored by 1/f^(exponent/2) in the frequency domain; Laplace
    rather than Gaussian seeding preserves excess kurtosis after filtering,
    since jointly Gaussian sources would be unidentifiable by ICA). Two
    sources -- by default the last two, indices C-2 (theta_idx) and C-1
    (gamma_idx), overridable via those two arguments -- additionally carry a
    Tukey-tapered (alpha=0.5) burst -- theta-band (default 6Hz) and
    gamma-band (default 40Hz) respectively -- on top of their own 1/f floor,
    independently randomly timed and durationed per trial, at burst_scale
    relative to the (unit-energy) 1/f floor. The mixing matrix is drawn from
    the eigenspaces of the Hilbert matrix with condition number exp(kappa)
    (create_hilbert_matrix), rather than a standard-normal matrix.

    NOTE on theta_idx/gamma_idx: create_hilbert_matrix assigns exponentially
    spaced eigenvalues by index (lambda_i = exp(kappa*i/(C-1))), and columns
    near the top of that spectrum (high index) are geometrically much more
    mutually similar than columns spread further apart or drawn from the
    lower/mid spectrum -- e.g. at kappa=5, C=10, the default (C-2, C-1) =
    (8, 9) pair has cosine similarity 0.968 (5th-highest of all 45 column
    pairs), while (0, 9) has similarity 0.228 (the lowest). Since CSP (and
    any second-order-statistics method) can only ever recover a source
    direction up to how distinguishable its true column is from the others,
    placing theta/gamma at two indices that are already nearly collinear in
    A makes them inherently hard to cleanly separate regardless of method --
    overriding theta_idx/gamma_idx to two less-similar indices (e.g. 0, 9)
    removes that confound.

    Labels are the theta/gamma burst lengths (0 outside a trial's burst
    window is not represented; a trial always has exactly one burst of each
    type), matching generate_multioutput_simulated_3's labels[:, 0]/[:, 1]
    convention for its 2 supervised sources.
    """
    TAPER_ALPHA = 0.5
    if theta_idx is None:
        theta_idx = C - 2
    if gamma_idx is None:
        gamma_idx = C - 1

    def generate_1f_noise(rng):
        white = rng.laplace(size=T)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)
        scale = np.zeros_like(freqs)
        scale[1:] = 1.0 / (freqs[1:] ** (aperiodic_exponent / 2.0))
        colored = np.fft.irfft(spectrum * scale, n=T)
        return colored.astype(np.float32)

    def make_burst(freq, length_):
        burst_time = np.arange(length_)
        signal = np.sin(2 * np.pi * freq * burst_time / sampling_rate)
        return signal * tukey(length_, alpha=TAPER_ALPHA)

    rng = np.random.default_rng(generator)

    sources = np.zeros((N, C, T), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            sources[i, c] = normalize_energy(generate_1f_noise(rng))

    mixing_mat, cond_number = create_hilbert_matrix(C, kappa)
    mixing_mat = mixing_mat.astype(np.float32)

    labels = np.zeros((N, 2), dtype=np.float32)
    theta_lengths = np.zeros(N, dtype=np.int32)
    gamma_lengths = np.zeros(N, dtype=np.int32)
    theta_starts = np.zeros(N, dtype=np.int32)
    gamma_starts = np.zeros(N, dtype=np.int32)

    for i in range(N):
        theta_base = sources[i, theta_idx]
        gamma_base = sources[i, gamma_idx]

        t_len = rng.integers(min_duration, max_duration + 1)
        g_len = rng.integers(min_duration, max_duration + 1)
        t_start = rng.integers(0, T - min_duration)
        g_start = rng.integers(0, T - min_duration)
        t_end = min(t_start + t_len, T)
        g_end = min(g_start + g_len, T)
        theta_starts[i], gamma_starts[i] = t_start, g_start
        theta_lengths[i], gamma_lengths[i] = t_end - t_start, g_end - g_start
        labels[i, 0] = t_end - t_start
        labels[i, 1] = g_end - g_start

        theta_burst = burst_scale * normalize_energy(make_burst(theta_freq, t_end - t_start))
        gamma_burst = burst_scale * normalize_energy(make_burst(gamma_freq, g_end - g_start))

        theta_signal = theta_base.copy()
        theta_signal[t_start:t_end] += theta_burst
        gamma_signal = gamma_base.copy()
        gamma_signal[g_start:g_end] += gamma_burst

        sources[i, theta_idx] = normalize_energy(theta_signal)
        sources[i, gamma_idx] = normalize_energy(gamma_signal)

    x = np.einsum("ij,njt->nit", mixing_mat, sources).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, random_state=generator
    )
    sources_train, sources_test = train_test_split(
        sources, train_size=train_size, random_state=generator
    )

    metadata = {
        "mixing_mat": mixing_mat,
        "cond_number": cond_number,
        "labels": labels,
        "theta_freq": theta_freq,
        "gamma_freq": gamma_freq,
        "aperiodic_exponent": aperiodic_exponent,
        "theta_starts": theta_starts,
        "gamma_starts": gamma_starts,
        "theta_lengths": theta_lengths,
        "gamma_lengths": gamma_lengths,
        "theta_idx": theta_idx,
        "gamma_idx": gamma_idx,
    }

    return x_train, y_train, x_test, y_test, sources_train, sources_test, metadata


def generate_multioutput_simulated_3_legacy(
    dataset_name="mini",
    M=2,
    noise_scale=0.0,
    signal_scale=10,
    snr=1,
    cond_number=5,
    train_size=0.8,
    generator=None,
):
    """Preserved, byte-for-byte original version of
    generate_multioutput_simulated_3 (C=10: 8 pure-Laplace background
    sources + pulse [fixed 500ms start] + gamma [random start], no taper,
    no distractors), kept so results already cached/fit against this exact
    data distribution (e.g. the raw-amplitude 9x8 grid sweep,
    simulated_data_SpecMLP_lr_grid_sweep_shared_bin_presence_classifier_multitaper_rawamplitude_20k.py)
    can still be correctly re-visualized/re-validated with matching ground
    truth, since generate_multioutput_simulated_3 itself was later changed
    (Tukey-tapered bursts, randomized pulse start, 6 oscillatory
    distractors replacing 6 of the 8 background channels) and calling it
    with the same seed no longer reproduces the same dataset.
    """
    rng = np.random.default_rng(generator)
    if dataset_name == "mini":
        N, C, T = 1000, 10, 2 * 10**3
    elif dataset_name == "tall":
        N, C, T = 10**4, 10, 2 * 10**3
    else:
        raise ValueError

    assert M <= C
    assert M >= 2
    sampling_rate = 10**3

    mixing_mat = rng.standard_normal(size=(C, C))

    pulse_start = int(0.5 * sampling_rate)  # 500 ms
    scale = signal_scale * np.sqrt(snr)

    sources = np.zeros((N, C, T), dtype=np.float32)
    labels = np.zeros((N, M), dtype=np.float32)
    pulse_lengths = np.zeros((N, M), dtype=np.int32)
    gamma_lengths = np.zeros((N, M), dtype=np.int32)
    step_freq = 10.0
    gamma_freq = 90.0

    signals = []
    for i in range(N):
        background = rng.laplace(size=(C - M, T))
        for c in range(C - M):
            sources[i, c] = normalize_energy(background[c])

        pulse_length = rng.integers(100, 501)
        gamma_length = rng.integers(100, 501)

        labels[i, 0] = pulse_length
        labels[i, 1] = gamma_length
        pulse_lengths[i, 0] = pulse_length
        gamma_lengths[i, 1] = gamma_length

        pulse_end = pulse_start + pulse_length
        gamma_start = rng.integers(1000, 1501)
        gamma_end = min(gamma_start + gamma_length, T)

        step_time = np.arange(pulse_end - pulse_start)
        step_signal = np.sin(2 * np.pi * step_freq * step_time / sampling_rate)
        sources[i, C - 1, pulse_start:pulse_end] = scale * normalize_energy(step_signal)

        gamma_time = np.arange(gamma_end - gamma_start)
        gamma_signal = np.sin(2 * np.pi * gamma_freq * gamma_time / sampling_rate)
        sources[i, C - 2, gamma_start:gamma_end] = scale * normalize_energy(gamma_signal)

        signals.append(
            (mixing_mat + noise_scale * rng.standard_normal(size=(C, C))) @ sources[i]
        )

    x = np.stack(signals)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, train_size=train_size, random_state=generator
    )
    sources_train, sources_test = train_test_split(
        sources, train_size=train_size, random_state=generator
    )

    metadata = {
        "mixing_mat": mixing_mat,
        "labels": labels,
        "pulse_lengths": pulse_lengths,
        "gamma_lengths": gamma_lengths,
    }

    return x_train, y_train, x_test, y_test, sources_train, sources_test, metadata


####################################
# Portable dataset dispatcher (used by experiments/ and create_representations.ipynb)
####################################


def load_dataset(dataset="laplace_mini", data_path="data/", seed=0, task=None, params={"noise_scale": 0.1, "cond_number": 3}):
    # assert task in ["regression", "classification"]
    if not os.path.exists(data_path):
        raise ValueError(
            f"Invalid 'data_path': '{data_path}'! Please make sure data directory exists."
        )
    data_dir = os.path.join(data_path, dataset)
    if dataset == "laplace_mini":
        C = 10
        T = int(1e3)
        N = 100

        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append(A @ s[i])
        x = np.stack(signals)
        x -= x.mean(axis=-1)[:, :, None]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "laplace_tall":
        C = 10
        T = int(1e6)
        N = 10
        noise_scale = params["noise_scale"]
        
        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append((A + noise_scale * np.random.normal(size=A.shape)) @ s[i])
        x = np.stack(signals)
        x -= x.mean(axis=-1)[:, :, None]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "laplace_wide":
        C = 30
        T = int(1e5)
        N = 10

        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        A = np.random.normal(size=(C, C))
        signals = []
        for i in range(N):
            signals.append((A + noise_scale * np.random.normal(size=A.shape)) @ s[i])
        x = np.stack(signals)

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        return x[train_idx], x[test_idx], {"mixing_mat": A}
    elif dataset == "multioutput_simulated":
        C = 10
        T = int(1e3)
        N = 6000
        M = 5
        alpha = params["cond_number"]
        noise_scale = params["noise_scale"]
        A, cond_number = create_hilbert_matrix(C, alpha)
    
        np.random.seed(seed)
        s = np.random.laplace(size=(N, C, T))
        n_features = 41 * 26 # n_cells in spectogram for T = 1000
        label_mat = np.random.normal(size=(M, n_features))
        signals = []
        labels = []
        for i in range(N):
            signals.append(A @ s[i])
            h = extract_spectrogram_features(torch.from_numpy(s[i]))
            h = torch.flatten(h, start_dim=1).numpy()

            # only consider the first M sources
            labels.append((label_mat * h[:M]).sum(axis=1))
        x = np.stack(signals)
        labels = np.stack(labels).astype(np.float32) # [N * M]

        train_idx = np.random.choice(N, size=(int(0.8 * N)), replace=False)
        test_idx = np.delete(np.arange(N), train_idx)
        assert len(np.union1d(train_idx, test_idx)) == N
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        return x_train, y_train, x_test, y_test, {"mixing_mat": A, "label_mat": label_mat}
    else:
        raise ValueError(f"No dataset found at {os.path.join(data_path, dataset)}!")
    
    
if __name__ == "__main__":
    for dataset in ["laplace_mini", "laplace_tall", "laplace_wide"]:    
        data, _ = load_dataset(dataset=dataset)
        print(data.shape)
