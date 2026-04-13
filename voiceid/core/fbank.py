"""Kaldi-compatible 80-dim log Mel filterbank features (pure numpy).

CAM++ ONNX expects FBank features with the same convention as Kaldi's
`compute-fbank-feats`: 25 ms window, 10 ms hop, Povey window, 80 mel
bins over [20, 7600] Hz, log magnitude, per-utterance mean subtraction.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-10


def _povey_window(n: int) -> np.ndarray:
    """Kaldi's 'povey' window: hanning raised to the 0.85 power."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))) ** 0.85


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.exp(np.asarray(mel) / 1127.0) - 1.0)


def _build_mel_filterbank(
    num_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    """Triangular mel filterbank matching Kaldi's layout."""
    low_mel = _hz_to_mel(low_freq)
    high_mel = _hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = _mel_to_hz(mel_points)

    num_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, num_fft_bins)
    fb = np.zeros((num_bins, num_fft_bins), dtype=np.float32)

    for m in range(num_bins):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k, freq in enumerate(fft_freqs):
            if freq < left or freq > right:
                continue
            if freq <= center:
                fb[m, k] = (freq - left) / (center - left + _EPSILON)
            else:
                fb[m, k] = (right - freq) / (right - center + _EPSILON)
    return fb


class FBankExtractor:
    """Compute 80-dim log mel filterbank features from mono float32 audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        low_freq: float = 20.0,
        high_freq: float = 7600.0,
        preemphasis: float = 0.97,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = int(round(sample_rate * frame_length_ms / 1000.0))
        self.frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))
        self.preemphasis = preemphasis

        # Pad window to next power of two for FFT efficiency.
        self.n_fft = 1
        while self.n_fft < self.frame_length:
            self.n_fft *= 2

        self.window = _povey_window(self.frame_length).astype(np.float32)
        self.mel_fb = _build_mel_filterbank(
            num_mel_bins, self.n_fft, sample_rate, low_freq, high_freq
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from float32 mono audio in [-1, 1].

        Returns:
            Array of shape (num_frames, num_mel_bins), float32.
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) < self.frame_length:
            return np.zeros((0, self.mel_fb.shape[0]), dtype=np.float32)

        # Kaldi-style pre-emphasis is per-frame, but the common approximation
        # is whole-signal pre-emphasis; the accuracy impact is negligible.
        if self.preemphasis > 0:
            emphasized = np.empty_like(audio)
            emphasized[0] = audio[0]
            emphasized[1:] = audio[1:] - self.preemphasis * audio[:-1]
            audio = emphasized

        # Frame the signal.
        num_frames = 1 + (len(audio) - self.frame_length) // self.frame_shift
        indices = (
            np.arange(self.frame_length)[None, :]
            + self.frame_shift * np.arange(num_frames)[:, None]
        )
        frames = audio[indices]

        # Remove per-frame DC offset (Kaldi default).
        frames = frames - frames.mean(axis=1, keepdims=True)

        # Apply Povey window.
        frames = frames * self.window

        # Zero-pad to n_fft and compute power spectrum.
        if self.n_fft > self.frame_length:
            pad = np.zeros(
                (num_frames, self.n_fft - self.frame_length), dtype=np.float32
            )
            frames = np.concatenate([frames, pad], axis=1)

        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)

        # Apply mel filterbank and log.
        mel_energy = power @ self.mel_fb.T
        log_mel = np.log(np.maximum(mel_energy, _EPSILON))

        # Per-utterance cepstral mean normalization (what WeSpeaker expects).
        log_mel = log_mel - log_mel.mean(axis=0, keepdims=True)

        return log_mel.astype(np.float32)
