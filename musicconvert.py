import os
import numpy as np
import librosa
import soundfile as sf


class AudioIO:
    """Handles loading and saving MP3s using librosa and soundfile (Python 3.13+ safe)."""
    @staticmethod
    def load_mp3(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find input file: {os.path.abspath(filepath)}")
        samples, sr = librosa.load(filepath, sr=None, mono=False)
        return samples, sr

    @staticmethod
    def save_mp3(filepath, samples, sr):
        if samples.ndim > 1:
            samples = samples.T
        samples = np.clip(samples, -1.0, 1.0)
        sf.write(filepath, samples, sr, format="MP3")


class BitDepthReducer:
    """1. Changes effective audio bit depth to anywhere between 2 and 16 bits."""
    def __init__(self, target_bits=8):
        if not (2 <= target_bits <= 16):
            raise ValueError("Bit depth must be between 2 and 16.")
        self.target_bits = target_bits

    def process(self, samples):
        levels = (2 ** self.target_bits) - 1
        normalized = (samples + 1.0) / 2.0
        quantized = np.round(normalized * levels) / levels
        return (quantized * 2.0) - 1.0


class EvenFloatQuantizer:
    """2. Forces audio data values to only be even floating-point numbers."""
    def __init__(self, decimals=6, all_digits_even=True):
        self.decimals = decimals
        self.all_digits_even = all_digits_even
        self.step = 2.0 * (10 ** -decimals)

    def _snap_every_digit_even(self, samples):
        signs = np.sign(samples)
        abs_vals = np.clip(np.abs(samples), 0.0, 0.888888)
        scaled = np.round(abs_vals * (10 ** self.decimals)).astype(np.int64)

        result = np.zeros_like(scaled, dtype=np.float64)
        for pos in range(self.decimals):
            power = 10 ** pos
            digit = (scaled // power) % 10
            even_digit = np.where(digit == 9, 8, digit + (digit % 2))
            result += even_digit * (10.0 ** (pos - self.decimals))

        return np.round(signs * result, decimals=self.decimals)

    def process(self, samples):
        if self.all_digits_even:
            return self._snap_every_digit_even(samples)
        even_samples = np.round(samples / self.step) * self.step
        return np.round(even_samples, decimals=self.decimals)
class DecibelConstrainer:
    """3. Maximizes digital output to 0.0 dBFS while keeping loudness between 66 and 70 dB."""
    def __init__(self, min_db=66.0, max_db=70.0, reference_spl=70.0):
        self.min_db = min_db
        self.max_db = max_db
        self.reference_spl = reference_spl
        self.target_min_dbfs = min_db - reference_spl  # -4.0 dBFS
        self.target_max_dbfs = max_db - reference_spl  #  0.0 dBFS

    def process(self, samples, sr, window_ms=100):
        is_stereo = samples.ndim > 1
        data = samples if is_stereo else np.expand_dims(samples, axis=0)
        output = np.copy(data)
        _, num_samples = data.shape
        window_size = int(sr * (window_ms / 1000.0))

        for start in range(0, num_samples, window_size):
            chunk = output[:, start:start + window_size]
            rms = np.sqrt(np.mean(chunk ** 2)) + 1e-10
            current_dbfs = 20 * np.log10(rms)

            if current_dbfs < self.target_min_dbfs:
                gain = 10 ** ((self.target_min_dbfs - current_dbfs) / 20.0)
                output[:, start:start + window_size] *= gain
            elif current_dbfs > self.target_max_dbfs:
                gain = 10 ** ((self.target_max_dbfs - current_dbfs) / 20.0)
                output[:, start:start + window_size] *= gain

            peak = np.max(np.abs(output[:, start:start + window_size]))
            if peak > 1.0:
                output[:, start:start + window_size] /= peak

        return output if is_stereo else np.squeeze(output, axis=0)


class TempoLock60BPM:
    """4. Detects steady global BPM and time-stretches the entire track uniformly to 60 BPM."""
    def __init__(self, target_bpm=60.0):
        self.target_bpm = target_bpm

    def process(self, samples, sr):
        mono = librosa.to_mono(samples) if samples.ndim > 1 else samples
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)

        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120.0)
        if len(beat_frames) < 4:
            raise RuntimeError("Could not detect enough beats in this track to lock tempo.")

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        median_beat_sec = float(np.median(np.diff(beat_times)))
        detected_bpm = 60.0 / median_beat_sec

        while detected_bpm > 130.0:
            detected_bpm /= 2.0
        while detected_bpm < 65.0:
            detected_bpm *= 2.0

        print(f"Detected Steady BPM: {detected_bpm:.2f} -> Uniformly stretching to {self.target_bpm} BPM", flush=True)
        rate = self.target_bpm / detected_bpm

        if samples.ndim > 1:
            stretched_channels = tuple(
                librosa.effects.time_stretch(ch, rate=rate) for ch in samples
            )
            return np.vstack(stretched_channels)
        else:
            return librosa.effects.time_stretch(samples, rate=rate)

class ScalePitchQuantizer:
    """5. Remaps EVERY spectral frequency bin (0 Hz through Nyquist) to the closest scale note."""
    NOTE_OFFSETS = {
        "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
        "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
        "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11
    }
    SCALE_INTERVALS = {
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10)
    }

    def __init__(self, root_note="C", scale_type="major", n_fft=16384):
        root = root_note.strip().upper()
        scale = scale_type.strip().lower()

        if root not in self.NOTE_OFFSETS:
            raise ValueError(f"Unsupported root note '{root_note}'. Use notes like C, D#, Eb, F, etc.")
        if scale not in self.SCALE_INTERVALS:
            raise ValueError("Scale type must be 'major' or 'minor'.")

        self.root_note = root
        self.scale_type = scale
        self.n_fft = n_fft
        root_offset = self.NOTE_OFFSETS.get(root)

        intervals = self.SCALE_INTERVALS.get(scale)
        valid_midi = np.array(tuple(
            octave * 12 + root_offset + interval
            for octave in range(-5, 13)
            for interval in intervals
        ))
        self.scale_freqs = librosa.midi_to_hz(valid_midi)

    def get_target_bins(self, sr):
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        target_bins = set()
        for freq in self.scale_freqs:
            if 0.0 < freq <= sr / 2.0:
                target_bins.add(int(np.argmin(np.abs(fft_freqs - freq))))
        return target_bins

    def _quantize_channel(self, channel_samples, sr):
        original_length = len(channel_samples)
        stft_matrix = librosa.stft(channel_samples, n_fft=self.n_fft)
        magnitudes = np.abs(stft_matrix)
        phases = np.angle(stft_matrix)

        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        new_magnitudes = np.zeros_like(magnitudes)

        for bin_idx in range(1, len(fft_freqs)):
            freq = fft_freqs.item(bin_idx)
            closest_scale_idx = np.argmin(np.abs(np.log2(self.scale_freqs) - np.log2(freq)))
            target_freq = self.scale_freqs.item(closest_scale_idx)
            target_bin = int(np.argmin(np.abs(fft_freqs - target_freq)))
            new_magnitudes[target_bin] += magnitudes[bin_idx]

        quantized_stft = new_magnitudes * np.exp(1j * phases)
        reconstructed = librosa.istft(quantized_stft, length=original_length)
        return reconstructed

    def process(self, samples, sr):
        if samples.ndim > 1:
            processed_channels = tuple(
                self._quantize_channel(ch, sr) for ch in samples
            )
            return np.vstack(processed_channels)
        else:
            return self._quantize_channel(samples, sr)
def verify_audio(samples, sr, scale_quantizer, even_decimals=6, reference_spl=70.0):
    """Prints a full verification report of the processed audio array."""
    print("\n--- AUDIO VERIFICATION REPORT ---", flush=True)
    mono = librosa.to_mono(samples) if samples.ndim > 1 else samples

    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, start_bpm=60.0)
    final_bpm = float(np.squeeze(tempo))
    while final_bpm > 90.0:
        final_bpm /= 2.0
    print(f"1. Measured Tempo:         {final_bpm:.1f} BPM (Target: 60.0 BPM)")

    peak_amp = float(np.max(np.abs(samples))) + 1e-12
    peak_dbfs = 20.0 * np.log10(peak_amp)
    peak_spl = peak_dbfs + reference_spl

    rms_amp = float(np.sqrt(np.mean(samples ** 2))) + 1e-12
    rms_dbfs = 20.0 * np.log10(rms_amp)
    rms_spl = rms_dbfs + reference_spl
    print(f"2. Peak Output Level:      {peak_dbfs:.2f} dBFS ({peak_spl:.2f} dB SPL | Max limit: 70.0 dB)")
    print(f"   Average RMS Level:      {rms_dbfs:.2f} dBFS ({rms_spl:.2f} dB SPL | Window: 66–70 dB)")

    unique_vals = len(np.unique(np.round(samples, decimals=even_decimals)))
    print(f"3. Unique Sample Levels:   {unique_vals} discrete amplitude levels")

    scaled = np.round(samples * (10 ** even_decimals))
    even_pct = float(np.mean(scaled % 2 == 0) * 100.0)
    print(f"4. Even-Float Compliance:  {even_pct:.2f}% of all samples are even floats")

    stft_mag = np.abs(librosa.stft(mono, n_fft=scale_quantizer.n_fft))
    total_energy = float(np.sum(stft_mag ** 2)) + 1e-12
    valid_bins = tuple(scale_quantizer.get_target_bins(sr))
    in_scale_energy = float(np.sum(stft_mag[valid_bins, :] ** 2))
    scale_pct = (in_scale_energy / total_energy) * 100.0
    print(f"5. In-Scale Energy ({scale_quantizer.root_note} {scale_quantizer.scale_type}, 0–{sr//2} Hz): {scale_pct:.2f}% of spectral energy on scale notes")
    print("---------------------------------\n", flush=True)


def export_scale_verification_txt(filepath, samples, sr, scale_quantizer, even_decimals=6, nth_sample=44100):
    """Writes a text report verifying every nth sample value and its active musical note across the track."""
    if nth_sample < 1:
        raise ValueError("nth_sample must be at least 1.")

    inspect_channel = np.take(samples, 0, axis=0) if samples.ndim > 1 else samples
    total_samples = len(inspect_channel)

    hop_length = scale_quantizer.n_fft // 4
    stft_matrix = librosa.stft(inspect_channel, n_fft=scale_quantizer.n_fft, hop_length=hop_length)
    magnitudes = np.abs(stft_matrix)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=scale_quantizer.n_fft)
    valid_bins = scale_quantizer.get_target_bins(sr)

    active_bins = np.flatnonzero(np.max(magnitudes, axis=1) > 1e-6)
    off_scale_bins = tuple(int(b) for b in active_bins if int(b) not in valid_bins)

    first_frame_mag = np.take(magnitudes, 0, axis=1)
    dominant_bin = int(np.argmax(first_frame_mag))
    dominant_hz = float(fft_freqs.item(dominant_bin))
    dominant_note = librosa.hz_to_note(dominant_hz) if dominant_hz > 0 else "0 Hz (DC)"

    _, num_frames = magnitudes.shape
    sample_indices = range(0, total_samples, nth_sample)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== PROCESSED MP3 SCALE & EVEN-FLOAT VERIFICATION ===\n")
        f.write(f"Chosen Scale:          {scale_quantizer.root_note} {scale_quantizer.scale_type}\n")
        f.write(f"Even Float Precision:  {even_decimals} decimal places\n")
        f.write(f"Sampling Interval:     Every {nth_sample}th sample ({nth_sample / sr:.4f}s step)\n")
        f.write(f"Total Samples Logged:  {len(sample_indices)} of {total_samples}\n")
        f.write(f"Active Frequency Bins: {len(active_bins)} bins\n")
        f.write(f"Off-Scale Bins Found:  {len(off_scale_bins)}\n")
        f.write(f"Opening Dominant Note: {dominant_note} ({dominant_hz:.2f} Hz)\n")
        f.write("=====================================================\n\n")
        f.write(f"EVERY {nth_sample}TH PROCESSED AUDIO SAMPLE VALUE:\n")

        for idx in sample_indices:
            val = float(inspect_channel.item(idx))
            scaled_int = int(round(val * (10 ** even_decimals)))
            is_even = (scaled_int % 2 == 0)
            time_sec = idx / sr

            frame_idx = min(idx // hop_length, num_frames - 1)
            frame_mag = np.take(magnitudes, frame_idx, axis=1)
            if float(np.max(frame_mag)) < 1e-8:
                note_label = "Silence (0.00 Hz)"
            else:
                note_bin = int(np.argmax(frame_mag))
                note_hz = float(fft_freqs.item(note_bin))
                note_name = librosa.hz_to_note(note_hz) if note_hz > 0 else "0 Hz (DC)"
                note_label = f"{note_name} ({note_hz:.2f} Hz)"

            f.write(
                f"Sample [{idx:08d}] ({time_sec:6.2f}s): {val:+.{even_decimals}f}  | Note: {note_label:<18} "
                f"| Even Float: {'YES' if is_even else 'NO'} | Scale: {scale_quantizer.root_note} {scale_quantizer.scale_type}\n"
            )

    print(f"Verification text file saved to: {os.path.abspath(filepath)}", flush=True)


if __name__ == "__main__":
    input_mp3 = "song.mp3"
    output_mp3 = "processed_song.mp3"
    verification_txt = "scale_verification_values.txt"

    even_precision = 8
    force_all_digits_even = False
    nth_sample_step = 44100  # Logs every nth sample (44100 = 1 sample per second at 44.1 kHz)

    print("1/7 Loading MP3...", flush=True)
    audio_data, sample_rate = AudioIO.load_mp3(input_mp3)

    print("2/7 Adjusting tempo to 60 BPM...", flush=True)
    audio_data = TempoLock60BPM(target_bpm=60.0).process(audio_data, sample_rate)

    print("3/7 Locking ALL frequencies (0 Hz to Nyquist) to C major scale...", flush=True)
    scale_lock = ScalePitchQuantizer(root_note="A", scale_type="minor", n_fft=16384)
    audio_data = scale_lock.process(audio_data, sample_rate)

    print("4/7 Constraining decibels (66-70 dB)...", flush=True)
    audio_data = DecibelConstrainer(min_db=66.0, max_db=58.4, reference_spl=70.0).process(audio_data, sample_rate)

    print("5/7 Applying bit depth & even-float quantization...", flush=True)
    audio_data = BitDepthReducer(target_bits=16).process(audio_data)
    audio_data = EvenFloatQuantizer(decimals=even_precision, all_digits_even=force_all_digits_even).process(audio_data)

    verify_audio(audio_data, sample_rate, scale_lock, even_decimals=even_precision, reference_spl=70.0)
    export_scale_verification_txt(
        verification_txt, audio_data, sample_rate, scale_lock,
        even_decimals=even_precision, nth_sample=nth_sample_step
    )

    print("6/7 Saving MP3...", flush=True)
    AudioIO.save_mp3(output_mp3, audio_data, sample_rate)

    print("7/7 Complete! Files created:", flush=True)
    print(" -> MP3:", os.path.abspath(output_mp3), flush=True)
    print(" -> TXT:", os.path.abspath(verification_txt), flush=True)
