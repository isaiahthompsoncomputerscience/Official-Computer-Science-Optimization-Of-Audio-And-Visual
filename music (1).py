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
    def __init__(self, decimals=2):
        self.step = 2.0 * (10 ** -decimals)

    def process(self, samples):
        even_samples = np.round(samples / self.step) * self.step
        return np.round(even_samples, decimals=6)


class DecibelConstrainer:
    """3. Maximizes digital output to 0.0 dBFS while keeping loudness between 66 and 70 dB."""
    def __init__(self, min_db=66.0, max_db=70.0, reference_spl=70.0):
        self.target_min_dbfs = min_db - reference_spl  # -4.0 dBFS
        self.target_max_dbfs = max_db - reference_spl  #  0.0 dBFS

    def process(self, samples, sr, window_ms=100):
        is_stereo = samples.ndim > 1
        data = samples if is_stereo else samples[np.newaxis, :]
        output = np.copy(data)
        window_size = int(sr * (window_ms / 1000.0))

        for start in range(0, data.shape[1], window_size):
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

        return output if is_stereo else output[0]


class TempoLock60BPM:
    """4. Detects track BPM (correcting double-time detection) and time-stretches to 60 BPM."""
    def __init__(self, target_bpm=60.0):
        self.target_bpm = target_bpm

    def process(self, samples, sr):
        mono = librosa.to_mono(samples) if samples.ndim > 1 else samples
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr, start_bpm=100.0)
        detected_bpm = float(np.squeeze(tempo))

        if detected_bpm <= 0:
            raise RuntimeError("Could not detect a valid beat in this track.")

        while detected_bpm > 135.0:
            detected_bpm /= 2.0

        print(f"Detected BPM: {detected_bpm:.1f} -> Time-stretching to {self.target_bpm} BPM", flush=True)
        rate = self.target_bpm / detected_bpm

        if samples.ndim > 1:
            stretched_channels = [
                librosa.effects.time_stretch(ch, rate=rate) for ch in samples
            ]
            return np.vstack(stretched_channels)
        else:
            return librosa.effects.time_stretch(samples, rate=rate)


if __name__ == "__main__":
    input_mp3 = "song.mp3"
    output_mp3 = "processed_song.mp3"

    print("1/5 Loading MP3...", flush=True)
    audio_data, sample_rate = AudioIO.load_mp3(input_mp3)

    print("2/5 Adjusting tempo to 60 BPM (this may take 10-20 seconds)...", flush=True)
    audio_data = TempoLock60BPM(target_bpm=60.0).process(audio_data, sample_rate)

    print("3/5 Constraining decibels (66-70 dB)...", flush=True)
    audio_data = DecibelConstrainer(min_db=58.4, max_db=70.0, reference_spl=70.0).process(audio_data, sample_rate)

    print("4/5 Applying bit depth & even-float quantization...", flush=True)
    audio_data = BitDepthReducer(target_bits=16).process(audio_data)
    audio_data = EvenFloatQuantizer(decimals=2).process(audio_data)

    print("5/5 Saving MP3...", flush=True)
    AudioIO.save_mp3(output_mp3, audio_data, sample_rate)
    print("Done! Saved to:", os.path.abspath(output_mp3), flush=True)
