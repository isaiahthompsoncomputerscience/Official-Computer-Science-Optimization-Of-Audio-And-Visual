import os
import math
import numpy as np
import soundfile as sf


class MP3FFTSampleNoteAnalyzer:
    """
    Decodes an MP3 file on Windows 11 into exact 32-bit floating-point samples,
    detects the true musical pitch at every nth sample using windowed FFT with
    parabolic interpolation, and exports the full report and float32 audio file.
    """

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, mp3_path: str, fft_window_size: int = 4096):
        if not os.path.isfile(mp3_path):
            raise FileNotFoundError(f"Audio file not found: {mp3_path}")
        self.mp3_path = mp3_path
        self.fft_window_size = fft_window_size
        self.samples = None
        self.sample_rate = None
        self.channels = None
        self._window_func = np.hanning(self.fft_window_size)

    def load_audio(self) -> None:
        """Loads the MP3 file as exact 32-bit floating-point PCM values."""
        data, samplerate = sf.read(self.mp3_path, dtype="float32", always_2d=True)
        self.samples = data
        self.sample_rate = samplerate
        self.channels = data.shape[1]

    def detect_pitch_at_index(self, channel_data: np.ndarray, center_idx: int) -> tuple[float, str]:
        """
        Extracts a Hann-windowed slice centered at `center_idx`, computes the FFT,
        refines the peak frequency via parabolic interpolation, and returns (freq_hz, note_str).
        """
        half_win = self.fft_window_size // 2
        total_len = len(channel_data)

        start = center_idx - half_win
        end = start + self.fft_window_size

        # Zero-pad cleanly if the window extends past the start or end of the track
        slice_buf = np.zeros(self.fft_window_size, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(total_len, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        if src_end > src_start:
            slice_buf[dst_start:dst_end] = channel_data[src_start:src_end]

        # Skip silent or near-silent frames (below -80 dBFS)
        rms = float(np.sqrt(np.mean(slice_buf ** 2)))
        if rms < 1e-4:
            return 0.0, "Silence"

        # Apply Hann window and compute real FFT magnitudes
        windowed = slice_buf * self._window_func
        spectrum = np.abs(np.fft.rfft(windowed))

        # Ignore DC offset (bin 0)
        spectrum[0] = 0.0
        peak_bin = int(np.argmax(spectrum))

        # Parabolic interpolation around the peak bin for sub-bin frequency accuracy
        if 0 < peak_bin < len(spectrum) - 1:
            alpha = spectrum[peak_bin - 1]
            beta = spectrum[peak_bin]
            gamma = spectrum[peak_bin + 1]
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if denom != 0.0 else 0.0
        else:
            delta = 0.0

        true_bin = peak_bin + delta
        freq_hz = true_bin * (self.sample_rate / self.fft_window_size)

        if freq_hz <= 0.0:
            return 0.0, "Silence"

        note_label = self.hz_to_note(freq_hz)
        return freq_hz, note_label

    def hz_to_note(self, freq_hz: float) -> str:
        """Converts any frequency in Hz to the closest 12-TET musical note and cent offset."""
        midi_exact = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
        midi_round = int(round(midi_exact))
        cents = (midi_exact - midi_round) * 100.0

        note_name = self.NOTE_NAMES[midi_round % 12]
        octave = (midi_round // 12) - 1
        return f"{note_name}{octave} ({freq_hz:.2f} Hz, {cents:+.0f}c)"

    def analyze_and_export(
        self,
        nth_sample: int = 441,
        output_txt_path: str = "fft_sample_note_report.txt",
        export_audio_copy_path: str = None,
        print_to_console: bool = True,
    ) -> None:
        """
        Iterates through every nth sample across the entire track, prints the exact
        floating-point sample value and FFT-detected musical note, and writes to disk.
        """
        if nth_sample < 1:
            raise ValueError("nth_sample must be an integer >= 1.")

        if self.samples is None:
            self.load_audio()

        total_frames = self.samples.shape[0]
        duration_sec = total_frames / self.sample_rate

        header = (
            f"Source File     : {os.path.abspath(self.mp3_path)}\n"
            f"Sample Rate     : {self.sample_rate} Hz\n"
            f"Channels        : {self.channels}\n"
            f"Total Frames    : {total_frames}\n"
            f"Duration        : {duration_sec:.4f} seconds\n"
            f"Step Interval   : Every {nth_sample} sample(s)\n"
            f"FFT Window Size : {self.fft_window_size} samples (~{(self.fft_window_size / self.sample_rate) * 1000:.1f} ms)\n"
            + "=" * 95
            + "\n"
        )

        if print_to_console:
            print(header, end="")

        with open(output_txt_path, "w", encoding="utf-8") as report_file:
            report_file.write(header)

            for idx in range(0, total_frames, nth_sample):
                timestamp = idx / self.sample_rate

                if self.channels == 1:
                    val = float(self.samples[idx, 0])
                    _, note = self.detect_pitch_at_index(self.samples[:, 0], idx)
                    line = (
                        f"Sample {idx:<10d} | {timestamp:8.4f}s | "
                        f"Float: {val:+.10f} | Pitch: {note}\n"
                    )
                else:
                    left_val = float(self.samples[idx, 0])
                    right_val = float(self.samples[idx, 1])
                    _, left_note = self.detect_pitch_at_index(self.samples[:, 0], idx)
                    _, right_note = self.detect_pitch_at_index(self.samples[:, 1], idx)
                    line = (
                        f"Sample {idx:<10d} | {timestamp:8.4f}s | "
                        f"L: {left_val:+.10f} [{left_note}] | "
                        f"R: {right_val:+.10f} [{right_note}]\n"
                    )

                report_file.write(line)
                if print_to_console:
                    print(line, end="")

        if export_audio_copy_path:
            sf.write(export_audio_copy_path, self.samples, self.sample_rate, subtype="FLOAT")
            msg = f"\nSaved 32-bit float audio file to: {os.path.abspath(export_audio_copy_path)}\n"
            if print_to_console:
                print(msg)


if __name__ == "__main__":
    input_mp3 = r"C:\yolo_project\track.mp3"
    output_report = r"C:\yolo_project\mp3_fft_samples_notes.txt"
    output_float_wav = r"C:\yolo_project\decoded_float32_audio.wav"

    # fft_window_size=4096 gives ~10.7 Hz bin resolution at 44.1 kHz,
    # refined further by parabolic peak interpolation.
    analyzer = MP3FFTSampleNoteAnalyzer(input_mp3, fft_window_size=4096)

    analyzer.analyze_and_export(
        nth_sample=441,
        output_txt_path=output_report,
        export_audio_copy_path=output_float_wav,
        print_to_console=True,
    )
