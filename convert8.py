import os
import subprocess
import numpy as np
import cv2
import imageio_ffmpeg


class ColorQuantizer32FPS:
    def __init__(self, target_fps=32, max_seconds=128, group_size=4):
        self.target_fps = target_fps
        self.max_seconds = max_seconds      # 128s = 2 minutes and 8 seconds
        self.group_size = group_size        # 4 frames per color block
        self.max_frames = self.target_fps * self.max_seconds  # 4,096 frames
        self.color_names = [
            "Pure Blue", "Cerulean Blue", "Sky Blue", "Aquamarine",
            "Mint Green", "Kelly Green", "Lime Green", "Chartreuse",
            "Pure Yellow", "Safety Orange", "Red-Orange", "Pure Red",
            "Fuchsia / Pink", "Magenta", "Deep Purple", "Indigo / Violet"
        ]
        self.palette_rgb = self._build_16_pure_palette()

    def _build_16_pure_palette(self):
        """Builds the 16 pure clockwise Blue-start colors as an (16, 3) RGB array."""
        palette_rgb = []
        step_size = 1530 / 16
        blue_start_offset = 1020  # Pure Blue coordinate on the 1,530 perimeter

        for i in range(16):
            hue_step = (blue_start_offset - (i * step_size)) % 1530

            if hue_step < 255:
                r, g, b = 255, int(hue_step), 0
            elif hue_step < 510:
                r, g, b = int(510 - hue_step), 255, 0
            elif hue_step < 765:
                r, g, b = 0, 255, int(hue_step - 510)
            elif hue_step < 1020:
                r, g, b = 0, int(1020 - hue_step), 255
            elif hue_step < 1275:
                r, g, b = int(hue_step - 1020), 0, 255
            else:
                r, g, b = 255, 0, int(1530 - hue_step)

            palette_rgb.append((r, g, b))

        return np.array(palette_rgb, dtype=np.float32)

    def _get_average_rgb_of_group(self, frames_bgr):
        """Calculates the mean (R, G, B) color across a 4-frame block."""
        b_sum = 0.0
        g_sum = 0.0
        r_sum = 0.0

        for frame in frames_bgr:
            # cv2.mean returns (mean_B, mean_G, mean_R, alpha)
            mean_b, mean_g, mean_r, _ = cv2.mean(frame)
            b_sum += mean_b
            g_sum += mean_g
            r_sum += mean_r

        count = len(frames_bgr)
        return np.array([r_sum / count, g_sum / count, b_sum / count], dtype=np.float32)

    def _map_rgb_to_closest_pure_color(self, avg_rgb):
        """Finds the closest of the 16 pure colors using 3D Euclidean RGB distance."""
        distances = np.linalg.norm(self.palette_rgb - avg_rgb, axis=1)
        closest_idx = int(np.argmin(distances))
        matched_rgb = tuple(int(c) for c in self.palette_rgb[closest_idx])
        return matched_rgb, self.color_names[closest_idx], closest_idx

    def _build_labeled_frame(self, width, height, rgb, color_name, color_idx):
        """Creates a solid BGR frame with the current color name drawn in high-contrast text."""
        r, g, b = rgb
        frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

        # Use black text on bright yellow/green hues (indices 3-9) and white text on darker hues
        text_color = (0, 0, 0) if 3 <= color_idx <= 9 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, width / 600.0)
        thickness = max(2, int(font_scale * 2))

        (text_w, text_h), _ = cv2.getTextSize(color_name, font, font_scale, thickness)
        text_x = max(10, (width - text_w) // 2)
        text_y = max(text_h + 10, (height + text_h) // 2)

        cv2.putText(frame, color_name, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        return frame

    def process(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"[!] File not found: {input_path}")
            return

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_cfr_video = "temp_32fps_cfr.mp4"
        temp_quantized_silent = "temp_quantized_silent.mp4"

        print("Step 1/3: Resampling source video to exact 32 FPS (max 2m 08s)...")
        subprocess.run([
            ffmpeg_exe, "-y", "-i", input_path,
            "-t", str(self.max_seconds),
            "-vf", f"fps={self.target_fps}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-an", temp_cfr_video
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cap = cv2.VideoCapture(temp_cfr_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_quantized_silent, fourcc, self.target_fps, (width, height))

        print("Step 2/3: Computing average RGB color per 4 frames and mapping to 16 pure colors...")
        frame_buffer = []
        total_written = 0

        while total_written < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)

            if len(frame_buffer) == self.group_size:
                avg_rgb = self._get_average_rgb_of_group(frame_buffer)
                rgb, color_name, color_idx = self._map_rgb_to_closest_pure_color(avg_rgb)

                solid_frame = self._build_labeled_frame(width, height, rgb, color_name, color_idx)
                for _ in range(self.group_size):
                    out.write(solid_frame)
                    total_written += 1

                frame_buffer.clear()

        # Handle any leftover frames at the very end (< 4 frames)
        if frame_buffer and total_written < self.max_frames:
            avg_rgb = self._get_average_rgb_of_group(frame_buffer)
            rgb, color_name, color_idx = self._map_rgb_to_closest_pure_color(avg_rgb)
            solid_frame = self._build_labeled_frame(width, height, rgb, color_name, color_idx)
            for _ in range(len(frame_buffer)):
                if total_written < self.max_frames:
                    out.write(solid_frame)
                    total_written += 1

        cap.release()
        out.release()

        print("Step 3/3: Muxing synced original audio track into final 32 FPS MP4...")
        subprocess.run([
            ffmpeg_exe, "-y",
            "-i", temp_quantized_silent,
            "-i", input_path,
            "-t", str(total_written / self.target_fps),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Clean up temporary files
        for tmp in (temp_cfr_video, temp_quantized_silent):
            if os.path.exists(tmp):
                os.remove(tmp)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print("\n" + "=" * 50)
        print("RGB COLOR QUANTIZATION COMPLETE")
        print(f"Saved to: {os.path.abspath(output_path)}")
        print(f"Total Frames: {total_written} ({total_written // self.group_size} color blocks)")
        print(f"Duration: {total_written / self.target_fps:.2f} seconds")
        print(f"File Size: {size_mb:.2f} MB")
        print("=" * 50)


if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_32fps_16rgb_quantized.mp4"

    quantizer = ColorQuantizer32FPS()
    quantizer.process(input_video, output_video)
