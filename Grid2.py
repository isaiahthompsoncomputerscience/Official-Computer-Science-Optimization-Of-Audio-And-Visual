import gc
import os
import tempfile
import time
import cv2
import numpy as np
from moviepy import VideoFileClip


class BaseGridHueVideoConverter:
    def __init__(self, grid_size: int, output_scale: int = 1, line_thickness: int = 2):
        """
        Converts any MP4 into an NxN pixel grid mapped to 16 pure hue angles.
        :param grid_size: Grid dimension N (e.g., 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).
        :param output_scale: Multiplier to upscale the NxN grid for media players.
        :param line_thickness: Pixel thickness of black grid lines outlining each square.
        """
        self.grid_size = max(1, int(grid_size))
        self.output_scale = max(1, int(output_scale))
        self.line_thickness = max(0, int(line_thickness))

        # Exact 16 colors in RGB order
        self.palette_rgb = np.array(
            (
                (0, 0, 255),      # 1. Pure Blue (Start)
                (0, 95, 255),     # 2. Cerulean Blue
                (0, 191, 255),    # 3. Sky Blue
                (0, 255, 223),    # 4. Aquamarine / Cyan
                (0, 255, 127),    # 5. Mint Green
                (0, 255, 31),     # 6. Kelly Green
                (63, 255, 0),     # 7. Lime Green
                (159, 255, 0),    # 8. Chartreuse
                (255, 255, 0),    # 9. Pure Yellow
                (255, 159, 0),    # 10. Safety Orange
                (255, 63, 0),     # 11. Red-Orange
                (255, 0, 31),     # 12. Pure Red
                (255, 0, 127),    # 13. Fuchsia / Pink
                (255, 0, 223),    # 14. Magenta
                (191, 0, 255),    # 15. Deep Purple
                (95, 0, 255),     # 16. Indigo / Violet
            ),
            dtype=np.uint8,
        )

        self.palette_bgr = self.palette_rgb[:, ::-1]
        self.palette_hues = self._bgr_to_hue_degrees(self.palette_bgr)

    @staticmethod
    def _bgr_to_hue_degrees(bgr_pixels: np.ndarray) -> np.ndarray:
        """
        Computes pure hue angle in degrees [0, 360) directly from BGR values,
        completely ignoring brightness (Value) and Saturation.
        """
        bgr_norm = bgr_pixels.astype(np.float32) / 255.0
        b = bgr_norm[:, 0]
        g = bgr_norm[:, 1]
        r = bgr_norm[:, 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        hue = np.zeros_like(cmax)
        non_zero = delta > 1e-6

        mask_r = non_zero & (cmax == r)
        mask_g = non_zero & (cmax == g)
        mask_b = non_zero & (cmax == b)

        hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
        hue[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
        hue[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0

        return hue

    def _draw_grid_lines(self, upscaled_frame: np.ndarray) -> np.ndarray:
        """
        Draws solid black lines outlining every square in the upscaled grid
        when each cell is at least 4 pixels wide.
        """
        if self.line_thickness <= 0 or self.output_scale < 4:
            return upscaled_frame

        h, w = upscaled_frame.shape[:2]
        line_color = (0, 0, 0)

        for i in range(1, self.grid_size):
            pos = i * self.output_scale
            cv2.line(upscaled_frame, (pos, 0), (pos, h), line_color, self.line_thickness)
            cv2.line(upscaled_frame, (0, pos), (w, pos), line_color, self.line_thickness)

        cv2.rectangle(upscaled_frame, (0, 0), (w - 1, h - 1), line_color, self.line_thickness)
        return upscaled_frame

    def _map_frame_to_grid(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Downsamples frame to an NxN grid, extracts pure hue angles,
        maps each cell to the closest circular hue angle in the 16-color palette,
        upscales using nearest-neighbor interpolation, and outlines each square.
        """
        small_grid = cv2.resize(
            frame_bgr,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_AREA
        )
        pixels_bgr = small_grid.reshape(-1, 3)

        pixel_hues = self._bgr_to_hue_degrees(pixels_bgr)

        diff = np.abs(pixel_hues[:, np.newaxis] - self.palette_hues[np.newaxis, :])
        circular_dist = np.minimum(diff, 360.0 - diff)

        closest_indices = np.argmin(circular_dist, axis=1)
        mapped_grid = self.palette_bgr[closest_indices].reshape(
            self.grid_size, self.grid_size, 3
        )

        if self.output_scale > 1:
            target_dim = self.grid_size * self.output_scale
            upscaled = cv2.resize(
                mapped_grid,
                (target_dim, target_dim),
                interpolation=cv2.INTER_NEAREST
            )
            return self._draw_grid_lines(upscaled)

        return mapped_grid

    def convert(self, input_mp4_path: str, output_mp4_path: str) -> None:
        """
        Processes the input MP4 and writes the NxN pure-hue MP4 with synced audio.
        """
        if not os.path.exists(input_mp4_path):
            raise FileNotFoundError(f"Input video not found: {input_mp4_path}")

        cap = cv2.VideoCapture(input_mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_mp4_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dim = self.grid_size * self.output_scale
        tag = f"[{self.grid_size}x{self.grid_size} Hue]"

        temp_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_dim, out_dim))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._map_frame_to_grid(frame)
                out_writer.write(processed_frame)

                frame_idx += 1
                if frame_idx % 100 == 0 or frame_idx == total_frames:
                    print(f"{tag} Processed {frame_idx}/{total_frames} frames...", end="\r")

            print(f"\n{tag} Video frames complete. Muxing synced audio...")
            out_writer.release()
            cap.release()

            original_clip = VideoFileClip(input_mp4_path)
            processed_clip = VideoFileClip(temp_video_path)

            if original_clip.audio is not None:
                final_clip = processed_clip.with_audio(original_clip.audio)
            else:
                final_clip = processed_clip

            final_clip.write_videofile(
                output_mp4_path,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                logger=None
            )

            final_clip.close()
            processed_clip.close()
            original_clip.close()
            print(f"{tag} Saved successfully to: {output_mp4_path}")

        finally:
            if cap.isOpened():
                cap.release()
            gc.collect()
            for _ in range(10):
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    break
                except PermissionError:
                    time.sleep(0.2)
class Grid2x2HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 256, line_thickness: int = 2):
        super().__init__(grid_size=2, output_scale=output_scale, line_thickness=line_thickness)


class Grid4x4HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 128, line_thickness: int = 2):
        super().__init__(grid_size=4, output_scale=output_scale, line_thickness=line_thickness)


class Grid8x8HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 64, line_thickness: int = 2):
        super().__init__(grid_size=8, output_scale=output_scale, line_thickness=line_thickness)


class Grid16x16HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 32, line_thickness: int = 2):
        super().__init__(grid_size=16, output_scale=output_scale, line_thickness=line_thickness)


class Grid32x32HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 16, line_thickness: int = 2):
        super().__init__(grid_size=32, output_scale=output_scale, line_thickness=line_thickness)


class Grid64x64HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 8, line_thickness: int = 2):
        super().__init__(grid_size=64, output_scale=output_scale, line_thickness=line_thickness)


class Grid128x128HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 4, line_thickness: int = 1):
        super().__init__(grid_size=128, output_scale=output_scale, line_thickness=line_thickness)


class Grid256x256HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 2, line_thickness: int = 0):
        super().__init__(grid_size=256, output_scale=output_scale, line_thickness=line_thickness)


class Grid512x512HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 1, line_thickness: int = 0):
        super().__init__(grid_size=512, output_scale=output_scale, line_thickness=line_thickness)


class Grid1024x1024HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 1, line_thickness: int = 0):
        super().__init__(grid_size=1024, output_scale=output_scale, line_thickness=line_thickness)
if __name__ == "__main__":
    input_file = r"C:\yolo_project\sample.mp4"

    # All 10 grid converter classes (defaulting to 512x512 output except 1024x1024)
    Grid2x2HueVideoConverter().convert(input_file, r"C:\Videos\output_2x2_hue.mp4")
    Grid4x4HueVideoConverter().convert(input_file, r"C:\Videos\output_4x4_hue.mp4")
    #Grid8x8HueVideoConverter().convert(input_file, r"C:\Videos\output_8x8_hue.mp4")
    #Grid16x16HueVideoConverter().convert(input_file, r"C:\Videos\output_16x16_hue.mp4")
    #Grid32x32HueVideoConverter().convert(input_file, r"C:\Videos\output_32x32_hue.mp4")
    #Grid64x64HueVideoConverter().convert(input_file, r"C:\Videos\output_64x64_hue.mp4")
    Grid128x128HueVideoConverter().convert(input_file, r"C:\Videos\output_128x128_hue.mp4")
    #Grid256x256HueVideoConverter().convert(input_file, r"C:\Videos\output_256x256_hue.mp4")
    #Grid512x512HueVideoConverter().convert(input_file, r"C:\Videos\output_512x512_hue.mp4")
    #Grid1024x1024HueVideoConverter().convert(input_file, r"C:\Videos\output_1024x1024_hue.mp4")
