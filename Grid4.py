import gc
import math
import os
import tempfile
import time
import cv2
import numpy as np
from moviepy import VideoFileClip


class BaseGridHueVideoConverter:
    BASE_16_RGB = np.array(
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

    def __init__(
        self,
        grid_size: int,
        output_scale: int = 1,
        line_thickness: int = 2,
        num_colors: int = 16,
    ):
        """
        Converts any MP4 into an NxN pixel grid mapped to 1..16,777,216 colors
        derived directly from the original 16-color clockwise Blue-start palette.
        """
        if not (1 <= int(num_colors) <= 16_777,216):
            raise ValueError("num_colors must be between 1 and 16777216.")

        self.grid_size = max(1, int(grid_size))
        self.output_scale = max(1, int(output_scale))
        self.line_thickness = max(0, int(line_thickness))
        self.num_colors = int(num_colors)

        self._setup_derivative_palette()
    def _interpolate_16_ring_rgb(self, count: int) -> np.ndarray:
        """
        Generates `count` colors directly along the closed 16-color RGB loop,
        preserving the exact 16 colors whenever `count` is a multiple of 16.
        """
        if count == 16:
            return self.BASE_16_RGB.copy()
        if count < 16:
            indices = np.linspace(0, 16, num=count, endpoint=False, dtype=int)
            return self.BASE_16_RGB[indices]

        closed_ring = np.vstack([self.BASE_16_RGB, self.BASE_16_RGB[0:1]]).astype(np.float32)
        positions = np.linspace(0.0, 16.0, num=count, endpoint=False)
        idx0 = np.floor(positions).astype(int)
        frac = (positions - idx0)[:, np.newaxis]
        interp_rgb = closed_ring[idx0] * (1.0 - frac) + closed_ring[idx0 + 1] * frac
        return np.clip(np.round(interp_rgb), 0, 255).astype(np.uint8)

    @staticmethod
    def _factor_hsv_levels(total_colors: int):
        """
        Factors `total_colors` (up to 16,777,216 = 256^3) into (H_levels, S_levels, V_levels)
        derived from the 16-hue base, assigning excess prime factors to Hue first, then Value, then Saturation.
        """
        n = total_colors
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)

        h_levels, s_levels, v_levels = 1, 1, 1
        for f in sorted(factors, reverse=True):
            if h_levels * f <= 256 and h_levels <= v_levels and h_levels <= s_levels:
                h_levels *= f
            elif v_levels * f <= 256 and v_levels <= s_levels:
                v_levels *= f
            elif s_levels * f <= 256:
                s_levels *= f
            elif h_levels * f <= 256:
                h_levels *= f
            elif v_levels * f <= 256:
                v_levels *= f
            else:
                s_levels *= f
        return h_levels, s_levels, v_levels

    def _setup_derivative_palette(self):
        """
        Configures either 1D pure-hue ring mapping (1..4096 colors) or
        3D Hue-Saturation-Value derivative quantization (4097..16,777,216 colors).
        """
        if self.num_colors == 16_777_216:
            self.use_3d_derivative = True
            self.h_levels, self.s_levels, self.v_levels = 256, 256, 256
            self.hue_ring_bgr = self._interpolate_16_ring_rgb(256)[:, ::-1]
            self.hue_ring_angles = self._bgr_to_hue_degrees(self.hue_ring_bgr)
            sort_order = np.argsort(self.hue_ring_angles)
            self.sorted_hues = self.hue_ring_angles[sort_order]
            self.sorted_ring_bgr = self.hue_ring_bgr[sort_order]
        elif self.num_colors <= 4096:
            self.use_3d_derivative = False
            ring_bgr = self._interpolate_16_ring_rgb(self.num_colors)[:, ::-1]
            ring_angles = self._bgr_to_hue_degrees(ring_bgr)
            sort_order = np.argsort(ring_angles)
            self.sorted_hues = ring_angles[sort_order]
            self.sorted_ring_bgr = ring_bgr[sort_order]
        else:
            self.use_3d_derivative = True
            self.h_levels, self.s_levels, self.v_levels = self._factor_hsv_levels(self.num_colors)
            hue_count = max(16, self.h_levels) if self.h_levels >= 16 else self.h_levels
            ring_bgr = self._interpolate_16_ring_rgb(hue_count)[:, ::-1]
            ring_angles = self._bgr_to_hue_degrees(ring_bgr)
            sort_order = np.argsort(ring_angles)
            self.sorted_hues = ring_angles[sort_order]
            self.sorted_ring_bgr = ring_bgr[sort_order]

    @staticmethod
    def _bgr_to_hue_degrees(bgr_pixels: np.ndarray) -> np.ndarray:
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
    def _nearest_ring_bgr(self, pixel_hues: np.ndarray) -> np.ndarray:
        """
        Fast O(log K) circular nearest-hue lookup against the sorted 16-color derivative ring.
        """
        n_ring = len(self.sorted_hues)
        if n_ring == 1:
            return np.repeat(self.sorted_ring_bgr, len(pixel_hues), axis=0)

        idx_right = np.searchsorted(self.sorted_hues, pixel_hues) % n_ring
        idx_left = (idx_right - 1) % n_ring

        dist_right = np.abs(pixel_hues - self.sorted_hues[idx_right])
        dist_right = np.minimum(dist_right, 360.0 - dist_right)

        dist_left = np.abs(pixel_hues - self.sorted_hues[idx_left])
        dist_left = np.minimum(dist_left, 360.0 - dist_left)

        chosen = np.where(dist_left <= dist_right, idx_left, idx_right)
        return self.sorted_ring_bgr[chosen]

    def _draw_grid_lines(self, upscaled_frame: np.ndarray) -> np.ndarray:
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
        small_grid = cv2.resize(
            frame_bgr,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_AREA
        )
        pixels_bgr = small_grid.reshape(-1, 3)

        if self.num_colors == 16_777_216:
            mapped_bgr = pixels_bgr
        elif not self.use_3d_derivative:
            pixel_hues = self._bgr_to_hue_degrees(pixels_bgr)
            mapped_bgr = self._nearest_ring_bgr(pixel_hues)
        else:
            # Map Hue to the 16-color derivative ring, and quantize Saturation & Value
            # into `s_levels` and `v_levels` so total palette size equals `num_colors`.
            pixel_hues = self._bgr_to_hue_degrees(pixels_bgr)
            base_hue_bgr = self._nearest_ring_bgr(pixel_hues).astype(np.float32) / 255.0

            bgr_norm = pixels_bgr.astype(np.float32) / 255.0
            cmax = np.max(bgr_norm, axis=1)
            cmin = np.min(bgr_norm, axis=1)
            sat = np.where(cmax > 1e-6, (cmax - cmin) / np.maximum(cmax, 1e-6), 0.0)
            val = cmax

            if self.s_levels > 1:
                sat_q = np.round(sat * (self.s_levels - 1)) / (self.s_levels - 1)
            else:
                sat_q = np.ones_like(sat)

            if self.v_levels > 1:
                val_q = np.round(val * (self.v_levels - 1)) / (self.v_levels - 1)
            else:
                val_q = np.ones_like(val)

            # Reconstruct BGR from pure 16-derivative hue + quantized saturation & brightness
            tinted = 1.0 - sat_q[:, np.newaxis] * (1.0 - base_hue_bgr)
            final_norm = tinted * val_q[:, np.newaxis]
            mapped_bgr = np.clip(np.round(final_norm * 255.0), 0, 255).astype(np.uint8)

        mapped_grid = mapped_bgr.reshape(self.grid_size, self.grid_size, 3)

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
        if not os.path.exists(input_mp4_path):
            raise FileNotFoundError(f"Input video not found: {input_mp4_path}")

        cap = cv2.VideoCapture(input_mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_mp4_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dim = self.grid_size * self.output_scale
        tag = f"[{self.grid_size}x{self.grid_size} | {self.num_colors} colors]"

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
    def __init__(self, output_scale: int = 256, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=2, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid4x4HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 128, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=4, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid8x8HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 64, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=8, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid16x16HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 32, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=16, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid32x32HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 16, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=32, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid64x64HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 8, line_thickness: int = 2, num_colors: int = 16):
        super().__init__(grid_size=64, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid128x128HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 4, line_thickness: int = 0, num_colors: int = 16):
        super().__init__(grid_size=128, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid256x256HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 2, line_thickness: int = 0, num_colors: int = 16):
        super().__init__(grid_size=256, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid512x512HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 1, line_thickness: int = 0, num_colors: int = 16):
        super().__init__(grid_size=512, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)


class Grid1024x1024HueVideoConverter(BaseGridHueVideoConverter):
    def __init__(self, output_scale: int = 1, line_thickness: int = 0, num_colors: int = 16):
        super().__init__(grid_size=1024, output_scale=output_scale, line_thickness=line_thickness, num_colors=num_colors)
if __name__ == "__main__":
    input_file = r"C:\Videos\input.mp4"

    # Choose any num_colors from 1 to 16777216 (default = 16)
    selected_colors = 8192

    Grid2x2HueVideoConverter(num_colors=selected_colors).convert(
        input_file, r"C:\Videos\output_2x2.mp4"
    )
    Grid4x4HueVideoConverter(num_colors=selected_colors).convert(
        input_file, r"C:\Videos\output_4x4.mp4"
    )
    #Grid16x16HueVideoConverter(num_colors=selected_colors).convert(
       # input_file, r"C:\Users\YourName\Videos\output_16x16.mp4"
    #)
   # Grid128x128HueVideoConverter(num_colors=selected_colors).convert(
    #    input_file, r"C:\Videos\output_128x128.mp4"
    #)
    #Grid256x256HueVideoConverter(num_colors=selected_colors).convert(
     #   input_file, r"C:\Videos\output_256x256.mp4"
    #)
    Grid512x512HueVideoConverter(num_colors=selected_colors).convert(
        input_file, r"C:\Videos\output_512x512.mp4"
    )
    #Grid1024x1024HueVideoConverter(num_colors=selected_colors).convert(
        #input_file, r"C:\Users\YourName\Videos\output_1024x1024.mp4"
    #)
