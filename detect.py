import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip


class OcularGazeVideoAnalyzer:
    """
    Analyzes an MP4 video on Windows 11 to estimate human visual saliency (ocular heatmap),
    matches gaze hotspots to YOLO11 detected objects (ignoring shadows), tracks dwell time
    and spatial area statistics per frame, and exports a new MP4 with synchronized audio.
    """

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.45):
        # Load lightweight YOLO11 nano model (automatically downloads on first run)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Initialize OpenCV Spectral Residual Saliency detector
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        # State trackers across frames
        self.object_dwell_frames = {}
        self.first_fixation_label = None
        self.first_fixation_recorded = False

    def _compute_shadow_free_saliency(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes a normalized 0-255 saliency map while suppressing dark shadow regions.
        """
        # Convert to HSV and suppress low-brightness (shadow) pixels in the Value channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Create a shadow mask: pixels with very low luminance (V < 45) are treated as shadows
        shadow_mask = (v < 45).astype(np.uint8)

        # Lift shadow luminance to the median brightness of non-shadow areas so saliency ignores shadow edges
        non_shadow_pixels = v[shadow_mask == 0]
        median_v = int(np.median(non_shadow_pixels)) if non_shadow_pixels.size > 0 else 128
        v_deshadowed = np.where(shadow_mask == 1, median_v, v).astype(np.uint8)

        deshadowed_bgr = cv2.cvtColor(cv2.merge([h, s, v_deshadowed]), cv2.COLOR_HSV2BGR)

        # Compute saliency map on the shadow-suppressed frame
        success, saliency_map = self.saliency.computeSaliency(deshadowed_bgr)
        if not success:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Normalize to 0-255 and apply Gaussian blur to simulate smooth human foveal vision
        saliency_uint8 = (saliency_map * 255).astype("uint8")
        saliency_smooth = cv2.GaussianBlur(saliency_uint8, (31, 31), 0)
        saliency_norm = cv2.normalize(saliency_smooth, None, 0, 255, cv2.NORM_MINMAX)

        # Explicitly zero out saliency inside detected shadow masks
        saliency_norm[shadow_mask == 1] = 0
        return saliency_norm

    def _analyze_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
        """
        Processes a single frame: generates heatmap, runs YOLO11 detection, calculates
        fixation metrics, and draws overlays.
        """
        height, width, _ = frame.shape
        total_frame_area = float(width * height)

        # 1. Compute shadow-ignored ocular saliency map
        saliency_norm = self._compute_shadow_free_saliency(frame)

        # Locate peak gaze point (where human eye looks first/most intensely in this frame)
        _, max_val, _, max_loc = cv2.minMaxLoc(saliency_norm)
        peak_x, peak_y = max_loc

        # Create colorized heatmap overlay (JET colormap)
        heatmap_color = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_JET)

        # Only blend strong saliency hotspots so the underlying video stays clear
        alpha_mask = (saliency_norm / 255.0) * 0.55
        alpha_3ch = np.dstack([alpha_mask] * 3)
        annotated_frame = (frame * (1.0 - alpha_3ch) + heatmap_color * alpha_3ch).astype(np.uint8)

        # 2. Run YOLO11 object detection (CPU-friendly single frame inference)
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            agnostic_nms=True,
            verbose=False
        )

        active_gaze_target = "Background / Unclassified Region"
        active_target_area_pct = 0.0
        active_target_dwell_sec = 0.0
        detected_objects_summary = []

        # 3. Correlate ocular hotspots with detected objects
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]
                conf = float(box.conf[0].item())

                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                box_area = box_w * box_h
                area_pct = (box_area / total_frame_area) * 100.0

                # Extract saliency intensity inside this bounding box
                roi_saliency = saliency_norm[y1:y2, x1:x2]
                mean_saliency = float(np.mean(roi_saliency)) if roi_saliency.size > 0 else 0.0

                # Check if the peak gaze coordinate falls inside this object or if mean saliency is high
                is_gazed_at = (x1 <= peak_x <= x2 and y1 <= peak_y <= y2) or (mean_saliency > 140)

                if is_gazed_at:
                    self.object_dwell_frames[label] = self.object_dwell_frames.get(label, 0) + 1
                    dwell_sec = self.object_dwell_frames[label] / fps

                    if not self.first_fixation_recorded:
                        self.first_fixation_label = label
                        self.first_fixation_recorded = True

                    # Update primary gaze target if peak falls here
                    if x1 <= peak_x <= x2 and y1 <= peak_y <= y2:
                        active_gaze_target = label
                        active_target_area_pct = area_pct
                        active_target_dwell_sec = dwell_sec

                    box_color = (0, 255, 255)  # Yellow highlight for active gaze object
                else:
                    dwell_sec = self.object_dwell_frames.get(label, 0) / fps
                    box_color = (0, 220, 0)    # Green for peripheral detected objects

                detected_objects_summary.append((label, area_pct, dwell_sec, mean_saliency))

                # Draw bounding box & per-object stats
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                tag = f"{label} ({conf:.2f}) | Area:{area_pct:.1f}% | Gaze:{dwell_sec:.2f}s"
                cv2.putText(
                    annotated_frame,
                    tag,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )
        # Fallback dwell tracking if gaze is on a salient non-YOLO region
        if active_gaze_target == "Background / Unclassified Region":
            # Measure contiguous hotspot area above saliency threshold 160
            _, thresh_map = cv2.threshold(saliency_norm, 160, 255, cv2.THRESH_BINARY)
            hotspot_pixels = cv2.countNonZero(thresh_map)
            active_target_area_pct = (hotspot_pixels / total_frame_area) * 100.0
            self.object_dwell_frames["Unclassified Hotspot"] = (
                self.object_dwell_frames.get("Unclassified Hotspot", 0) + 1
            )
            active_target_dwell_sec = self.object_dwell_frames["Unclassified Hotspot"] / fps

        # Draw foveal gaze reticle at peak saliency point
        cv2.circle(annotated_frame, (peak_x, peak_y), 14, (0, 0, 255), 2)
        cv2.circle(annotated_frame, (peak_x, peak_y), 3, (255, 255, 255), -1)

        # 4. Render semi-transparent HUD panel on every frame with full statistics
        self._draw_hud(
            annotated_frame,
            frame_idx=frame_idx,
            fps=fps,
            peak_coords=(peak_x, peak_y),
            peak_intensity=max_val,
            first_looked=self.first_fixation_label or "Scanning...",
            current_target=active_gaze_target,
            target_area_pct=active_target_area_pct,
            target_dwell_sec=active_target_dwell_sec,
            total_objects=len(detected_objects_summary),
        )

        return annotated_frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        peak_coords: tuple,
        peak_intensity: float,
        first_looked: str,
        current_target: str,
        target_area_pct: float,
        target_dwell_sec: float,
        total_objects: int,
    ) -> None:
        """
        Draws a dark, high-contrast statistics banner at the top-left of every frame.
        """
        overlay = frame.copy()
        panel_w = min(560, frame.shape[1] - 20)
        panel_h = 155
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        timestamp_sec = frame_idx / fps
        lines = [
            f"Time: {timestamp_sec:06.2f}s (Frame {frame_idx}) | Objects Detected: {total_objects}",
            f"First Looked Object: {first_looked}",
            f"Current Gaze Focus: {current_target}",
            f"Focus Section Size: {target_area_pct:.2f}% of frame | Dwell Time: {target_dwell_sec:.2f}s",
            f"Peak Ocular Focal Point: X={peak_coords[0]}, Y={peak_coords[1]} (Score: {int(peak_intensity)}/255)",
        ]

        y_offset = 34
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y_offset += 26

    def process_video(self, input_mp4: str, output_mp4: str) -> None:
        """
        Reads input_mp4, writes annotated frames to a temporary video, and muxes
        the original audio track into output_mp4 in perfect sync.
        """
        if not os.path.exists(input_mp4):
            raise FileNotFoundError(f"Input video not found: {input_mp4}")

        # Reset tracking state for new video
        self.object_dwell_frames.clear()
        self.first_fixation_label = None
        self.first_fixation_recorded = False

        cap = cv2.VideoCapture(input_mp4)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video stream: {input_mp4}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_silent_video = f"temp_silent_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_silent_video, fourcc, fps, (width, height))

        print(f"Processing '{input_mp4}' ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames)...")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._analyze_frame(frame, frame_idx, fps)
                writer.write(processed_frame)

                frame_idx += 1
                if frame_idx % 60 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames...")
        finally:
            cap.release()
            writer.release()

        print("Synchronizing original audio track...")
        original_clip = VideoFileClip(input_mp4)
        processed_clip = VideoFileClip(temp_silent_video)

        try:
            if original_clip.audio is not None:
                final_clip = processed_clip.with_audio(original_clip.audio)
            else:
                final_clip = processed_clip

            final_clip.write_videofile(
                output_mp4,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                logger=None,
            )
        finally:
            original_clip.close()
            processed_clip.close()
            if os.path.exists(temp_silent_video):
                os.remove(temp_silent_video)

        print(f"Complete! Saved synchronized output to: {output_mp4}")


# Example Windows 11 Execution
if __name__ == "__main__":
    analyzer = OcularGazeVideoAnalyzer(model_path="yolo11n.pt", conf_threshold=0.45)
    analyzer.process_video(
        input_mp4=r"C:\Videos\sample_input.mp4",
        output_mp4=r"C:\Videos\sample_ocular_output.mp4",
    )
