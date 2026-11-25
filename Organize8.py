import os
import sqlite3
from pathlib import Path
import cv2
import numpy as np


class Video2x2PatternIndexer:
    """
    Indexes every frame of all videos in a folder into one of 65,536 (16^4)
    2x2 color grid patterns using a 16-color RGB palette, storing lightweight
    pointers in SQLite instead of bloating disk space with raw frame images.
    """

    PALETTE_NAMES = [
        "Pure Blue",
        "Cerulean Blue",
        "Sky Blue",
        "Aquamarine / Cyan",
        "Mint Green",
        "Kelly Green",
        "Lime Green",
        "Chartreuse",
        "Pure Yellow",
        "Safety Orange",
        "Red-Orange",
        "Pure Red",
        "Fuchsia / Pink",
        "Magenta",
        "Deep Purple",
        "Indigo / Violet",
    ]

    PALETTE_RGB = np.array(
        [
            [0, 0, 255],      # 0: Pure Blue
            [0, 95, 255],     # 1: Cerulean Blue
            [0, 191, 255],    # 2: Sky Blue
            [0, 255, 223],    # 3: Aquamarine / Cyan
            [0, 255, 127],    # 4: Mint Green
            [0, 255, 31],     # 5: Kelly Green
            [63, 255, 0],     # 6: Lime Green
            [159, 255, 0],    # 7: Chartreuse
            [255, 255, 0],    # 8: Pure Yellow
            [255, 159, 0],    # 9: Safety Orange
            [255, 63, 0],     # 10: Red-Orange
            [255, 0, 31],     # 11: Pure Red
            [255, 0, 127],    # 12: Fuchsia / Pink
            [255, 0, 223],    # 13: Magenta
            [191, 0, 255],    # 14: Deep Purple
            [95, 0, 255],     # 15: Indigo / Violet
        ],
        dtype=np.float32,
    )

    def __init__(self, db_path: str = "video_2x2_patterns.db", use_hue_matching: bool = True):
        self.db_path = db_path
        self.use_hue_matching = use_hue_matching
        self.palette_hues = self._rgb_to_hue_degrees(self.PALETTE_RGB)
        self._init_db()

    @staticmethod
    def _rgb_to_hue_degrees(rgb_array: np.ndarray) -> np.ndarray:
        """Converts an (N, 3) float32 RGB array [0-255] into hue angles [0, 360)."""
        rgb_norm = rgb_array / 255.0
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        cmax = np.max(rgb_norm, axis=1)
        cmin = np.min(rgb_norm, axis=1)
        delta = cmax - cmin

        hue = np.zeros_like(cmax)
        mask = delta > 1e-6

        r_mask = mask & (cmax == r)
        g_mask = mask & (cmax == g)
        b_mask = mask & (cmax == b)

        hue[r_mask] = (60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0))
        hue[g_mask] = (60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0))
        hue[b_mask] = (60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0))

        return hue % 360.0

    def _init_db(self) -> None:
        """Creates the SQLite database and indexes for instant pattern lookups."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    pattern_code TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pattern_id ON frame_patterns(pattern_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_path ON frame_patterns(video_path);"
            )

    def _map_quadrants_to_indices(self, bgr_2x2: np.ndarray) -> np.ndarray:
        """
        Maps a 2x2 BGR image (4 pixels) to palette indices [0..15].
        Order: [Top-Left, Top-Right, Bottom-Left, Bottom-Right].
        """
        rgb_flat = bgr_2x2[:, :, ::-1].reshape(4, 3).astype(np.float32)

        if self.use_hue_matching:
            frame_hues = self._rgb_to_hue_degrees(rgb_flat)  # Shape: (4,)
            # Compute circular angular distance on the 360-degree color wheel
            diff = np.abs(frame_hues[:, None] - self.palette_hues[None, :])
            circular_dist = np.minimum(diff, 360.0 - diff)
            return np.argmin(circular_dist, axis=1)
        else:
            # Euclidean RGB distance
            diff = rgb_flat[:, None, :] - self.PALETTE_RGB[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            return np.argmin(dist_sq, axis=1)

    @staticmethod
    def encode_pattern_id(indices: np.ndarray) -> tuple[int, str]:
        """
        Encodes 4 palette indices [0..15] into a unique 16-bit integer (0..65535)
        and a 4-character hex string (e.g., '0A3F') for folder naming.
        """
        tl, tr, bl, br = int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])
        pattern_id = (tl << 12) | (tr << 8) | (bl << 4) | br
        pattern_code = f"{tl:X}{tr:X}{bl:X}{br:X}"
        return pattern_id, pattern_code

    def _save_pattern_reference_image(self, pattern_code: str, out_path: Path, size: int = 256) -> None:
        """
        Generates and saves a visual 2x2 grid image representing the 4 palette colors
        for a given 4-character hex pattern code with 2-pixel black grid lines.
        """
        tl_idx = int(pattern_code[0], 16)
        tr_idx = int(pattern_code[1], 16)
        bl_idx = int(pattern_code[2], 16)
        br_idx = int(pattern_code[3], 16)

        grid_rgb = np.array(
            [
                [self.PALETTE_RGB[tl_idx], self.PALETTE_RGB[tr_idx]],
                [self.PALETTE_RGB[bl_idx], self.PALETTE_RGB[br_idx]],
            ],
            dtype=np.uint8,
        )
        grid_bgr = grid_rgb[:, :, ::-1]
        ref_img = cv2.resize(grid_bgr, (size, size), interpolation=cv2.INTER_NEAREST)

        # Draw 2-pixel black outer border and center divider lines
        half = size // 2
        ref_img[:2, :] = 0
        ref_img[-2:, :] = 0
        ref_img[:, :2] = 0
        ref_img[:, -2:] = 0
        ref_img[half - 1 : half + 1, :] = 0
        ref_img[:, half - 1 : half + 1] = 0

        cv2.imwrite(str(out_path), ref_img)

    def index_video(
        self,
        video_path: str,
        max_duration_sec: float = 128.0,
        deduplicate_consecutive: bool = False,
    ) -> int:
        """
        Processes a single video up to max_duration_sec (default 2m08s = 128s)
        and inserts frame pattern records into SQLite.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        max_frames = int(fps * max_duration_sec)

        records = []
        frame_idx = 0
        last_pattern_id = -1

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Area interpolation averages all pixels inside each of the 4 quadrants
            small_2x2 = cv2.resize(frame, (2, 2), interpolation=cv2.INTER_AREA)
            indices = self._map_quadrants_to_indices(small_2x2)
            pattern_id, pattern_code = self.encode_pattern_id(indices)

            if not deduplicate_consecutive or pattern_id != last_pattern_id:
                timestamp_ms = (frame_idx / fps) * 1000.0
                records.append(
                    (pattern_id, pattern_code, str(video_path), frame_idx, timestamp_ms)
                )
                last_pattern_id = pattern_id

            frame_idx += 1

        cap.release()

        if records:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO frame_patterns
                    (pattern_id, pattern_code, video_path, frame_idx, timestamp_ms)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    records,
                )

        return len(records)

    def index_folder(
        self,
        folder_path: str,
        extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv"),
        deduplicate_consecutive: bool = False,
    ) -> None:
        """Scans a Windows folder for videos and indexes all frames."""
        folder = Path(folder_path)
        video_files = [
            f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions
        ]

        print(f"Found {len(video_files)} video(s) in '{folder_path}'.")
        total_indexed = 0

        for idx, video_file in enumerate(video_files, start=1):
            count = self.index_video(
                str(video_file),
                max_duration_sec=128.0,
                deduplicate_consecutive=deduplicate_consecutive,
            )
            total_indexed += count
            print(f"[{idx}/{len(video_files)}] {video_file.name}: {count} frames indexed.")

        print(f"Done. Total frame records stored: {total_indexed}")

    def get_top_patterns(self, limit: int | None = None) -> list[tuple]:
        """Returns the most frequent 2x2 patterns across all indexed videos."""
        with sqlite3.connect(self.db_path) as conn:
            if limit is not None:
                cursor = conn.execute(
                    """
                    SELECT pattern_id, pattern_code, COUNT(*) as frame_count,
                           COUNT(DISTINCT video_path) as video_count
                    FROM frame_patterns
                    GROUP BY pattern_id
                    ORDER BY frame_count DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT pattern_id, pattern_code, COUNT(*) as frame_count,
                           COUNT(DISTINCT video_path) as video_count
                    FROM frame_patterns
                    GROUP BY pattern_id
                    ORDER BY frame_count DESC;
                    """
                )
            return cursor.fetchall()

    def export_frames_for_pattern(
        self,
        pattern_code: str,
        output_base_dir: str = "exported_patterns",
        max_export: int | None = None,
    ) -> int:
        """
        Extracts and saves actual PNG images ONLY for a specific 4-char hex pattern
        (e.g., 'A99A') into a dedicated subfolder on demand, along with a visual
        reference image of the 2x2 color grid.
        """
        pattern_code = pattern_code.upper()
        target_dir = Path(output_base_dir) / f"pattern_{pattern_code}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save the visual 2x2 pattern reference image inside the folder
        ref_image_path = target_dir / f"_pattern_{pattern_code}_grid.png"
        self._save_pattern_reference_image(pattern_code, ref_image_path)

        with sqlite3.connect(self.db_path) as conn:
            if max_export is not None:
                cursor = conn.execute(
                    """
                    SELECT video_path, frame_idx
                    FROM frame_patterns
                    WHERE pattern_code = ?
                    ORDER BY video_path, frame_idx
                    LIMIT ?;
                    """,
                    (pattern_code, max_export),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT video_path, frame_idx
                    FROM frame_patterns
                    WHERE pattern_code = ?
                    ORDER BY video_path, frame_idx;
                    """,
                    (pattern_code,),
                )
            rows = cursor.fetchall()

        if not rows:
            print(f"No frames found for pattern {pattern_code}.")
            return 0

        # Group frame numbers by video so each video is opened only once
        frames_by_video: dict[str, list[int]] = {}
        for video_path, frame_idx in rows:
            frames_by_video.setdefault(video_path, []).append(frame_idx)

        saved_count = 0
        for video_path, frame_indices in frames_by_video.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            video_stem = Path(video_path).stem
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    out_file = target_dir / f"{video_stem}_frame_{f_idx:06d}.png"
                    cv2.imwrite(str(out_file), frame)
                    saved_count += 1

            cap.release()

        print(f"Exported {saved_count} frame(s) and pattern grid image to '{target_dir}'.")
        return saved_count


if __name__ == "__main__":
    # Example Windows 11 Usage
    VIDEO_FOLDER = r"C:\Images"
    DB_FILE = r"C:\Images\frame_patterns.db"

    indexer = Video2x2PatternIndexer(db_path=DB_FILE, use_hue_matching=True)

    # 1. Index all videos (set deduplicate_consecutive=True to skip identical back-to-back patterns)
    indexer.index_folder(VIDEO_FOLDER, deduplicate_consecutive=False)

    # 2. View all detected 2x2 patterns across your videos
    detected_patterns = indexer.get_top_patterns(limit=None)
    print(f"\nDetected {len(detected_patterns)} Unique 2x2 Pattern(s):")
    for pid, pcode, f_count, v_count in detected_patterns:
        print(f"Pattern {pcode} (ID {pid:5d}): {f_count} frames across {v_count} video(s)")

    # 3. Export physical images and pattern reference grid for EVERY detected pattern
    for pid, pcode, f_count, v_count in detected_patterns:
        indexer.export_frames_for_pattern(pcode, output_base_dir="exported_patterns", max_export=None)
