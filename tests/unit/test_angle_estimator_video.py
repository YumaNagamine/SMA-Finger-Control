import sys
from pathlib import Path
from unittest import TestCase
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from observation.vision.angle_estimator_video import ensure_dirs, store_csv


class TestAngleEstimatorVideo(TestCase):
    def test_ensure_dirs_creates_output_paths(self):
        with TemporaryDirectory() as tmpdir:
            csv_dir = Path(tmpdir) / "csv"
            video_dir = Path(tmpdir) / "video"
            output_cfg = {"csv_dir": str(csv_dir), "video_dir": str(video_dir)}

            created_csv, created_video = ensure_dirs(output_cfg)

            self.assertTrue(created_csv.exists())
            self.assertTrue(created_video.exists())

    def test_store_csv_writes_time_column(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "angles.csv"
            measurements = [
                [0, 10.0, 20.0, 30.0],
                [1, 11.0, 21.0, 31.0],
            ]

            store_csv(measurements, fps=2.0, output_path=csv_path)

            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertListEqual(list(df.columns), ["frame", "angle_0", "angle_1", "angle_2", "time"])
            self.assertAlmostEqual(df.loc[1, "time"], 0.5, delta=1e-6)
