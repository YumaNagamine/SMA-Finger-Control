import math
import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from observation.vision.angle_processor import AngleProcessor


def _make_config(**adjustments):
    return {
        "video": {"denoising_mode": "off"},
        "markers": {
            "lab_ranges": [
                {"l": [10, 10], "a": [0, 0], "b": [0, 0]},
                {"l": [20, 20], "a": [0, 0], "b": [0, 0]},
                {"l": [30, 30], "a": [0, 0], "b": [0, 0]},
                {"l": [40, 40], "a": [0, 0], "b": [0, 0]},
            ],
            "area_thresholds": [1, 1, 1, 1],
            "adjustments": adjustments,
        },
        "output": {"csv_dir": "out/csv", "video_dir": "out/video"},
    }


class TestAngleProcessor(TestCase):
    def test_detect_markers_orders_and_pairs(self):
        config = _make_config()
        processor = AngleProcessor(config)
        lab_frame = np.zeros((20, 20, 3), dtype=np.uint8)

        distal_points = [(1, 1), (1, 3)]
        medial_points = [(5, 5), (5, 8)]
        proximal_points = [(10, 10), (10, 14)]
        palm_points = [(15, 15)]

        for x, y in distal_points:
            lab_frame[y, x] = (10, 0, 0)
        for x, y in medial_points:
            lab_frame[y, x] = (20, 0, 0)
        for x, y in proximal_points:
            lab_frame[y, x] = (30, 0, 0)
        for x, y in palm_points:
            lab_frame[y, x] = (40, 0, 0)

        markers = processor.detect_markers(lab_frame)

        self.assertCountEqual(markers["distal"], distal_points)
        self.assertCountEqual(markers["medial"], medial_points)
        self.assertCountEqual(markers["proximal"], proximal_points)
        self.assertEqual(markers["palm"][0], palm_points[0])
        self.assertEqual(len(markers["palm"]), 2)

    def test_modify_markers_rotates_and_shifts(self):
        config = _make_config(theta=math.pi / 2, distance=2)
        processor = AngleProcessor(config)
        markers = {
            "distal": [(0, 0), (0, 1)],
            "medial": [(0, 0), (1, 0)],
            "proximal": [(0, 0), (0, 2)],
            "palm": [(0, 0), (0, 1)],
        }

        modified = processor.modify_markers(markers)

        self.assertEqual(modified["distal"], [(0, 0), (0, 1)])
        self.assertEqual(modified["medial"], [(0, 0), (0, 1)])
        self.assertEqual(modified["proximal"], [(2, 0), (2, 2)])

    def test_calculate_angles_returns_expected_value(self):
        config = _make_config()
        processor = AngleProcessor(config)
        modified = {
            "distal": [(0, 1), (0, 0)],
            "medial": [(1, 0), (0, 0)],
            "proximal": [(1, 0), (0, 0)],
            "palm": [(1, 0), (0, 0)],
        }

        angle_0, angle_1, angle_2 = processor.calculate_angles(modified)

        self.assertAlmostEqual(angle_0, 90.0, delta=0.1)
        self.assertAlmostEqual(angle_1, 180.0, delta=0.1)
        self.assertAlmostEqual(angle_2, 180.0, delta=0.1)

    def test_estimate_joints_uses_offsets(self):
        config = _make_config(joint_shifters=[10, 20], mcp_offset=[-5, 5])
        processor = AngleProcessor(config)
        modified = {
            "medial": [(0, 0), (0, 10)],
            "proximal": [(0, 0), (0, 20)],
            "palm": [(100, 100), (110, 110)],
        }

        joints = processor.estimate_joints(modified)

        self.assertEqual(joints["DIP"], (0, -10))
        self.assertEqual(joints["PIP"], (0, -20))
        self.assertEqual(joints["MCP"], (95, 105))

    def test_ensure_pair_for_empty(self):
        config = _make_config()
        processor = AngleProcessor(config)
        self.assertEqual(processor._ensure_pair([]), [(-1, -1), (-1, -1)])
