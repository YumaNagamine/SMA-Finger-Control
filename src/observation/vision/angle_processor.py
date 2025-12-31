from __future__ import annotations

import cv2
import numpy as np


class AngleProcessor:
    """Core processing logic to detect markers, adjust them, and compute angles."""

    def __init__(self, config: dict):
        markers_cfg = config["markers"]
        video_cfg = config.get("video", {})
        adjustments = markers_cfg.get("adjustments", {})

        self.lab_ranges = markers_cfg["lab_ranges"]
        self.area_thresholds = markers_cfg.get("area_thresholds", [10] * len(self.lab_ranges))
        self.line_padding = markers_cfg.get("line_padding", 5)
        self.denoising_mode = video_cfg.get("denoising_mode", "monocolor")

        self.theta = adjustments.get("theta", 0.0)
        self.distance_shift = adjustments.get("distance", 0.0)
        self.joint_shifters = adjustments.get("joint_shifters", [15, 110])
        self.mcp_offset = tuple(adjustments.get("mcp_offset", [-130, 0]))

        self.colors = markers_cfg.get(
            "draw_colors",
            [(255, 0, 0), (127, 0, 255), (0, 127, 0), (0, 127, 255)],
        )

    @staticmethod
    def _order_by_distance(points: list, reference: tuple) -> list:
        ref = np.array(reference)
        return sorted(points, key=lambda p: np.linalg.norm(np.array(p) - ref), reverse=True)

    @staticmethod
    def _find_components(mask: np.ndarray, area_threshold: int) -> list:
        mask_uint8 = np.uint8(mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        valid_indices = [i for i, stat in enumerate(stats[1:], start=1) if stat[4] >= area_threshold]
        valid_indices.sort(key=lambda idx: stats[idx][4], reverse=True)
        points = []
        for idx in valid_indices[:2]:
            centroid = centroids[idx]
            points.append((int(centroid[0]), int(centroid[1])))
        return points

    def _segment_masks(self, lab_frame: np.ndarray) -> list:
        masks = []
        for rng in self.lab_ranges:
            lower = np.array([rng["l"][0], rng["a"][0], rng["b"][0]], dtype=np.uint8)
            upper = np.array([rng["l"][1], rng["a"][1], rng["b"][1]], dtype=np.uint8)
            mask = cv2.inRange(lab_frame, lower, upper)
            if self.denoising_mode == "monocolor":
                mask = cv2.fastNlMeansDenoising(np.uint8(mask), None, 5, 3, 5)
            masks.append(mask > 0)
        return masks

    @staticmethod
    def _calculate_vector(point1: tuple, point2: tuple) -> np.ndarray:
        return np.array(point2) - np.array(point1)

    @staticmethod
    def _rotate_vector(vector: np.ndarray, theta: float) -> np.ndarray:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return rotation_matrix @ vector

    @staticmethod
    def _shift_markers(markers: list, distance: float) -> list:
        markers_arr = np.array(markers)
        vector = markers_arr[0] - markers_arr[1]
        rotate_matrix = np.array([[0, -1], [1, 0]])
        vertical_vector = rotate_matrix @ vector
        norm = np.linalg.norm(vertical_vector)
        if norm == 0:
            return markers
        shifter = vertical_vector * (distance / norm)
        modified_markers = [markers_arr[0] + shifter, markers_arr[1] + shifter]
        return [(int(x), int(y)) for x, y in modified_markers]

    @staticmethod
    def _calculate_angle(distal_markers: list, proximal_markers: list) -> float:
        try:
            distalis = np.array(distal_markers)
            proximal = np.array(proximal_markers)
            distalis_vec = distalis[0] - distalis[1]
            proximal_vec = proximal[0] - proximal[1]

            dot_product = np.dot(proximal_vec, distalis_vec)
            cross_product = np.cross(proximal_vec, distalis_vec)
            norm_distalis = np.linalg.norm(distalis_vec)
            norm_proximal = np.linalg.norm(proximal_vec)
            if norm_distalis == 0 or norm_proximal == 0:
                return float("nan")
            angle_rad = np.arctan2(cross_product, dot_product)
            angle_degree = np.degrees(angle_rad)
            if angle_rad < 0:
                joint_angle = abs(angle_degree) + 180
            else:
                joint_angle = 180 - angle_degree
            return joint_angle
        except Exception:
            return float("nan")

    def _ensure_pair(self, points: list) -> list:
        if len(points) == 1:
            return [points[0], (points[0][0] + 100, points[0][1])]
        if len(points) == 0:
            return [(-1, -1), (-1, -1)]
        if len(points) == 2:
            return points
        return points[:2]

    def detect_markers(self, lab_frame: np.ndarray) -> dict:
        masks = self._segment_masks(lab_frame)

        palm_points = self._find_components(masks[3], self.area_thresholds[3])
        palm = palm_points[0] if palm_points else None

        proximal_points = self._find_components(masks[2], self.area_thresholds[2])
        if palm and len(proximal_points) == 2:
            proximal_points = self._order_by_distance(proximal_points, palm)

        medial_points = self._find_components(masks[1], self.area_thresholds[1])
        if palm and len(medial_points) == 2:
            medial_points = self._order_by_distance(medial_points, palm)

        medial_distal = medial_points[0] if medial_points else palm
        distal_points = self._find_components(masks[0], self.area_thresholds[0])
        if medial_distal and len(distal_points) == 2:
            distal_points = self._order_by_distance(distal_points, medial_distal)

        return {
            "distal": self._ensure_pair(distal_points),
            "medial": self._ensure_pair(medial_points),
            "proximal": self._ensure_pair(proximal_points),
            "palm": self._ensure_pair(palm_points),
        }

    def modify_markers(self, markers: dict) -> dict:
        modified = {}
        # 0: distal markers (as-is)
        modified["distal"] = [tuple(markers["distal"][0]), tuple(markers["distal"][1])]

        # 1: medial markers - rotate vector by theta
        vec_medial = self._calculate_vector(markers["medial"][0], markers["medial"][1])
        rotated_vec = self._rotate_vector(vec_medial, self.theta)
        medial_distal = np.array(markers["medial"][0])
        medial_proximal = medial_distal + rotated_vec
        modified["medial"] = [
            tuple(medial_distal.astype(int)),
            tuple(medial_proximal.astype(int)),
        ]

        # 2: proximal markers - shift perpendicular
        modified["proximal"] = self._shift_markers(markers["proximal"], self.distance_shift)

        # 3: palm marker - synthetic line
        palm_pair = self._ensure_pair(markers["palm"])
        modified["palm"] = palm_pair

        return modified

    def estimate_joints(self, modified: dict) -> dict:
        direction_0 = np.array(modified["medial"][0]) - np.array(modified["medial"][1])
        direction_1 = np.array(modified["proximal"][0]) - np.array(modified["proximal"][1])

        dip = np.array(modified["medial"][0])
        pip = np.array(modified["proximal"][0])

        if np.linalg.norm(direction_0) > 0:
            dip = dip + (self.joint_shifters[0] / np.linalg.norm(direction_0)) * direction_0
        if np.linalg.norm(direction_1) > 0:
            pip = pip + (self.joint_shifters[1] / np.linalg.norm(direction_1)) * direction_1

        mcp = (
            modified["palm"][0][0] + int(self.mcp_offset[0]),
            modified["palm"][0][1] + int(self.mcp_offset[1]),
        )

        return {
            "DIP": tuple(dip.astype(int)),
            "PIP": tuple(pip.astype(int)),
            "MCP": mcp,
        }

    def calculate_angles(self, modified: dict) -> tuple:
        angle_0 = self._calculate_angle(modified["distal"], modified["medial"])
        angle_1 = self._calculate_angle(modified["medial"], modified["proximal"])
        angle_2 = self._calculate_angle(modified["proximal"], modified["palm"])
        return angle_0, angle_1, angle_2

    def draw_overlays(self, frame: np.ndarray, raw_markers: dict, modified_markers: dict, joints: dict) -> np.ndarray:
        out = frame.copy()
        ordered_keys = ["distal", "medial", "proximal", "palm"]
        for idx, key in enumerate(ordered_keys):
            color = self.colors[idx % len(self.colors)]
            for i, point in enumerate(modified_markers[key]):
                cv2.circle(out, (int(point[0]), int(point[1])), radius=i * 10 + 5, color=color, thickness=2)
            p1, p2 = modified_markers[key]
            direction = self._calculate_vector(p2, p1)
            line_p1 = (int(p2[0] - self.line_padding * direction[0]), int(p2[1] - self.line_padding * direction[1]))
            line_p2 = (int(p1[0] + self.line_padding * direction[0]), int(p1[1] + self.line_padding * direction[1]))
            cv2.line(out, line_p1, line_p2, color, 3)

            # Raw markers shown in white for reference
            for i, point in enumerate(raw_markers[key]):
                cv2.circle(out, (int(point[0]), int(point[1])), radius=i * 10 + 5, color=(255, 255, 255), thickness=1)

        for joint in ("DIP", "PIP", "MCP"):
            cv2.circle(out, joints[joint], radius=6, color=(0, 255, 0), thickness=-1)

        return out
