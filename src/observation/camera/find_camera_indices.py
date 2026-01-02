from __future__ import annotations

import argparse
import sys

import cv2


def _resolve_backend(backend_name: str | None) -> int | None:
    if not backend_name:
        return None
    return {
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_ANY": cv2.CAP_ANY,
    }.get(backend_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan camera indices that can be opened.")
    parser.add_argument("--max-index", type=int, default=10, help="Scan indices 0..N-1.")
    parser.add_argument("--backend", type=str, default=None, help="CAP_DSHOW or CAP_ANY.")
    parser.add_argument("--check-frame", action="store_true", help="Read one frame to verify.")
    return parser.parse_args()


def _open_camera(index: int, backend: int | None, check_frame: bool) -> tuple[bool, bool]:
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False, False
    if not check_frame:
        cap.release()
        return True, True
    ret, frame = cap.read()
    cap.release()
    return True, ret and frame is not None


def main() -> int:
    args = parse_args()
    backend = _resolve_backend(args.backend)

    found = []
    for idx in range(max(0, args.max_index)):
        opened, has_frame = _open_camera(idx, backend, args.check_frame)
        if opened and (has_frame or not args.check_frame):
            found.append(idx)

    if found:
        print("Available camera indices:", ", ".join(str(idx) for idx in found))
        return 0

    print("No available camera indices found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
