from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick Lab color values from an image.")
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument("--points", type=int, default=4, help="Number of points to sample.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("config_color.json")),
        help="Where to store sampled Lab values as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    sampled = []
    window_name = "Pick colors (left-click)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            lab_value = lab_image[y, x].tolist()  # [L, a, b]
            sampled.append({"position": [int(x), int(y)], "lab": lab_value})
            print(f"Sampled #{len(sampled)} at ({x}, {y}): {lab_value}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if len(sampled) >= args.points:
                break
    finally:
        cv2.destroyAllWindows()

    output_path = Path(args.output)
    output_data = {
        "image_path": str(image_path),
        "lab_samples": sampled,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(sampled)} samples to {output_path}")


if __name__ == "__main__":
    # Allow running this file directly without modifying PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    main()
