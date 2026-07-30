from __future__ import annotations

import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        print("WARNING: Python 3.11 is recommended.")

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except Exception as exc:
        print(f"OpenCV import failed: {exc}")
        return 1

    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except Exception as exc:
        print(f"NumPy import failed: {exc}")
        return 1

    print("Setup passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
