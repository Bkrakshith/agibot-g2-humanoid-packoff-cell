#!/usr/bin/env python3
"""Exercise 9 for the head cameras only: the solution's info() and grab() on kHeadColor/kHeadDepth.

Solution 09 also opens both wrist cameras, which the sim does not publish yet, so it is an
expected skip in run_training_solutions.sh. This script runs the same solution functions on the
cameras the sim has. Needs `source sim.sh`; reads only (no motion).

    python3 head_camera_check.py [TRAINING_DIR]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import agibot_gdk


def main() -> int:
    training = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["GDK_TRAINING_DIR"])
    spec = importlib.util.spec_from_file_location("camera_solution", training / "solutions/09_camera/camera.py")
    solution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solution
    spec.loader.exec_module(solution)

    C = agibot_gdk.CameraType
    cameras = {"head": C.kHeadColor, "head_depth": C.kHeadDepth}
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        sys.exit("gdk_init failed")
    camera = agibot_gdk.Camera(list(cameras.values()))
    time.sleep(3.0)
    failed = False
    try:
        for name, kind in cameras.items():
            print(name, asdict(solution.info(camera, kind)))
        for name, kind in cameras.items():
            image = None
            for _ in range(20):
                image = solution.grab(camera, kind)
                if image is not None:
                    break
                time.sleep(0.2)
            print(f"{name}: {'no frame' if image is None else f'{image.shape} {image.dtype}'}")
            failed |= image is None
    finally:
        camera.close_camera()
        agibot_gdk.gdk_release()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
