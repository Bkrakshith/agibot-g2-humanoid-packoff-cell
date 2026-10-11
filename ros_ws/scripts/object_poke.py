#!/usr/bin/env python3
"""Teleport / push a scene rigid body through the object pose relay (engine started with OBJ_RELAY=1).

Publishes one std_msgs/String JSON on /g2cell/object_poke; the engine-side hook (g2cell_bringup/object_relay.py)
applies it on the physics thread between steps. Positions are in the body's parent frame (= world for
/World/Bags/*, /World/Cartons/*, /World/Totes/*).

Usage: python3 object_poke.py /World/Bags/Bag_01 [--xyz X Y Z | --dxyz DX DY DZ] [--vel VX VY VZ]
"""
import argparse
import json
import time

import rclpy
from std_msgs.msg import String


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--xyz", type=float, nargs=3)
    g.add_argument("--dxyz", type=float, nargs=3)
    ap.add_argument("--vel", type=float, nargs=3)
    a = ap.parse_args()
    req = {"path": a.path}
    for k in ("xyz", "dxyz", "vel"):
        if getattr(a, k) is not None:
            req[k] = getattr(a, k)
    rclpy.init()
    n = rclpy.create_node("g2cell_object_poke")
    pub = n.create_publisher(String, "/g2cell/object_poke", 10)
    t0 = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t0 < 5.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        raise SystemExit("no /g2cell/object_poke subscriber (sim not running with OBJ_RELAY=1?)")
    pub.publish(String(data=json.dumps(req)))
    time.sleep(0.3)
    print("sent", json.dumps(req))


if __name__ == "__main__":
    main()
