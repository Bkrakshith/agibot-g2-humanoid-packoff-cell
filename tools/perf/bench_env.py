import math, time, sys
import numpy as np
from g2cell_demo.env_check import EnvChecker, path_check, head_clearance
URDF = "/geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf"; COLL = "/geniesim_assets/scenes/g2cell/cell_colliders.json"
t0=time.perf_counter(); chk = EnvChecker(URDF, COLL); print("load", time.perf_counter()-t0, "boxes", len(chk.c), {k: len(v) for k,v in chk.points.items() if k.startswith(("arm_l","gripper_l"))})
j = {"idx03_body_joint3": math.radians(20)}
for side, b in (("l", 20), ("r", 60)):
    for k, deg in ((2, -90), (4, -110), (6, -30)):
        j[f"idx{b+k}_arm_{side}_joint{k}"] = math.radians(deg)
N=50
t0=time.perf_counter()
for i in range(N): chk.link_transforms(j)
t1=time.perf_counter()
for i in range(N): chk.clearance(j, "left", base=(0,0,0))
t2=time.perf_counter()
for i in range(N): head_clearance(chk, j, "left")
t3=time.perf_counter()
print("link_transforms ms", (t1-t0)/N*1e3, "clearance ms", (t2-t1)/N*1e3, "head ms", (t3-t2)/N*1e3, "near", len(chk._near((0,0,0))))
path=[dict(j), {**j, "idx05_body_joint5": 1.0}, dict(j)]
t0=time.perf_counter(); r=path_check(chk, path, "left", samples=8, fixed=j); print("path_check 2 seg", (time.perf_counter()-t0)*1e3, "ms", r)
