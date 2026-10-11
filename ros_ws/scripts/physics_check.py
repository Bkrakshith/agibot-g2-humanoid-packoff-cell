"""Standalone headless PhysX settle test of cell.usda (no robot). Run in the g2cell container:
   python3 /g2cell_ws/scripts/physics_check.py [--usd /geniesim_assets/scenes/g2cell/cell.usda] [--seconds 60]
Prints start/end world positions of every rigid body and the max drift."""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--usd", default="/geniesim_assets/scenes/g2cell/cell.usda")
ap.add_argument("--seconds", type=float, default=60.0); ap.add_argument("--lift", type=float, default=0.0); ap.add_argument("--out", default="/g2cell_output/physics_check.json")
a = ap.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from isaacsim.core.utils.stage import open_stage
from pxr import Usd, UsdGeom, UsdPhysics
import omni.timeline, omni.physx
open_stage(a.usd)
for _ in range(10): app.update()
stage = omni.usd.get_context().get_stage()
bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.GetAttribute("physics:kinematicEnabled").Get()]
def pose(p):
    m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); t = m.ExtractTranslation()
    zaxis = m.TransformDir((0,0,1)); return [round(t[0],4), round(t[1],4), round(t[2],4)], round(zaxis[2],4)
if a.lift:  # drop test: raise every dynamic body before simulating
    from pxr import Gf
    for p in bodies:
        op = [o for o in UsdGeom.Xformable(p).GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
        v = op.Get(); op.Set(type(v)(v[0], v[1], v[2] + a.lift))
start = {p.GetPath().pathString: pose(p) for p in bodies}
px = omni.physx.get_physx_interface()
px.start_simulation()
dt = 1/60.0; n = int(a.seconds/dt); samples = {}
for i in range(n):
    px.update_simulation(dt, i*dt); px.update_transformations(False, True, True, False)
    if i % 600 == 0 or i == n-1:
        samples[round(i*dt,1)] = {k: pose(stage.GetPrimAtPath(k))[0] for k in start}
end = {k: pose(stage.GetPrimAtPath(k)) for k in start}
res = {}
for k in start:
    d = [end[k][0][j]-start[k][0][j] for j in range(3)]
    res[k] = {"start": start[k][0], "end": end[k][0], "drift_mm": [round(1000*x,1) for x in d], "upright_z": end[k][1]}
    print(f"{k:45s} start={start[k][0]} end={end[k][0]} drift_mm={res[k]['drift_mm']} up={end[k][1]}")
json.dump({"seconds": a.seconds, "bodies": res, "samples": samples}, open(a.out, "w"), indent=1)
app.close()
