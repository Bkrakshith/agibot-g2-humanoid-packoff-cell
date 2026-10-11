# Asset conventions (Stage 2)

Every cell object is its own USD asset, composed into the scene by reference
(spec: "Keep each object ... as its own USD asset with physics on a separate payload layer").

## Units and frames

- `metersPerUnit = 1.0`, `upAxis = "Z"`, kilograms. Set both on every layer.
- Asset origin = **centre of its footprint on the floor (Z = 0)**. The scene places it
  with one translate equal to the spec's "Centre X, Y" (cell frame: origin at the G2 base,
  +X robot → conveyor, +Y robot's left, Z up).
- Long axis conventions: the conveyor runs along **Y**; crates sit with their **long side
  along Y** (on the conveyor they travel long-side first along Y).

## Files per asset

```
assets/<name>/
  <name>.usda    root layer: defaultPrim /<Name> (Xform, kind=component);
                 references ./geo.usda, payloads ./physics.usda
  geo.usda       geometry + UsdPreviewSurface placeholder materials (final MDL is Stage 6)
  physics.usda   `over` prims only: UsdPhysics CollisionAPI (+ MeshCollisionAPI approximation
                 "convexDecomposition"/"convexHull"/boxes), RigidBodyAPI + MassAPI on movable
                 parts, kinematic or static for fixed parts, physics materials (friction)
  README.md      one table: spec value, modelled value, source
```

- Build every asset from a **generator script** `assets/<name>/build.py` (pxr only; run with
  `/home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python`). The .usda files are its output and
  are regenerated, never hand-edited.
- Prefer simple primitives/meshes with sensible triangle counts; no external textures yet.
- Physics materials: stainless 0.3 static/dynamic friction, HDPE 0.5, rubber/pads 1.0.
- Name prims in PascalCase; parts that later stages drive (stop gate, photo-eye, roller
  surface) get stable, documented prim paths.

## Tests

`assets/tests/test_<name>.py` (pytest, same venv) checks, with the real USD API:
- the root layer opens, has defaultPrim, metersPerUnit 1, upAxis Z;
- world-space bounding box of the visual geometry matches the spec dimensions within ±20 mm
  (the spec's acceptance tolerance);
- collision APIs exist on the parts that need them;
- documented prim paths exist.

Write the test first, watch it fail, then write `build.py`.
