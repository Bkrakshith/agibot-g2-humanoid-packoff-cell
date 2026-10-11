"""Generate a batch of episodes: seeded randomisation -> reset -> record -> PickBag (or the operator) ->
stop + mark -> episode.json. Rerun the same command to resume (done seeds are skipped).

    ros2 run g2cell_datagen episode_runner --batch pick_v1 --seeds 0-49                  # scripted
    ros2 run g2cell_datagen episode_runner --batch pick_v1 --seeds 0-49 --reset restart  # bags really moved
    ros2 run g2cell_datagen episode_runner --batch teleop_v1 --seeds 0-9 --mode operator # human teleop
    ros2 run g2cell_datagen episode_runner --batch pick_v1 --seeds 0-9 --dry-run         # print the plan only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import yaml

from g2cell_datagen.randomize import load_config, validate_config
from g2cell_datagen.runner import EpisodeRunner
from g2cell_datagen.store import EpisodeStore, parse_seeds

HERE = Path(__file__).resolve().parent


def _share(pkg: str, rel: str) -> str:
    """Installed share file, else the source tree (host runs)."""
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415
        p = Path(get_package_share_directory(pkg)) / rel
        if p.is_file():
            return str(p)
    except Exception:  # noqa: BLE001
        pass
    return str(HERE.parent.parent / pkg / rel)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="episode_runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="batch name -> <episodes-root>/<batch>/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", help="seed list, e.g. 0-49 or 0-9,20")
    g.add_argument("--n", type=int, help="number of episodes (seeds --seed-start .. +n-1)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--limit", type=int, help="run at most this many pending episodes now")
    ap.add_argument("--mode", choices=("scripted", "operator"), default="scripted")
    ap.add_argument("--reset", choices=("none", "teleport", "restart"), help="override datagen.yaml reset.mode")
    ap.add_argument("--profile", default=os.environ.get("PROFILE", "collect"),
                    help="render profile for reset=restart (stage1_up.sh PROFILE)")
    ap.add_argument("--episodes-root", default="/g2cell_output/episodes")
    ap.add_argument("--config", default=None, help="datagen.yaml (default: the package's)")
    ap.add_argument("--tasks-file", default=None, help="g2cell_tasks tasks.yaml (default: the package's)")
    ap.add_argument("--unverified-as", choices=("true", "false", "unknown"), default="unknown",
                    help="HDF5 success for PickBag 'unverified' (motion done, no scorer yet)")
    ap.add_argument("--no-retry-errors", action="store_true", help="skip seeds whose last attempt was an error")
    ap.add_argument("--max-consecutive-errors", type=int, default=3)
    ap.add_argument("--task-timeout", type=float, default=600.0)
    ap.add_argument("--external-stack", action="store_true",
                    help="do not start motion_server/pick_bag_server (an external pick_bag_server ignores the "
                         "per-episode tasks.yaml: only bag/target are randomised then)")
    ap.add_argument("--no-home", action="store_true", help="do not home both arms before each episode")
    ap.add_argument("--no-record", action="store_true", help="run without the recorder (plumbing tests)")
    ap.add_argument("--node-ros-arg", action="append", default=[],
                    help="extra --ros-args for the managed nodes (offline smoke remaps), repeatable")
    ap.add_argument("--joint-states-topic", default="/joint_states")
    ap.add_argument("--dry-run", action="store_true", help="print the per-seed randomisation; no ROS, no files")
    return ap


def seeds_of(a) -> list:
    return parse_seeds(a.seeds) if a.seeds else list(range(a.seed_start, a.seed_start + a.n))


def load_configs(a):
    tasks_path = a.tasks_file or _share("g2cell_tasks", "config/tasks.yaml")
    cfg_path = a.config or _share("g2cell_datagen", "config/datagen.yaml")
    tasks_doc = yaml.safe_load(Path(tasks_path).read_text())
    doc = yaml.safe_load(Path(cfg_path).read_text())
    return doc, tasks_doc, load_config(doc, tasks_doc), {"config": cfg_path, "tasks_file": tasks_path}


def console_operator(input_fn=input, print_fn=print):
    """Operator-in-the-loop prompt: the episode is recording; the human teleoperates in the teleop window
    (ros2 run g2cell_teleop keyboard_teleop) and answers here when done."""
    keys = {"s": "success", "f": "failure", "d": "discard", "q": "quit"}

    def ask(name, spec):
        tgt = "pack carton" if spec.target == "crate" else f"rack tote {spec.lane}"
        print_fn(f"\n=== {name} (seed {spec.seed}) RECORDING ===\n"
                 f"  pick {spec.bag_id} (label {spec.label}) at x {spec.x:.3f} y {spec.y:.3f} yaw {spec.yaw_deg:.0f} deg"
                 f" -> {tgt}\n  teleop in the teleop window (do NOT press its r key: this runner records), then answer:"
                 f"\n  [s] success  [f] failure  [d] discard (redo later)  [q] discard + quit")
        while True:
            k = input_fn(f"{name}> ").strip().lower()[:1]
            if k in keys:
                return keys[k]
            print_fn("  type s, f, d or q, then Enter")
    return ask


def write_batch_json(store: EpisodeStore, a, doc, tasks_doc, paths, seeds):
    p = store.dir / "batch.json"
    runs = []
    if p.is_file():
        try:
            runs = json.loads(p.read_text()).get("runs", [])
        except ValueError:
            runs = []
    runs.append({"at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "argv": sys.argv[1:], "seeds": seeds})
    store.dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"batch": a.batch, "mode": a.mode, "reset": a.reset or doc["reset"]["mode"],
                             "unverified_as": a.unverified_as, "paths": paths, "datagen": doc,
                             "tasks": tasks_doc, "runs": runs}, indent=2))


def main(argv=None, backend_factory=None, operator=None) -> int:
    a = build_parser().parse_args(argv)
    doc, tasks_doc, cfg, paths = load_configs(a)
    problems = validate_config(cfg)
    if problems:
        print("datagen config invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    seeds = seeds_of(a)
    store = EpisodeStore(Path(a.episodes_root) / a.batch)
    if a.dry_run:
        run = EpisodeRunner(cfg, None, store, mode="scripted", batch=a.batch, log=lambda *_: None)
        pend = set(store.pending(seeds))
        for p in run.plan(seeds):
            b = p["bag"]
            print(f"seed {p['seed']:5d} {'pending' if p['seed'] in pend else 'done   '} {b['id']} {b['label']:6s} "
                  f"x {b['x']:+.3f} y {b['y']:.3f} yaw {b['yaw_deg']:+6.1f} -> {p['target']} {p['lane']}")
        return 0
    reset_mode = a.reset or doc["reset"]["mode"]
    write_batch_json(store, a, doc, tasks_doc, paths, seeds)
    if backend_factory is None:
        from g2cell_datagen.ros_backend import RosBackend  # noqa: PLC0415 - host dry runs need no rclpy

        def backend_factory():
            return RosBackend(batch=a.batch, batch_dir=store.dir, tasks_doc=tasks_doc, reset_cfg=doc["reset"],
                              mode=a.mode, reset_mode=reset_mode, profile=a.profile,
                              manage_stack=not a.external_stack, home_before=not a.no_home,
                              record=not a.no_record, task_timeout_s=a.task_timeout,
                              node_ros_args=[x for r in a.node_ros_arg for x in r.split()],
                              joint_states_topic=a.joint_states_topic, log=_log)
    backend = backend_factory()
    if a.mode == "operator" and operator is None:
        operator = console_operator()
    run = EpisodeRunner(cfg, backend, store, mode=a.mode, operator=operator, unverified_as=a.unverified_as,
                        batch=a.batch, retry_errors=not a.no_retry_errors,
                        max_consecutive_errors=a.max_consecutive_errors, log=_log)
    try:
        summary = run.run(seeds, limit=a.limit)
    except KeyboardInterrupt:
        print("interrupted: the current episode is marked 'interrupted' and reruns on resume", file=sys.stderr)
        return 130
    finally:
        backend.close()
    print(json.dumps(summary, indent=2))
    return 1 if summary["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
