"""Single-publisher guard for /joint_command (no ROS imports).

Genie Sim applies whatever arrives on /joint_command; two publishers (e.g. the keyboard teleop
and the motion server) would fight and make the arm jump. The motion server refuses to start
while any other node publishes there.
"""
from __future__ import annotations


def _fq(ns: str, name: str) -> str:
    ns = ns.rstrip("/")
    return f"{ns}/{name}" if ns else f"/{name}"


def publisher_conflicts(infos, own=None) -> list[str]:
    """infos: [(node_namespace, node_name)] of /joint_command publishers. Returns the fully
    qualified names of every publisher that is not `own`, sorted, de-duplicated."""
    me = _fq(*own) if own else None
    return sorted({_fq(ns, n) for ns, n in infos if _fq(ns, n) != me})
