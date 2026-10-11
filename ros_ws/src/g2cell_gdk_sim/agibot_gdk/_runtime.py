"""The process-wide Controller that gdk_init() creates (tests install one with a fake port)."""
from __future__ import annotations

_controller = None
_backend = None


def install(controller, backend=None) -> None:
    global _controller, _backend
    _controller, _backend = controller, backend


def uninstall():
    """Forgets the controller and returns the backend (to shut it down)."""
    global _controller, _backend
    backend, _controller, _backend = _backend, None, None
    return backend


def active() -> bool:
    return _controller is not None


def controller():
    if _controller is None:
        raise RuntimeError("kRuntimeError: agibot_gdk (sim) is not initialised: call agibot_gdk.gdk_init() first")
    return _controller
