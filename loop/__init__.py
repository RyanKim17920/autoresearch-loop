"""loop — a lean, task-agnostic research loop built on ``kernel``.

Design of record: ``docs/loop/DESIGN.md``. Every module codes against the typed
interfaces in :mod:`loop.contracts`. Nothing under ``loop/`` may import the root
``config`` module (it resolves one task's noise model at import time); see
DESIGN section 0.4.
"""

__all__ = ["contracts"]
