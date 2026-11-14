"""kernel — task-independent facts. Imports nothing above it.

The bottom layer of the rebuild (plan section 2). Nothing in here may import
``config``, ``policy``, ``agent``, ``runtime``, ``ui`` or any adapter: the
dependency arrow points INTO this package and never out of it. That rule is
enforced by a test (``tests/test_kernel_stats.py``), not by convention, because
the current tree's single worst structural property is that its "constants"
module imports half the system.
"""
