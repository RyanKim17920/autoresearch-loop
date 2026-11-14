"""loop.core — the v3 decision engine (docs/loop-core-v3.md): one posterior from raw runs.

``data``     read-only typed view of the ledger (nodes, runs, observations, seed pools, lanes, carriers)
``model``    noise components, CUPED, screening composite, hierarchical EB prior, posterior, max-of-K
``seeds``    SELECT / TEST seed pools, never-used draws, ``check_submit`` (SeedPoolViolation), historical blocks
``test``     the sequential fresh-seed e-value test vs the incumbent
``status``   pure ``derive()`` / ``chain()``; ``decide_open`` for the open test blocks
``acquire``  knowledge-gradient-per-GPU-h ranking of idea / seed / wave / test actions, harm limit, fill rule
``backtest`` time-ordered replay of the posterior against fresh-seed outcomes; checks (c), (d)

Stdlib only. Nothing here writes to the ledger except ``seeds.migrate`` (the additive ``seed_blocks`` table)."""
