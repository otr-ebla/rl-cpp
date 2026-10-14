"""
Multi-agent lawn mowing (coverage path planning) in JAX.

Everything that runs per step is written in JAX so that thousands of
environments can be simulated and trained in parallel on a GPU:
  marl.env       multi-agent mowing environment (pure functions, jit/vmap-able)
  marl.maps      map loading, random map generation and the map bank
  marl.networks  actor and centralized critic (Flax)
  marl.mappo     MAPPO training (centralized training, decentralized execution)
"""
