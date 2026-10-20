"""
Train robots for multi-agent lawn mowing with MAPPO or IPPO in JAX.

Examples:
  python train_marl.py --logdir experiments/marl_4robots --num_agents 4
  python train_marl.py --logdir experiments/ippo_4robots --num_agents 4 --algo ippo
  python train_marl.py --logdir experiments/marl_small --num_envs 64 --num_steps 64 --total_steps 1000000

Every field of marl.config.EnvConfig and marl.config.TrainConfig is available
as a command line argument (booleans as --flag / --no-flag).
"""
import argparse
import dataclasses
import json
import os
from argparse import BooleanOptionalAction
from datetime import datetime

# Let JAX preallocate 90% of the GPU memory instead of the default 75%
# (set before JAX is imported, an explicit value in the environment wins)
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from marl.config import EnvConfig, TrainConfig


def add_dataclass_args(parser, cls, title):
    group = parser.add_argument_group(title)
    for f in dataclasses.fields(cls):
        if f.type in (bool, 'bool'):
            group.add_argument(f'--{f.name}', default=f.default, action=BooleanOptionalAction)
        else:
            typ = {'int': int, 'float': float, 'str': str}.get(f.type, f.type)
            group.add_argument(f'--{f.name}', default=f.default, type=typ)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--maps_dir', default='maps', type=str)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='checkpoint directory to initialize the actor and critic from')
    add_dataclass_args(parser, EnvConfig, 'env')
    add_dataclass_args(parser, TrainConfig, 'train')
    args = vars(parser.parse_args())

    env_cfg = EnvConfig.from_dict(args)
    train_cfg = TrainConfig.from_dict(args)
    logdir = args['logdir'] or os.path.join('experiments', datetime.now().strftime('marl_%Y-%m-%d_%H%M%S'))
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump({'env': env_cfg.to_dict(), 'train': train_cfg.to_dict()}, f, indent=2)

    import jax
    from marl.mappo import train
    print('JAX devices:', jax.devices(), flush=True)
    print('env:', env_cfg, flush=True)
    print('train:', train_cfg, flush=True)
    train(env_cfg, train_cfg, logdir, maps_dir=args['maps_dir'], checkpoint=args['checkpoint'])


if __name__ == '__main__':
    main()
