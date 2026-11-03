"""
Train robots for multi-agent lawn mowing with MAPPO or IPPO in JAX.

Examples:
  python -m scripts.multi_agent.train --logdir experiments/marl_4robots --num_agents 4
  python -m scripts.multi_agent.train --logdir experiments/ippo_4robots --num_agents 4 --algo ippo
  python -m scripts.multi_agent.train --logdir experiments/marl_small --num_envs 64 --num_steps 64 --total_steps 1000000
  python -m scripts.multi_agent.train --config_from experiments/mappo_sb3_v2 --algo ippo --logdir experiments/ippo_sb3_v2 \
      --checkpoint experiments/sb3_init/checkpoint

With --config_from, the settings of another run (its config.json) are the
defaults, so a run can repeat another one and only change some settings
(e.g. the same training as a MAPPO run, with IPPO). The seed is copied too.

Every field of marl.config.EnvConfig and marl.config.TrainConfig is available
as a command line argument (booleans as --flag / --no-flag).
"""
import argparse
import json
import os
from datetime import datetime

# Let JAX preallocate 90% of the GPU memory instead of the default 75%
# (set before JAX is imported, an explicit value in the environment wins)
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from marl.config import EnvConfig, TrainConfig, add_dataclass_args, read_config


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config_from', default=None, type=str)
    config_from = pre.parse_known_args()[0].config_from

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_from', default=None, type=str,
                        help='run directory, checkpoint directory or config.json whose settings are the defaults')
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--maps_dir', default='maps', type=str)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='checkpoint directory to initialize the actor and critic from')
    add_dataclass_args(parser, EnvConfig, 'env')
    add_dataclass_args(parser, TrainConfig, 'train')
    if config_from is not None:
        parser.set_defaults(**read_config(config_from))
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
