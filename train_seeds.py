"""
Train every method with several seeds, one run after the other, so that the
comparison of collect_marl.py / dashboard_marl.py can tell differences between
the methods from differences between training runs.

  python train_seeds.py
  python train_seeds.py --methods mappo mappo_comm5 --seeds 0 1 2

All runs use the same settings (COMMON, plus --extra), only the method and the
seed change. Runs go to <root>/<method>_s<seed>. Finished runs (the last update
is in progress.csv) are skipped, so the script can be restarted after an
interruption; an unfinished run is trained again from the start (a resumed run
would not have the same learning rate schedule). The runs are ordered seed by
seed, so that all methods have the same number of finished seeds at any time.
An existing finished run with the same settings can be used as a seed by
linking it, e.g. ln -s ../marl_4_v2 experiments/seeds/mappo_s0.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

METHODS = {
    'mappo': [],
    'ippo': ['--algo', 'ippo'],
    'mappo_comm5': ['--comm_range', '5'],
}
COMMON = ['--num_agents', '4', '--num_envs', '128', '--num_steps', '128', '--minibatch_size', '4096',
          '--total_steps', '50000000']


def num_updates(args):
    get = lambda k: int(args[args.index(k) + 1])
    return get('--total_steps') // (get('--num_envs') * get('--num_steps'))


def finished(logdir, updates):
    path = os.path.join(logdir, 'progress.csv')
    if not os.path.exists(path) or not os.path.exists(os.path.join(logdir, 'checkpoint', 'params.msgpack')):
        return False
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return len(rows) > 0 and int(rows[-1]['update']) >= updates


def check_settings(logdir, method, seed, args):
    """The config of an existing run must match the requested one"""
    with open(os.path.join(logdir, 'config.json')) as f:
        config = json.load(f)
    flat = {**config['env'], **config['train']}
    for k, v in list(zip(args[::2], args[1::2])) + [('--seed', str(seed))]:
        have = flat.get(k[2:])
        same = float(have) == float(v) if isinstance(have, (int, float)) else str(have) == v
        assert same, f'{logdir}: {k[2:]} is {have}, expected {v}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--root', default='experiments/seeds', type=str)
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='more train_marl.py arguments')
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    env = dict(os.environ)
    env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    for seed in args.seeds:
        for method in args.methods:
            logdir = os.path.join(args.root, f'{method}_s{seed}')
            run_args = COMMON + METHODS[method] + ['--seed', str(seed)] + args.extra
            if finished(logdir, num_updates(run_args)):
                check_settings(logdir, method, seed, COMMON + METHODS[method])
                print(f'{logdir}: finished, skipped', flush=True)
                continue
            if os.path.islink(logdir):
                sys.exit(f'{logdir} is a link to an unfinished run, remove it first')
            os.makedirs(logdir, exist_ok=True)
            cmd = [sys.executable, 'train_marl.py', '--logdir', logdir] + run_args
            print(f'{time.strftime("%Y-%m-%d %H:%M")}  {" ".join(cmd)}', flush=True)
            t0 = time.time()
            with open(os.path.join(logdir, 'train.log'), 'w') as log:
                ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            if ret != 0:
                sys.exit(f'{logdir}: training failed (exit code {ret}), see {logdir}/train.log')
            print(f'{logdir}: done in {(time.time() - t0) / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
