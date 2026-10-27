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

Several workers (GPUs, also on other machines with the same file system) can
share the runs: each worker claims the next run that is neither finished nor
claimed by creating <run>/claim.json atomically, so no run is trained twice.
A claim is stale (taken over) when its process is not running on this machine,
or, from another machine, when the run has not written anything for
--stale_minutes. Use one worker per GPU:

  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python train_seeds.py
  python train_seeds.py --check     # exit code 0 when all runs are finished
"""
import argparse
import csv
import json
import os
import socket
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


def last_activity(logdir):
    times = [os.path.getmtime(os.path.join(logdir, f)) for f in ('claim.json', 'train.log', 'progress.csv')
             if os.path.exists(os.path.join(logdir, f))]
    return max(times, default=0)


def claim(logdir, stale_minutes):
    """Claims the run for this worker, returns False if another worker has it"""
    path = os.path.join(logdir, 'claim.json')
    me = dict(host=socket.gethostname(), pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M'))
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                other = json.load(open(path))
            except (OSError, ValueError):
                return False      # being written by another worker
            if other['host'] == me['host']:
                stale = not os.path.exists(f'/proc/{other["pid"]}')
            else:
                stale = time.time() - last_activity(logdir) > 60 * stale_minutes
            if not stale:
                return False
            print(f'{logdir}: taking over the stale claim of {other}', flush=True)
            os.remove(path)
            continue
        with os.fdopen(fd, 'w') as f:
            json.dump(me, f)
        return True
    return False


def release(logdir):
    path = os.path.join(logdir, 'claim.json')
    if os.path.exists(path):
        os.remove(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--root', default='experiments/seeds', type=str)
    parser.add_argument('--stale_minutes', default=30, type=float,
                        help='a claim from another machine without activity for this long is taken over')
    parser.add_argument('--check', action='store_true', help='only report, exit code 0 when all runs are finished')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='more train_marl.py arguments')
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    env = dict(os.environ)
    env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    host = socket.gethostname()
    missing = 0
    # passes over the runs until one claims nothing: runs that another worker
    # held during a pass and then released (stopped) are trained in the next
    trained = True
    while trained:
        trained, missing = False, 0
        for seed, method in [(s, m) for s in args.seeds for m in args.methods]:
            logdir = os.path.join(args.root, f'{method}_s{seed}')
            run_args = COMMON + METHODS[method] + ['--seed', str(seed)] + args.extra
            if finished(logdir, num_updates(run_args)):
                check_settings(logdir, method, seed, COMMON + METHODS[method])
                if args.check:
                    print(f'{logdir}: finished', flush=True)
                continue
            if args.check:
                state = json.load(open(os.path.join(logdir, 'claim.json'))) \
                    if os.path.exists(os.path.join(logdir, 'claim.json')) else 'not started'
                print(f'{logdir}: not finished ({state})', flush=True)
                missing += 1
                continue
            if os.path.islink(logdir):
                sys.exit(f'{logdir} is a link to an unfinished run, remove it first')
            os.makedirs(logdir, exist_ok=True)
            if not claim(logdir, args.stale_minutes):
                continue
            cmd = [sys.executable, 'train_marl.py', '--logdir', logdir] + run_args
            print(f'{time.strftime("%Y-%m-%d %H:%M")}  [{host} GPU {env.get("CUDA_VISIBLE_DEVICES", "all")}]  '
                  f'{" ".join(cmd)}', flush=True)
            t0 = time.time()
            try:
                with open(os.path.join(logdir, 'train.log'), 'w') as log:
                    ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            finally:
                release(logdir)
            if ret != 0:
                sys.exit(f'{logdir}: training failed (exit code {ret}), see {logdir}/train.log')
            print(f'{logdir}: done in {(time.time() - t0) / 3600:.1f} h', flush=True)
            trained = True
        if args.check:
            break
    if args.check:
        print(f'{missing} runs not finished')
        sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
