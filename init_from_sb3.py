"""
Initialize the MARL actor from the trained single-agent policy (SAC of
stable-baselines3, e.g. weights/mowing_tv1), to start MAPPO from it with
train_marl.py --checkpoint.

The single-agent actor (rlm.architectures.StackedMapFeaturesExtractor, then
two 256 layers and the mean) has the same layers as marl.networks.Actor
without the pose branch of the other robots (PoseEncoder):
  * the multi-scale maps and lidar encoder is copied. The egocentric maps of
    rlm have the robot's forward direction along the columns and its right
    along the rows (cv2.warpAffine in MowerEnv._get_transform_matrix), the
    maps of MultiMowerEnv have forward up, i.e. rlm = rot90(marl, -1): the
    convolution kernels are rotated back and the dense layer after the
    convolutions is permuted accordingly
  * the first layer after the encoders gets the weights of the single-agent
    layer for the map/lidar features and zeros for the pose features, so
    the initial policy is the single-agent one and the pose branch is learned
  * the mean is copied (tanh of the mean is the deterministic action of SAC
    and of the squashed MARL actor). SAC has a state-dependent log std, the
    MARL actor a learned constant: it starts at init_log_std
The critic is newly initialized.

The policy is read from the zip without torch (the pickle of torch.save is
parsed with numpy). A numpy forward pass of the single-agent encoder checks
that the converted JAX encoder gives the same features.

  python init_from_sb3.py --load weights/mowing_tv1 --out experiments/sb3_init/checkpoint --lidar_fov 180
"""
import argparse
import io
import json
import os
import pickle
import zipfile

import numpy as np

from marl.config import EnvConfig, TrainConfig
from train_marl import add_dataclass_args


# ------------------------------------------------------------ torch-free load


class _Storage:
    def __init__(self, dtype, data):
        self.dtype, self.data = dtype, data


class _TorchUnpickler(pickle.Unpickler):
    DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
              'LongStorage': np.int64, 'IntStorage': np.int32, 'BoolStorage': np.bool_,
              'ByteStorage': np.uint8}

    def __init__(self, f, archive, prefix):
        super().__init__(f)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self._rebuild_tensor
        if module == 'torch' and name in self.DTYPES:
            return self.DTYPES[name]
        if module == 'collections' and name == 'OrderedDict':
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f'unexpected class {module}.{name}')

    def persistent_load(self, pid):
        _, dtype, key, _, numel = pid
        data = np.frombuffer(self.archive.read(f'{self.prefix}/data/{key}'), dtype=dtype, count=numel)
        return _Storage(dtype, data)

    @staticmethod
    def _rebuild_tensor(storage, offset, size, stride, *_):
        item = storage.data.itemsize
        return np.lib.stride_tricks.as_strided(storage.data[offset:], shape=size,
                                               strides=[s * item for s in stride]).copy()


def load_sb3_policy(path):
    """State dict (name: numpy array) of policy.pth in a stable-baselines3 zip"""
    with zipfile.ZipFile(path) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read('policy.pth')))
    prefix = inner.namelist()[0].split('/')[0]
    return _TorchUnpickler(io.BytesIO(inner.read(f'{prefix}/data.pkl')), inner, prefix).load()


# ---------------------------------------------------------------- conversion


def grouped_kernel(w, groups):
    """torch (out, in / groups, kh, kw) -> block diagonal (kh, kw, in, out)"""
    out_c, in_g = w.shape[:2]
    out_g = out_c // groups
    k = np.zeros(w.shape[2:] + (in_g * groups, out_c), dtype=np.float32)
    for g in range(groups):
        k[:, :, g * in_g:(g + 1) * in_g, g * out_g:(g + 1) * out_g] = \
            w[g * out_g:(g + 1) * out_g].transpose(2, 3, 1, 0)
    return k


def convert_actor(sd, actor_params, num_maps):
    """Actor params (flax dict) with the single-agent weights, see the module docstring"""
    p = 'actor.features_extractor.'
    params = jax.tree_util.tree_map(np.asarray, actor_params)
    enc = params['params']['MultiScaleEncoder_0']
    for i, t in enumerate([0, 2, 4, 6]):
        w = sd[f'{p}map_extractor.{t}.weight']
        # conv(rot90(x, -1), w) = rot90(conv(x, rot90(w, 1)), -1)
        enc[f'GroupedConv_{i}'] = {'kernel': np.rot90(grouped_kernel(w, num_maps), 1, axes=(0, 1)).copy(),
                                   'bias': sd[f'{p}map_extractor.{t}.bias']}
    # flatten: torch (C, a, b) of the rlm layout, flax (r, c, C) of the marl
    # layout, with rlm[a, b] = marl[S - 1 - b, a]
    w = sd[f'{p}map_extractor.9.weight']                                     # (256, C * S * S)
    C = enc['GroupedConv_3']['kernel'].shape[-1]
    S = int(round(np.sqrt(w.shape[1] // C)))
    w = w.reshape(-1, C, S, S)                                              # out, C, a, b
    r, c = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
    w = w[:, :, c, S - 1 - r]                                               # out, C, r, c
    enc['Dense_0'] = {'kernel': w.transpose(2, 3, 1, 0).reshape(-1, w.shape[0]),
                      'bias': sd[f'{p}map_extractor.9.bias']}
    enc['Dense_1'] = {'kernel': sd[f'{p}lidar_extractor.1.weight'].T, 'bias': sd[f'{p}lidar_extractor.1.bias']}
    enc['Dense_2'] = {'kernel': sd[f'{p}fused_extractor.0.weight'].T, 'bias': sd[f'{p}fused_extractor.0.bias']}

    head = params['params']
    w0 = sd['actor.latent_pi.0.weight'].T                                   # (256, 256)
    kernel = np.zeros_like(head['Dense_0']['kernel'])
    kernel[:w0.shape[0]] = w0                                               # pose features: 0
    head['Dense_0'] = {'kernel': kernel, 'bias': sd['actor.latent_pi.0.bias']}
    head['Dense_1'] = {'kernel': sd['actor.latent_pi.2.weight'].T, 'bias': sd['actor.latent_pi.2.bias']}
    head['Dense_2'] = {'kernel': sd['actor.mu.weight'].T, 'bias': sd['actor.mu.bias']}
    return params


def sb3_features(sd, maps, lidar, num_maps):
    """
    numpy forward pass of StackedMapFeaturesExtractor on rlm layout inputs:
    maps (B, num_maps * channels, S, S) in [0, 1], grouped by scale, lidar (B, R)
    """
    p = 'actor.features_extractor.'
    x = maps
    for t, stride in [(0, 2), (2, 1), (4, 1), (6, 1)]:
        w, b = sd[f'{p}map_extractor.{t}.weight'], sd[f'{p}map_extractor.{t}.bias']
        out_g, in_g, kh, kw = w.shape[0] // num_maps, w.shape[1], w.shape[2], w.shape[3]
        H = (x.shape[2] - kh) // stride + 1
        W = (x.shape[3] - kw) // stride + 1
        y = np.zeros((x.shape[0], w.shape[0], H, W), dtype=np.float64)
        for g in range(num_maps):
            xg = x[:, g * in_g:(g + 1) * in_g]
            wg = w[g * out_g:(g + 1) * out_g]
            for i in range(kh):
                for j in range(kw):
                    patch = xg[:, :, i:i + stride * H:stride, j:j + stride * W:stride]
                    y[:, g * out_g:(g + 1) * out_g] += np.einsum('bchw,oc->bohw', patch, wg[:, :, i, j])
        x = np.maximum(y + b[None, :, None, None], 0)
    relu = lambda v: np.maximum(v, 0)
    m = relu(x.reshape(x.shape[0], -1) @ sd[f'{p}map_extractor.9.weight'].T + sd[f'{p}map_extractor.9.bias'])
    l = relu(lidar @ sd[f'{p}lidar_extractor.1.weight'].T + sd[f'{p}lidar_extractor.1.bias'])
    return relu(np.concatenate([m, l], 1) @ sd[f'{p}fused_extractor.0.weight'].T + sd[f'{p}fused_extractor.0.bias'])


def check_encoder(sd, actor_params, env_cfg, train_cfg):
    """Max abs difference of the converted JAX encoder and the single-agent encoder on random inputs"""
    from marl.networks import MultiScaleEncoder
    rng = np.random.default_rng(0)
    H = env_cfg.input_size
    maps = rng.integers(0, 256, (8, H, H, env_cfg.obs_channels)).astype(np.uint8)
    lidar = rng.uniform(0, 1, (8, env_cfg.lidar_rays)).astype(np.float32)
    enc = MultiScaleEncoder(env_cfg.num_maps, train_cfg.hidden_dims, train_cfg.grouped_convs)
    with jax.default_matmul_precision('highest'):
        ours = enc.apply({'params': actor_params['params']['MultiScaleEncoder_0']}, maps, lidar)
    rlm_maps = np.rot90(maps.astype(np.float64) / 255, -1, axes=(1, 2)).transpose(0, 3, 1, 2)
    ref = sb3_features(sd, rlm_maps, lidar.astype(np.float64), env_cfg.num_maps)
    return float(np.abs(np.asarray(ours) - ref).max() / (np.abs(ref).max() + 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load', default='weights/mowing_tv1', type=str, help='single-agent weights directory')
    parser.add_argument('--out', required=True, type=str, help='checkpoint directory to write')
    add_dataclass_args(parser, EnvConfig, 'env')
    add_dataclass_args(parser, TrainConfig, 'train')
    args = vars(parser.parse_args())
    env_cfg = EnvConfig.from_dict(args)
    train_cfg = TrainConfig.from_dict(args)

    with open(os.path.join(args['load'], 'env_parameters.json')) as f:
        sa_env = json.load(f)
    for k in ('input_size', 'num_maps', 'scale_factor', 'meters_per_pixel', 'lidar_rays', 'lidar_range',
              'lidar_fov', 'overlap_observation', 'frontier_observation', 'all_unknown', 'constant_lin_vel'):
        if sa_env[k] != getattr(env_cfg, k):
            print(f'warning: {k} is {getattr(env_cfg, k)}, the single-agent policy was trained with {sa_env[k]}')
    assert env_cfg.map_channels == 2 + int(sa_env['frontier_observation']), \
        'the observation has other map channels than the single-agent one (robot_observation?)'

    sd = load_sb3_policy(os.path.join(args['load'], 'agent.zip'))
    actor, critic = make_networks(env_cfg, train_cfg)
    actor_params, critic_params = init_params(env_cfg, actor, critic, jax.random.PRNGKey(train_cfg.seed))
    actor_params = convert_actor(sd, actor_params, env_cfg.num_maps)
    err = check_encoder(sd, actor_params, env_cfg, train_cfg)
    print(f'encoder relative max difference to the single-agent encoder: {err:.2e}')
    assert err < 1e-4, 'the converted encoder does not match the single-agent encoder'

    os.makedirs(args['out'], exist_ok=True)
    state = {'actor': actor_params, 'critic': critic_params,
             'value_norm': {'mean': np.zeros(()), 'mean_sq': np.zeros(()), 'debias': np.zeros(())}}
    with open(os.path.join(args['out'], 'params.msgpack'), 'wb') as f:
        f.write(flax.serialization.to_bytes(jax.device_get(state)))
    with open(os.path.join(args['out'], 'config.json'), 'w') as f:
        json.dump({'env': env_cfg.to_dict(), 'train': train_cfg.to_dict(), 'init_from': args['load']}, f, indent=2)
    print(f'written {args["out"]}')


if __name__ == '__main__':
    import flax
    import jax
    from marl.mappo import init_params, make_networks
    main()
