"""
The original single-agent tests need the dependencies of requirements.txt
(gym, OpenCV, torch, ...), the multi-agent tests (test_marl.py) only need
those of requirements-marl.txt. Skip the tests whose dependencies are not
installed, so that pytest can be run in either environment.
"""
import importlib.util


def _installed(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


collect_ignore = []
if not _installed('gym', 'cv2', 'torch', 'matplotlib'):
    collect_ignore += ['test_mower_env.py', 'test_utils.py', 'test_make_giant_maps.py']
if not _installed('jax', 'flax', 'optax', 'scipy'):
    collect_ignore += ['test_marl.py']
