"""Shared helpers for reading the episode logs and per-view occupancy of a
scene dataset.

Used by common/dataset_UAB_unified.py, preprocessing/ and
preprocessing/reward_dataset/UAB_reward_generate_gt.py.
"""
import os
import re
import numpy as np
import torch


NUM_VIEWS = 98


def parse_fallen_log(path):
    """Parse fallen_log_revised.txt lines of the form:
        episode=000000015  push_1  post_03  fallen=['obj_10']
    Returns set of (scene_name, push_id, post_id).
    """
    bad = set()
    if not os.path.isfile(path):
        return bad
    pat = re.compile(r'episode=(\S+)\s+push_(\d+)\s+post_(\d+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                bad.add((m.group(1), int(m.group(2)), int(m.group(3))))
    return bad


_TRANSITION_PAIRS = {'AB': (1, 2), 'BC': (2, 3), 'CD': (3, 4), 'DE': (4, 5)}
_POSTS_PER_PUSH = 20


def parse_episode_log(log_path):
    """Parse all AB/BC/CD/DE transitions from episode_log.txt.

    Log format (one line per transition; only the `transition_XX: post_NN`
    prefix is matched, the rest of the line is free text):
        -> transition_AB: post_02 ... -> PRE_B
    post_XX is a GLOBAL step index across sub-scenes; convert to a LOCAL
    col_idx within push_{pre_push} via `global - (pre_push-1)*20`.
    Returns: list of {'col_idx': int, 'pre_push': int, 'post_push': int}.
    """
    transitions = []
    pat = re.compile(r'transition_(AB|BC|CD|DE)\s*:\s*post_(\d+)')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            pre_push, post_push = _TRANSITION_PAIRS[m.group(1)]
            global_post = int(m.group(2))
            col_idx = global_post - (pre_push - 1) * _POSTS_PER_PUSH
            transitions.append({
                'col_idx': col_idx,
                'pre_push': pre_push,
                'post_push': post_push,
            })
    return transitions


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def _load_views(occ_dir, view_indices):
    arrs = [np.load(os.path.join(occ_dir, f'{int(v):03d}.npz'))['occ']
            for v in view_indices]
    return np.stack(arrs).astype(np.float32)
