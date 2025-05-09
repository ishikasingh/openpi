import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import os
import pickle
import ipdb



config = _config.get_config("pi0_fast_libero")
checkpoint_dir = download.maybe_download("/lustre/fsw/portfolios/nvr/users/ishikas/openpi/checkpoints/pi0_fast_libero/my_experiment/13500")




# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

tmp_obs_path = f'/lustre/fsw/portfolios/nvr/users/ishikas/ogvla_realworld/data/eval_openpi/'

obs_input_path = tmp_obs_path + 'obs_input.pkl'

while True:
    while True:
        user_input = input("Press c to continue...")
        if user_input.lower() == 'c':
            break

    # Pull latest changes from ogvla repo
    ogvla_repo_path = '/lustre/fsw/portfolios/nvr/users/ishikas/ogvla_realworld'
    print(f'Pulling latest changes from {ogvla_repo_path}...')
    try:
        import subprocess
        subprocess.run(['git', '-C', ogvla_repo_path, 'pull'], check=True)
        print('Successfully pulled latest changes')
    except subprocess.CalledProcessError as e:
        print(f'Error pulling changes: {e}')
        print('Continuing with existing code...')

    print('waiting for observation input at...', tmp_obs_path)
    while not os.path.exists(obs_input_path):
        continue
    
    if os.path.exists(obs_input_path):
        with open(obs_input_path, 'rb') as f:
            obs_input = pickle.load(f)


    # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
    example = {
        "observation/image": obs_input["color"],
        "observation/state": obs_input["state"][:7],
        "prompt": obs_input["task_name"]
    }
    result = policy.infer(example)
    print(obs_input["state"][:7])
    print(result['actions'])

    os.remove(obs_input_path)
    action_path = tmp_obs_path + 'action'
    with open(action_path, 'wb') as f:
        pickle.dump(result['actions'].tolist(), f)

    try:
        subprocess.run(['git', '-C', os.path.dirname(action_path), 'add', os.path.basename(action_path)], check=True)
        subprocess.run(['git', '-C', os.path.dirname(action_path), 'commit', '-m', 'Update action file'], check=True)
        subprocess.run(['git', '-C', os.path.dirname(action_path), 'push'], check=True)
        print('Successfully pushed to git repo')
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to git repo: {e}")

