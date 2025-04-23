"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import os
import pickle
from tqdm import tqdm
import pdb
import numpy as np

REPO_NAME = "ishika/realworld_franka"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = True):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # "wrist_image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format

    # data_dir = '/lustre/fsw/portfolios/nvr/users/ishikas/xvila-robotics/realworld_data/train_final'
    files = os.listdir(data_dir)
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    print('processing files: ', files[0:5], ' .... ', files[-5:-1])

    for fidx in tqdm(range(len(files))): #[:50]):
        if files[fidx].endswith('.pkl'):
            with open(os.path.join(data_dir, files[fidx]), 'rb') as f:
                raw_data = pickle.load(f)
            timestep = 0
            print([raw_data[i]['gripper_state'] for i in range(len(raw_data))])
            for i, keypoint in enumerate(raw_data[1:]):
                # pdb.set_trace()
                print(len(keypoint['images']))

                # Calculate downsampling indices to match joint_traj length
                num_images = len(keypoint['images'])
                num_joints = len(keypoint['joint_traj'])
                if num_images > num_joints:
                    # Use linear spacing to get indices that match joint_traj length
                    sample_indices = np.linspace(0, num_images-1, num_joints, dtype=int)
                    keypoint['images'] = [keypoint['images'][i] for i in sample_indices]

                for idx in range(0, len(keypoint['joint_traj'])-1):
                    dataset.add_frame(
                        {
                            "image": keypoint['images'][idx]['color'][..., ::-1],
                            # "wrist_image": step["observation"]["wrist_image"],
                            "state": keypoint['joint_traj'][idx],
                            "actions": keypoint['joint_traj'][idx+1],
                        }
                    )
            language_instruction = files[fidx].split('_trajectory')[0]
            dataset.save_episode(task=language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["realworld", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
