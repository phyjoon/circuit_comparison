""" Experiment manager which helps saving experiments.

"""
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml

EXP_BASE_DIR = './results'
FORMAT = "%Y%m%d_%H%M%S"
DATETIME = datetime.now()


project_name = None
exp_name = None


def init(project, name, config):
    global project_name, exp_name
    project_name = project
    exp_name = name
    wandb.init(
        project=project_name,
        name=exp_name,
        dir=str(get_result_dir())
    )
    save_config(config)


def get_result_dir():
    tag = DATETIME.strftime(FORMAT)
    if project_name:
        tag = f'{tag}_{project_name}'
    if exp_name:
        tag = f'{tag}_{exp_name}'
    exp_dir = Path(EXP_BASE_DIR) / tag
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
    return exp_dir


def get_result_path(filename):
    """ Get file path under result directory.

    Args.
        tag: str, experiment tag name.
        filename: str, file name.
    Returns
        a file path.
    """
    exp_dir = get_result_dir()
    filepath = exp_dir / filename
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    return filepath


def load_config(filepath):
    """ Load experiment configuration from a file path.

    Args:
        filepath: str, configuration file path

    Returns:
        dict, experiment configuration.
    """
    config_path = Path(filepath)
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config):
    """ Save experiment configuration as a yaml file

    Args:
        config: Namespace or dict, experiment configuration.
    """
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    config_path = get_result_path('config.yaml')
    with config_path.open('w') as f:
        yaml.safe_dump(config, f)


if __name__ == '__main__':
    fp = get_result_path('nnnmm/asdf.png')
    print(fp)
