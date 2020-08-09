""" Experiment manager which helps saving experiments.

"""

from pathlib import Path

from datetime import datetime


EXP_BASE_DIR = './results'
FORMAT = "%Y%m%d_%H%M%S"
DATETIME = datetime.now()


def get_result_dir():
    exp_dir = Path(EXP_BASE_DIR) / DATETIME.strftime(FORMAT)
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
    return exp_dir


def get_result_path(tag, filename):
    """ Get file path under result directory.

    Args.
        tag: str, experiment tag name.
        filename: str, file name.
    Returns
        a file path.
    """
    exp_dir = get_result_dir()
    filepath = exp_dir / tag / filename
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    return filepath


if __name__ == '__main__':
    fp = get_result_path('test', 'nnnmm/asdf.png')
    print(fp)
