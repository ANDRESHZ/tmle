import os
import shutil
import json
import codecs
import argparse

# TODO(lukasz): add logger which will give a summary of every directory.
parser = argparse.ArgumentParser()
parser.add_argument('--json', '-j', type=str)
parser.add_argument('--source', '-s', type=str)
parser.add_argument('--destination', '-d', type=str)
args = parser.parse_args()


LABELS = [
    'bird',
    'boat',
    'cat',
    'dog',
    'flower',
    'frog',
    'jumbojet',
    'mushroom',
    'sportscar',
    'tree'
]


def load_json(path_to_json: str) -> dict:
    """Load json with information about train and test sets."""
    with codecs.open(path_to_json, encoding='utf-8') as train_test_info:
        return json.loads(train_test_info.read())


def create_dirs(parent_directories: list, childrens: list, *args, **kwargs) -> None:
    """Create directories in which we will store the images
    from train and test sets."""
    for parent_dir in parent_directories:
        for children in childrens:
            try:
                os.makedirs(os.path.join(parent_dir, children), *args, **kwargs)
            except OSError:
                msg = ('Directory {dir} already exists. In order to overwrite '
                       'directory please set exists_ok=True')
                print(msg.format(dir=os.path.join(parent_dir, children)))


def train_test_split(path_to_json: str, source_dir: str, data_dir: str) -> None:
    """Make train-test-split."""
    train_test_metadata = load_json(path_to_json)
    for dataset_name, images in train_test_metadata.items():
        for image in images:
            shutil.copy(
                src=os.path.join(source_dir, image),
                dst=os.path.join(data_dir, dataset_name, image)
            )


def main():
    # create directories
    create_dirs(
        parent_directories=[os.path.join(args.destination, dataset) for dataset in ['train', 'test']],
        childrens=LABELS
    )
    # perform train test split
    train_test_split(
        path_to_json=args.json,
        source_dir=args.source,
        data_dir=args.destination
    )


if __name__ == "__main__":
    main()