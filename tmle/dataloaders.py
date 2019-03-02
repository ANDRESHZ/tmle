import os
import random
import numpy as np
import torch
import torchvision

from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class ImageFoldersDataset(object):

    def __init__(
            self,
            path_to_data: str,
            transform: torchvision.transforms,
            seed: int = 42,
            *args,
            **kwargs
    ) -> None:
        """Initialize `ImageFoldersDataset`. This class implements methods to load
        (mini-batches or all) images from directories trees consistent with structure
        accepted by `torchvision.datasets.ImageFolder`.

        :param path_to_data:
        :param transform:
        :param seed:
        :param args:
        :param kwargs:
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.dataset = datasets.ImageFolder(
            root=path_to_data,
            transform=transform,
            *args,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def loader(
            self,
            batch_size: int = 4,
            shuffle: bool = True,
            seed: int = 42,
            *args,
            **kwargs
    ) -> torch.utils.data.DataLoader:
        """

        :param batch_size:
        :param shuffle:
        :param args:
        :param kwargs:
        :return:
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        return DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=self._init_fn(seed)
        )

    def load_all_images(
            self,
            img_shape: tuple = (224, 224, 3),
            save: bool = False,
            output_dir: Optional[str] = None,
            output_img_name: Optional[str] = None,
            output_label_name: Optional[str] = None,
            *args,
            **kwargs
    ) -> Optional[List[np.ndarray, np.ndarray]]:
        """

        :param img_shape:
        :param save:
        :param output_dir:
        :param output_img_name:
        :param output_label_name:
        :param args:
        :param kwargs:
        :return:
        """
        images, labels = np.zeros((len(self), *img_shape)), np.zeros(len(self))
        for data_idx, data in enumerate(self.dataset):
            image, label = data
            images[data_idx] = image.permute(1, 2, 0).numpy()
            labels[data_idx] = label.numpy()
        if save:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(
                file=os.path.join(output_dir, '.'.join([output_img_name, '.npy'])),
                arr=images,
                *args,
                **kwargs
            )
            np.save(
                file=os.path.join(output_dir, '.'.join([output_label_name, '.npy'])),
                arr=labels,
                *args,
                **kwargs
            )
            return None
        else:
            return images, labels
