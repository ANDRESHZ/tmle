import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from typing import Optional, Tuple
from sklearn.metrics import balanced_accuracy_score
from .dataloaders import ImageFoldersDataset


class TransferLearning:

    def __init__(
            self,
            model: torchvision.models,
            criterion: torch.nn,
            optimizer: torch.optim
    ) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
            self,
            train_dataset: ImageFoldersDataset,
            test_dataset: ImageFoldersDataset,
            n_epochs: int = 25,
            batch_size: int = 32,
            shuffle: bool = True
    ) -> None:
        """

        :param train_dataset:
        :param test_dataset:
        :param n_epochs:
        :param batch_size:
        :param shuffle:
        :return:
        """
        best_accuracy_test = 0.
        for epoch in range(n_epochs):
            running_loss = 0.0
            for data_idx, data in enumerate(train_dataset.loader(
                batch_size=batch_size,
                shuffle=shuffle
            )):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device('cuda:0'))
                    labels = labels.to(torch.device('cuda:0'))
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if data_idx % 100 == 0:
                    msg = '[%d, %5d] loss: %.3f'
                    print(msg % (epoch + 1, data_idx + 1, running_loss / 100))
                    running_loss = 0.0
            accuracy_train = self.score(train_dataset)
            accuracy_test = self.score(test_dataset)
            msg = '[%d] train score: %.3f, test score: %.3f'
            print(msg % (epoch + 1, accuracy_train, accuracy_test))
            if accuracy_test > best_accuracy_test:
                # TODO(lukasz): save model state
                pass

    def score(self, dataset: ImageFoldersDataset) -> float:
        """

        :param dataset:
        :return:
        """
        with torch.no_grad():
            y_true, y_pred = np.zeros(len(dataset)), np.zeros(len(dataset))
            batch_idx = 0
            for data in dataset.loader(batch_size=32):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device('cuda:0'))
                    labels = labels.to(torch.device('cuda:0'))
                outputs = self.model(inputs)
                _, pred = torch.max(outputs.data, 1)
                batch_size = labels.size(0)
                y_true[batch_idx:batch_idx+batch_size] = labels.cpu().numpy()
                y_pred[batch_idx:batch_idx+batch_size] = pred.detach().cpu().numpy()
        return balanced_accuracy_score(y_true, y_pred)