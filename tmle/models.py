import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from collections import defaultdict
from typing import Optional, Tuple
from sklearn.metrics import balanced_accuracy_score
from .dataloaders import ImageFoldersDataset


class TransferLearning:

    def train(
            self,
            model: torchvision.models,
            criterion: torch.nn,
            optimizer: torch.optim,
            train_dataset: ImageFoldersDataset,
            test_dataset: ImageFoldersDataset,
            model_dir: Optional[str] = None,
            model_name: Optional[str] = None,
            n_epochs: int = 25,
            batch_size: int = 32,
            shuffle: bool = True
    ):
        """

        :param train_dataset:
        :param test_dataset:
        :param model_dir:
        :param model_name:
        :param n_epochs:
        :param batch_size:
        :param shuffle:
        :return:
        """
        metrics = defaultdict(list)
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
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if data_idx % 100 == 0:
                    msg = '[%d, %5d] loss: %.3f'
                    print(msg % (epoch + 1, data_idx + 1, running_loss / 100))
                    running_loss = 0.0
            # TODO(lukasz): measure accuracy_train during training.
            accuracy_train = self.score(model, train_dataset)
            accuracy_test = self.score(model, test_dataset)
            metrics['acc_train'].append(accuracy_train)
            metrics['acc_test'].append(accuracy_test)
            msg = '[%d] train score: %.3f, test score: %.3f'
            print(msg % (epoch + 1, accuracy_train, accuracy_test))
            # save model (make sure that Google Colab do not destroy your results).
            if accuracy_test > best_accuracy_test:
                torch.save(
                    model,
                    os.path.join(model_dir, '.'.join([
                        model_name + '_' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())),
                        'pth'])))
                best_accuracy_test = accuracy_test

    def score(self, model, dataset: ImageFoldersDataset) -> float:
        """

        :param dataset:
        :return:
        """
        with torch.no_grad():
            # remember that you must call `model.eval()` to set dropout and batch
            # normalization layers to evaluation mode before running the inference.
            model.eval()
            y_true, y_pred = np.zeros(len(dataset)), np.zeros(len(dataset))
            batch_idx = 0
            for data in dataset.loader(batch_size=32):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device('cuda:0'))
                    labels = labels.to(torch.device('cuda:0'))
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                batch_size = labels.size(0)
                y_true[batch_idx:batch_idx+batch_size] = labels.cpu().numpy()
                y_pred[batch_idx:batch_idx+batch_size] = pred.detach().cpu().numpy()
        return balanced_accuracy_score(y_true, y_pred)