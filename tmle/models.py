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


class TransferLearning(object):
    # TODO(lukasz) inherit from transformers.CNNFeatures
    def __init__(
            self,
            experiments_path: Optional[str] = None,
            experiments_name: Optional[str] = None
    ) -> None:

        self.save_experiment = os.path.join(
            experiments_path,
            '.'.join([experiments_name, 'pth'])
        )
        self.metrics = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
            self,
            model: torchvision.models,
            criterion: torch.nn,
            optimizer: torch.optim,
            train_dataset: ImageFoldersDataset,
            test_dataset: ImageFoldersDataset,
            n_epochs: int = 25,
            batch_size: int = 32,
            shuffle: bool = True,
            *args, **kwargs
    ):
        # TODO(lukasz): add scheduler for learning rate
        metrics = defaultdict(list)
        best_score_test = 0.
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.
            for data_idx, data in enumerate(train_dataset.loader(
                batch_size=batch_size,
                shuffle=shuffle
                # TODO(lukasz): add sampler for imbalanced dataset
            )):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                model = model.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # TODO(lukasz): add as argument
                if data_idx % 100 == 0:
                    msg = '[%d, %5d] loss: %.3f'
                    print(msg % (epoch + 1, data_idx + 1, running_loss / 100))
                    running_loss = 0.
            score_train = self.score(model, train_dataset)
            score_test = self.score(model, test_dataset)
            metrics['score_train'].append(score_train)
            metrics['score_test'].append(score_test)
            msg = '[%d] train score: %.3f, test score: %.3f'
            print(msg % (epoch + 1, score_train, score_test))
            # save model (make sure that Google Colab do not destroy your results)
            if score_test > best_score_test:
                torch.save(model, self.save_experiment)
                best_score_test = score_test
        self.metrics = metrics
        return self

    def score(
            self,
            model: torchvision.models,
            dataset: ImageFoldersDataset
    ) -> float:
        with torch.no_grad():
            # remember that you must call `model.eval()` to set dropout and batch
            # normalization layers to evaluation mode before running the inference.
            model.eval()
            y_true, y_pred = np.zeros(len(dataset)), np.zeros(len(dataset))
            batch_idx = 0
            for data in dataset.loader(batch_size=32):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                model = model.to(self.device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                batch_size = labels.size(0)
                y_true[batch_idx:batch_idx+batch_size] = labels.cpu().numpy()
                y_pred[batch_idx:batch_idx+batch_size] = pred.detach().cpu().numpy()
        return balanced_accuracy_score(y_true, y_pred)