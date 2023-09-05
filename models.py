import os
import torch
import torch.nn.functional as F
import utils
from torchvision.models import resnet50, resnet101, resnet152
import torch.nn as nn

class AudioClassifier(torch.nn.Module):
    def __init__(self,  arch = 'resnet101', num_classes = None, embedding_size = 128):
        super().__init__()
        self.embedding_size = embedding_size
        if arch == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.embedding_head = torch.nn.Linear(2048, self.embedding_size)
            self.backbone[0] = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        elif arch == 'resnet101':
            self.backbone = resnet101(pretrained = True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.embedding_head = torch.nn.Linear(2048, self.embedding_size)
            self.backbone[0] = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        else:
            raise NotImplementedError("")
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        self.classification_head = torch.nn.Linear(self.embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.features(self.embedding_head(x))
        x = self.classification_head(x)
        return x

    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading pretrained from {filename}')
        return utils.torch_load(filename)



class AudioClassifierAM(torch.nn.Module):
    def __init__(self,  arch = 'resnet101', num_classes = None, embedding_size = 128, margin_softmax = None):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        if arch == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.embedding_head = torch.nn.Linear(2048, self.embedding_size)
            self.backbone[0] = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        elif arch == 'resnet101':
            self.backbone = resnet101(pretrained = True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.embedding_head = torch.nn.Linear(2048, self.embedding_size)
            self.backbone[0] = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        else:
            raise NotImplementedError("")
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_classes, self.embedding_size), device = 'cuda'))
        self.margin_softmax = margin_softmax


    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.features(self.embedding_head(x))
        norm_features = F.normalize(x)
        norm_weight = F.normalize(self.weight)
        logits = F.linear(norm_features, norm_weight)
        return logits

    def forward_am(self, x, labels):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.features(self.embedding_head(x))
        norm_features = F.normalize(x)
        norm_weight = F.normalize(self.weight)
        logits = F.linear(norm_features, norm_weight)
        logits = self.margin_softmax(logits, labels)
        return logits


    def forward_backbone(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.features(self.embedding_head(x))
        return x

    def calculate_am_loss(self, x, labels):
        # Input is feature embedding
        loss_obj = torch.nn.CrossEntropyLoss()
        norm_weight = F.normalize(self.weight)
        norm_features = F.normalize(x)
        logits = F.linear(norm_features, norm_weight)
        logits = self.margin_softmax(logits, labels)
        loss = loss_obj(logits, labels)
        return loss


    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading pretrained from {filename}')
        return utils.torch_load(filename)