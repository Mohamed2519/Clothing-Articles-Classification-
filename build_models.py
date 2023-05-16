import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
# import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# # Define device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pretrained_model(model_name,n_classes,train_on_gpu=True):
    print('[INFO]: Loading pre-trained weights')
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
          param.requires_grad = False

        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
        
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
          param.requires_grad = False

        # Get the input dimension of last layer
        kernel_count = model.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        model.classifier = nn.Sequential(nn.Linear(1024, 14),nn.ReLU(),nn.Dropout(0.4),nn.Linear(14, n_classes),)
    
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
          param.requires_grad = False
        model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes)

    if train_on_gpu:
        model = model.to('cuda')

    return model

