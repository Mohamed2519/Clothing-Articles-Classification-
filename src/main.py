from timeit import default_timer as timer
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from train_utils import train_eval, plot
from clothes_dataset import ClothesDataset
from build_models import get_pretrained_model

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def parsing():
  # Create argument parser
  parser = argparse.ArgumentParser(description='Training Models')

  # Add arguments
  parser.add_argument('--model', type=str, default='efficientnet', help='[efficientnet,vgg16, densenet]')
  parser.add_argument('--e', type=int, default=10, help='number of epochs')
  parser.add_argument('--bs', type=int, default=16, help='number of batch size')
  parser.add_argument('--n_cls', type=int, default=11, help='number of classes')
  parser.add_argument('--imgsz', type=int, default=16, help='image size')
  parser.add_argument('--vis', type=bool, default=False, help='plot training/ val graphs')


 # Parse the arguments
  args = parser.parse_args()
  return args


if __name__ == "__main__":

  args = parsing()
  BS = args.bs
  EPOCHS = args.e
  train_df = pd.read_csv('../data/fashion_dataset/train_data.csv')
  valid_df = pd.read_csv('../data/fashion_dataset/valid_data.csv')
  test_df = pd.read_csv('../data/fashion_dataset/test_data.csv')

  classes = train_df['articleType'].unique().tolist()
  num_classes = args.n_cls
  label_dict = {val: idx for idx, val in enumerate(classes)}

  

  train_transform = transforms.Compose([
        transforms.Resize((args.imgsz,args.imgsz)),
        transforms.RandomCrop((args.imgsz,args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  train_dataset = ClothesDataset(train_df,label_dict, transform=train_transform)
  train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

  valid_transform = transforms.Compose([
      transforms.Resize((args.imgsz,args.imgsz)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  valid_dataset = ClothesDataset(valid_df,label_dict, transform=valid_transform)
  valid_dataloader = DataLoader(valid_dataset, batch_size=BS, shuffle=False)

  test_transform = transforms.Compose([
      transforms.Resize((args.imgsz,args.imgsz)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  test_dataset = ClothesDataset(test_df,label_dict, transform=test_transform)
  test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

  # Sanity check for dataloaders 
  trainiter = iter(train_dataloader)
  features, labels = next(trainiter)
  print ('shape of train_dataloader for image',features.shape)
  print ('shape of train_dataloader for label',labels.shape)
  print('---------------------------------------')
  trainiter = iter(valid_dataloader)
  features, labels = next(trainiter)
  print ('shape of valid_dataloader for image',features.shape)
  print ('shape of valid_dataloader for label',labels.shape)
  print('---------------------------------------')
  trainiter = iter(test_dataloader)
  features, labels = next(trainiter)
  print ('shape of test_dataloader for image',features.shape)
  print ('shape of test_dataloader for label',labels.shape)
  print('---------------------------------------')

  # Build bodel
  model = get_pretrained_model(args.model,num_classes)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  
  #Start trainig
  model, history = train_eval(
      model,
      criterion,
      optimizer,
      train_dataloader,
      valid_dataloader,
      save_file_name='../models/model.pt',
      max_epochs_stop=5,
      n_epochs=EPOCHS,
      print_every=1)
  
  if args.vis:
    plot(history,'loss')
    plot(history,'accuracy')

