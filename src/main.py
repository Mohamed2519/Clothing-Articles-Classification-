from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from train_utils import train_eval, plot
from clothes_dataset import ClothesDataset
from build_models import get_pretrained_model

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":


  BS=16
  EPOCHS = 10
  data = pd.read_csv('../data/fashion_dataset/filtered_data.csv')#, header = 'infer',error_bad_lines = False)

  classes = data['articleType'].unique().tolist()
  num_classes = len(classes) 
  label_dict = {val: idx for idx, val in enumerate(classes)}

  train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
  train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)



  train_transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomCrop((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  train_dataset = ClothesDataset(train_df,label_dict, transform=train_transform)
  train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

  valid_transform = transforms.Compose([
      transforms.Resize((112,112)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  valid_dataset = ClothesDataset(valid_df,label_dict, transform=valid_transform)
  valid_dataloader = DataLoader(valid_dataset, batch_size=BS, shuffle=False)

  test_transform = transforms.Compose([
      transforms.Resize((112,112)),
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
  model = get_pretrained_model('efficientnet',num_classes)

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
    
  plot(history,'loss')
  plot(history,'accuracy')

