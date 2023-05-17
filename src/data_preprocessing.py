## preprocessingData
#1. Read .csv file of data
#2. Remove minor Classes
#3. add column for images directory  

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def parsing():
  # Create argument parser
  parser = argparse.ArgumentParser(description='Data Preprocessing')

  # Add arguments
  parser.add_argument('--csv_dir', type=str, default='../data/fashion-product-images-small/styles.csv', help='dir of csv data file')
  parser.add_argument('--imgs_per_cls', type=int, default=900, help='min images per class')
  parser.add_argument('--vis', type=bool, default=False, help='visualize randm 10 samples of the data')

 # Parse the arguments
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parsing()
  
  #reading the styles.csv file
  df = pd.read_csv(args.csv_dir, header = 'infer',error_bad_lines = False)
  df = df[['id','articleType']] 
  vc=df['articleType'].value_counts()

  #Choose only classes that have more than 900 image to avoid data unbalanced
  indx= vc.index
  valu = vc.values
  for i in range(len(vc)):

      if valu[i] < args.imgs_per_cls:
          break
      else:
        cloth_used = indx[:i]
        
  print('after filtering {} image per class,number of clothes used: {} '.format(args.imgs_per_cls,len(cloth_used)))

  #Filter dataframe with choosen clothes

  df = df[df['articleType'].isin(cloth_used)]

  #Add column for image directory
  base_path='../data/fashion-product-images-small/images/'
  df['path']=df['id'].apply(lambda x: str(base_path + str(x)+ '.jpg'))
  df=df.sample(frac=1).reset_index(drop=True)

  # Splitting data
  x = df['path']
  y = df['articleType']
  train_df, test_df = train_test_split(df, test_size=0.2,stratify = df['articleType'] , random_state=42)
  train_df, valid_df = train_test_split(train_df, test_size=0.2,stratify = train_df['articleType'], random_state=42)

  # saved filtred Dataframe
  df.to_csv('../data/fashion_dataset/filtered_data.csv', index=False)
  train_df.to_csv('../data/fashion_dataset/train_data.csv', index=False)
  valid_df.to_csv('../data/fashion_dataset/valid_data.csv', index=False)
  test_df.to_csv('../data/fashion_dataset/test_data.csv', index=False)


  ############################ For Visualizing images ############################
  if args.vis:
    # Choose 10 random images
    indices = random.sample(range(len(df)), 10)

    # Plot images
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i, ax in zip(indices, axes.flat):
        img_path = df.iloc[i]['path']
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(df.iloc[i]['articleType'])
        
    plt.show()
