## preprocessingData
#1. Read .csv file of data
#2. Remove minor Classes
#3. add column for images directory  

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

#reading the styles.csv file
df = pd.read_csv('/content/fashion-product-images-small/styles.csv', header = 'infer',error_bad_lines = False)
df = df[['id','articleType']] 
vc=df['articleType'].value_counts()
#Choose only classes that have more than 900 image to avoid data unbalanced
indx= vc.index
valu = vc.values
for i in range(len(vc)):

    if valu[i] <900:
        break
    else:
      cloth_used = indx[:i]
      
print('number of clothes used: ',len(cloth_used))

#Filter dataframe with choosen clothes

df = df[df['articleType'].isin(cloth_used)]

#Add column for image directory
base_path='/content/fashion-product-images-small/images/'
df['path']=df['id'].apply(lambda x: str(base_path + str(x)+ '.jpg'))
df=df.sample(frac=1).reset_index(drop=True)
# saved filtred Dataframe
df.to_csv('/content/drive/MyDrive/fashion_dataset/filtered_data.csv', index=False)

############################For Visualizing images############################

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
