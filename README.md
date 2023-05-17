# Clothing Articles Classification 
Clothes classification using various models. 

In this Repo, I Use Famous deepleaning architectures for image classification such as  `VGG16`, `Densenet121`,and `Efficientnetb0`.
 Training on 
 [Fashion Product Images (Small) Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)


Calculate FLOPS and MACCs for each model using `flopth` library
## Models
**`VGG16`** is a popular deep neural network architecture for image classification. It consists of 16 convolutional layers. It is known for its depth, simplicity, and strong performance in image classification tasks.
![picture](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

------------------------------------


**`DenseNet-121`** consists of multiple dense blocks, where each layer is connected to every other layer within the same block. This dense connectivity promotes feature reuse and helps in combating the vanishing gradient problem. DenseNet-121 also incorporates bottleneck layers and transition layers to reduce the number of parameters and control the spatial dimensions of the feature maps. It has been widely used for image classification tasks and has shown strong performance. DenseNet-121 is known for its efficient memory utilization and ability to capture intricate feature representations in deep networks.
![picture](https://miro.medium.com/v2/resize:fit:678/1*u4hyohOF9SIRRLBAzqYXfQ.jpeg)

------------------------------------



**`EfficientNet-B0`** is a compact and efficient convolutional neural network architecture that achieves a good balance between model size and performance. It is known for its efficiency in terms of parameter size and computational cost while still delivering competitive accuracy.

![picture](https://wisdomml.in/wp-content/uploads/2023/03/eff_banner.png)



## Steps
1. **Download the dataset in `data` folder**

2. **Install Requirements**
```shell
pip install -r requirements.txt
```
3. **Data Preprocessing**
```shell

python src/data_preprocessing.py --imgs_per_cls 900 --vis False
```

  * Read `.csv` file of the dataset 
  * filter dataset from minor classes (less than 900 images) to avoid data imbalance.
  * after filteration it becomes 11 classes
  * add column for images directory


4. **Training** 
```shell
python src/main.py --model efficientnet --bs 16 --e 10 --imgsz 112
```
  * `clothes_dataset.py` Building DataLoader
  * Image size (112,112,3)
  * Data Augumentation (`RandomCrop`,`RandomHorizontalFlip`)
  * `build_models.py` for building models  
  * `train_utils.py` contain the main function for `Train` & `Eval`


  **Crateria of training**:
  For choosing the best and proper model, I choose `Accuracy` as a metric of Evaluation.
  Train each network with configurations:

    `Batch Size` = 16

    `Epochs` = 10

    `Input Image Size` = (112, 112, 3)

    `optimizer` SGD



5. **Model Capacity**

  Using `flopth` library to calculate  `FLOPS` and `number of Parameters`

```shell
python src/model_capacity.py --model efficientnet --n_cls 11 --imgsz 112
```


6. **Results**


 _  | VGG16 | EfficientNet | DenseNet
--- | --- | --- | ---
Accuracy | **86.88%**| 84.28%|83.12%
FLOPS | 3.9 G |**0.1 G** |0.7 G
Param. |135.3 M | **4.0 M** | 6.9 M

  From previous table, `VGG16` has higher accuracy among models. 

  However, you can use `EfficientNet` with less parameters and FLOPS copmared to other models. with little decreasing in accuracy.

  Crateria of Choosing model depends on the your application. There's a trade off between `speed` and `accuracy`

  In my Case study, I'll choose `EfficientNet`, It achieves a good balance between model size and performance


## **Future Works**
* Choose Large Dataset 
* Train more EPOCHS to get best accuracy
* Apply different data augmentatoin techniques
* Use different version of EfficientNet such as `EfficientNet-B7`

## **Report** 

This [Report](https://docs.google.com/document/d/1DATIFjaQaOeX_naR5RgLoP5MnK2O1q8JFeYpzwgVlts/edit?usp=sharing) discuss `Receptive field` in genneral and specialy for `EfficientNet-B7`

Also, discuss `FLOPs`, `FLOPS`, and `MACCs` in definition, calculation, How to reduce them.
