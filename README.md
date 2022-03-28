# Semantic Segmentation
- The goal of semantic segmentation is to classify each pixel of the image in a specific label.
- We will implement this segmentation using U-Net model.

**UNet** is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin. 

## UNet architecture

![image](https://user-images.githubusercontent.com/96312883/160248939-95b54c7e-a0b3-440b-a346-09b0dd451780.png)

A U-shaped architecture consists of a specific encoder-decoder scheme: The encoder reduces the spatial dimensions in every layer and increases the channels. On the other hand, the decoder increases the spatial dims while reducing the channels. The tensor that is passed in the decoder is usually called bottleneck. In the end, the spatial dims are restored to make a prediction for each pixel in the input image. These kinds of models are extremely utilized in real-world applications.

Encoder (left side): It consists of the repeated application of two 3x3 convolutions. Each conv is followed by a ReLU and batch normalization. Then a 2x2 max pooling operation is applied to reduce the spatial dimensions. Again, at each downsampling step, we double the number of feature channels, while we cut in half the spatial dimensions.

Decoder path (right side): Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 transpose convolution, which halves the number of feature channels. We also have a concatenation with the corresponding feature map from the contracting path, and usually a 3x3 convolutional (each followed by a ReLU). At the final layer, a 1x1 convolution is used to map the channels to the desired number of classes.

## Training
In this project, we used a simple UNet seqmentation model. The model architecture goes as following:

![image](https://user-images.githubusercontent.com/96312883/160254105-4225b8f3-90fa-4536-9a21-daf4b539491a.png)


## Dataset
In this project, we used [Cityscapes Image Pairs](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs) from kaggle website. it consists of 2975 train images and 500 validation images.

![image](https://user-images.githubusercontent.com/96312883/160254239-5eaa840d-2a8d-4191-bbe1-639be8a9a34d.png)

![image](https://user-images.githubusercontent.com/96312883/160254243-bb5ee623-bdaa-4a50-bd73-63b395831ff7.png)


## Folder Structure
<pre>
<b>Semantic Segmentation/</b>
    │
    ├── <b>main.py</b>
    │
    ├── <b>utils_func.py</b>
    │
    ├── <b>segmentation_model.py</b>
    │
    ├── <b>model_arch.py</b>
    │
    ├── <b>requirements.txt</b>
    │
    ├── <b>images/</b>
    │       │
    │       ├── train
    │       └── val
    │
    └── <b>output/</b>
            │
            ├── acc.png
            ├── pred_1.png
            ├── model_arch.png
            └── unet_model.hdf5

</pre>
 
## Code Structure
<pre>
<b>main.py</b>
    │
    ├── <b>main()</b>
    │
    ├── <b>data_generator()</b>
    │
    ├── <b>show_predications()</b>
    │
    └── <b>get_acc()</b>

<b>utils_func.py</b>
    │
    ├── <b>load_images()</b>
    │
    ├── <b>bin_image()</b>
    │
    ├── <b>get_segmentation_arr()</b>
    │
    └── <b>give_color_to_seg_img()</b>

<b>segmentation_model.py</b>
    │
    ├── <b>down_block()</b>
    │   
    ├── <b>up_block()</b>
    │   
    ├── <b>bottleneck()</b>
    │   
    └── <b>Unet()</b>

<b>model_arch.py</b>
    │
    └── <b>get_model_arch()</b>

</pre> 

## Applications
- Self-driving cars
- Geosensing
- Agriculture
- Medical image analysis
- Facial segmentation
- Sports
- Fashion

## Conclusion

The output is not very accurate because it needs more time for training than 10 epochs.
