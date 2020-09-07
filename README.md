
# Image Segmentation #


## 1 Introduction

This project implements an image segmentation
architecture based on [This paper by Ronnenberger et. al](https://arxiv.org/pdf/1505.04597.pdf). A set of cat images
and masks are given and **should must be resized to 128 * 128 in the code.** The goals is to improve the
performance of the model on the cat segmentation dataset. The challenge is the size of the dataset which is
quite small. Small amount of training data is a problem in deep learning tasks and in this project, so I work towards practices that try to improve results of models that have access to only small datasets.

## 1.1 Implement U-NET

Read the paper by Ronnenberger and implement the architecture described in the paper to train for the
task of medical image segmentation. I implement two different loss functions, compare the
performance of the two and rationalize why one might work better or worse than the other. (Both should
be reasonable functions for the task of segmentation). Randomly divide the cat images into **(Train - 0.7)**
and **(Test - 0.3)** – **It is VERY IMPORTANT that  keep the train and test data separate and do not
train the model on the test data**. Report the performance of the trained U-NET (trained on train
images) on the Test set images using **Sørensen–Dice-coefficient**. 

**Observations:**
These are examples that MSE Loss wins against Dice loss. When a network is trained with dice loss, it often fails to recognize the cat and make two separate pieces of segment predictions, but except the first row, MSE loss doesn’t seem to have this kind of problem, and even the first row of MSE loss predictions is minor compared to the failure cases in Dice Loss predictions. 
Since my network outputs only 1 class which is just the intensity of each pixel, Euclidean distance seems to be a better choice for this kind of problem. The loss function penalizes when the same position pixels of prediction and ground truth are very different which is very intuitive. As a result, the model learns from it and improves its predictions. 



### 1.2 Data Augmentation
Data augmentation extends the size of your dataset by doing random flips, rotations, zooms, etc. They
can improve performance of the model in exchange for training time and resource allocation. Write a data
augmentation function that performs at least 4 different augmentation techniques on your input images.
Train and test your U-net again on the augmented training set. Report the performance (Dice score). for
images in the test set, include the image, ground truth masks and predicted masks of your model..
**For all train - test instances, keep your test set the same for all steps after your first randomized
split into train and test sets so your comparisons between each steps makes sense**

**Observations:**
By increasing data size, these are obvious improvements between two predictions. In 1.2, the model correctly recognize the cat’s shape and it is almost correct in my opinion. 


### 1.3 Transfer Learning

Transfer learning is one of the most practical ways of dealing with small datasets, given we have access
to some other dataset. First, familiarize yourself with transfer learning.  In this
part, you will find an online dataset of images and corresponding masks of your own choosing. First,
you train your u-net on this new dataset. Then you perform the task of transfer learning to help your
cat segmentation model by using what was learned from the other dataset. Explain how you perform the
transfer learning and include your reasoning in your report for every step you choose to take in this part,
include your code and results of the model trained using transfer learning on your test dataset. Report
the performance. for images in the test set, include the image, ground truth masks and predicted masks
outputted by your model.

**Observations:**
The two examples show great improvement in before and after transfer learning. The data set provides images of vehicle, planes, humans, animals, etc. along with the correct labels of them. I haven’t done much fine tuning for this question because without changing any of my parameters in my model, my model can run perfectly in the dataset and it achieves a dice score of 95% on every epoch. After training on VOC2012 data set, I apply the model to my data set, and it improves performance greatly as I show above. By doing find tuning, people often modify the last layer of the model and output classes in order to fit in the other data set, but VOC2012 data set has a one class label, which is same as my label, so I didn’t bother to change my last layer. As a result, transfer learning works well.

### 1.4 Visualizing segmentation predictions

Here, you have to write code that takes in the test images of the cat pictures, along with their corresponding
binary masks predicted by your trained model and apply contours around the cat. The final result should
look something like the example below. Include your code and 3 samples from your test set. If you have
failed segmentation, include one as one of the 3 images.