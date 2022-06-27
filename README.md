# Using PyTorch to classify images of Spiral Galaxies based on their arm count.


This python code was developed to extract images of a spiral galaxies from the Galaxy Zoo online repoisitory and process the images using various algorithms such as gamma correction, 
median filtering, adaptive thresholding and skeletonization.

Three CNN's were developed based on previous work in the field of Deep Learning in Astrophysics to classify each galaxy's arm count based on the raw RGB images and also its Skeletonized image to 
investigate the variation in classifcation results after image processing was implemented

## Data
Images in 424 x 424 x 3 .PNG format downloaded from Galaxy Zoo 2 online repository: https://zenodo.org/record/4573248#.YrmDdRVBw2w


## Description

### Section 1: Data Preprocessing
- Feature engineering
- Data Cleaning
- Quality Assurance


### Section 2: Image Processing and Data Augmentation
- Skeletonization function applies gamma correction, median filtering, and adaptive thresholding to define better feature edges and remove noise before skeletonizing binarized image
- Images are randomly flipped, rotated and cropped
- Datasets loaded into Image Loaders for training in Pytorch

### Section 3: Model Training
- Three CNN's developed with varying widths and depths to test images
- Model 3: Adapted Dieleman saw best results for Skeletonization due to decreased kernel size in first convolutional layer
- Change model at Line 553: `model = Baseline(n, drop = 0.5).to(device)`
    - n parameter tells model if input is skeletonized (n=1), rgb (n=3), or combined (n=4)
    - drop denotes dropout %, use 0.5   


