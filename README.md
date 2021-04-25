# CNN-SVR  
## Overview
CNN-SVR is a deep learning-based method for CRISPR/Cas9 guide RNA (gRNA) on-target cleavage efficacy prediction. It is composed of two major components: a merged CNN as the front-end for extracting gRNA and epigenetic features as well as an SVR as the back-end for regression and predicting gRNA cleavage efficiency. 

## Pre-requisite:  
* **Ubuntu 16.04**
* **Anaconda 3-5.2.0**
* **Python packages:**   
  [numpy](https://numpy.org/) 1.16.4  
  [pandas](https://pandas.pydata.org/) 0.23.0  
  [scikit-learn](https://scikit-learn.org/stable/) 0.19.1  
  [scipy](https://www.scipy.org/) 1.1.0  
 * **[Keras](https://keras.io/) 2.1.0**    
 * **Tensorflow and dependencies:**   
  [Tensorflow](https://tensorflow.google.cn/) 1.4.0    
  CUDA 8.0 (for GPU use)    
  cuDNN 6.0 (for GPU use)    
  
## Installation guide
#### **Operation system**  
Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop  
#### **Python and packages**  
Download Anaconda 3-5.2.0 tarball on https://www.anaconda.com/distribution/#download-section  
#### **Tensorflow installation:**  
pip install tensorflow-gpu==1.4.0 (for GPU use)  
pip install tensorflow==1.4.0 (for CPU use)  
#### **CUDA toolkit 8.0 (for GPU use)**     
Download CUDA tarball on https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run  
#### **cuDNN 6.1.10 (for GPU use)**      
Download cuDNN tarball on https://developer.nvidia.com/cudnn  

## Content
* **data:** the training and testing examples with gRNA sequence and corresponding epigenetic features and label indicating the on-target cleavage efficacy  
* **weights/weights.h5:** the well-trained weights for our model    
* **cnnsvr.py:** the python code, it can be ran to reproduce our results  

## Usage
#### **python cnnsvr.py**       
**Note:**  
* The input training and testing files should include gRNA sequence with length of 23 bp and four "A-N" symbolic corresponding epigenetic features seuqnces with length of 23 as well as label in each gRNA sequence.    
* The train.csv, test.csv can be replaced or modified to include gRNA sequence and four epigenetic features of interest  

## Demo instructions  
#### **Input (gRNA sequence and four epigenetic features):**               
* #### **Data format:**      
**gRNA:** TGAGAAGTCTATGAGCTTCAAGG (23bp)      
**CTCF:** NNNNNNNNNNNNNNNNNNNNNNN     
**Dnase:** AAAAAAAAAAAAAAAAAAAAAAA      
**H3K4me3:** NNNNNNNNNNNNNNNNNNNNNNN      
**RRBS:** NNNNNNNNNNNNNNNNNNNNNNN    
#### **Load weights (Pre-trained weight file):**        
weights/weights.h5   
#### **Run script:**       
python cnnsvr.py   
#### **Output (Predicted activity score for gRNA):** 
0.22743436 
