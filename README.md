#  Histopathologic Cancer Detection 

## Introduction
Metastasis is the spread of cancer cells to new areas of the body, often by way of the lymph
system or bloodstream. Hospitals are full of sophisticated diagnostic equipment that should
help the doctors search for the cancer cells but there is no proper scan or testing available. We
propose an algorithm to identify the metastatic cancer in small image patches taken from larger
digital pathology scans and classify these images to be either cancerous or non-cancerous.
## Description of dataset
The dataset being used for the project is taken from Kaggle. The data for this competition is a
modified version of the PatchCamelyon (PCam) benchmark dataset. The original PCam dataset
contains duplicate images due to its probabilistic sampling, but the dataset being used does not
contain duplicates. It comprises of the two datasets: the training data which has about 220k
rows and the testing data which consists of about 57k rows. The dataset consists of mainly two
columns, the Id of the image and the label associated with it, it can be either 0 or 1 based on
the indication that the centre 32x32px region of a patch contains at least one pixel of tumour
tissue.
## Brief implementation plan
After the analyses of the data set, this seems to be a binary classification problem and it needs to be
diagnosed with a supervised machine learning technique. We have decided to implement the
Support Vector Machine (SVM) model to train the labelled dataset which is already given, fit
the data into a model and then test the testing data by using the same model. An SVM model
is a representation of the examples as points in space, the classified data are separated by a
plane named as Hyperplane which is formed based on certain attributes and they mapped so
that the examples of the separate categories are divided by a clear gap that is as wide as
possible. New examples say testing data are then mapped into that same space and put to
prediction to obtain the analysis done using the model and the data is placed accordingly based
on which side of the gap they fall.

## Update as on December 01
## Libraries
* numpy
* pandas 
* seaborn
* matplotlib
* sklearn
* keras
* itertools

## To run the Repo
1. Clone the repo 
2. Putting the required files in required directories
    * train file in directory: data/train/
    * test file in directory: data/test/
    * **The data files in this repo are a part of samples, might not provide the correct output. Download the actual dataset from the links given ahead.**
3. Run the model
4. check the output file for the submission.csv
5. Based on the result, optimize the model.

## Dataset on github
1. Only a part of dataset has been picked up from the original dataset on kaggle: https://www.kaggle.com/c/histopathologic-cancer-detection/data. The original dataset include :
  * Train data of 2.2m images
  * Test data of about 57k
  ![GitHub Logo](/output/data description.png)
 
 ## Results
 * The result can be seen output --> submission.csv and AUC.png
  * AUC.png provides us with the knowledge of classifier being used is good or bad
  * The AUC score obtained is 0.986
  
 ## The actual code was run on the kaggle kernel for the reason that, we do not have to download the dataset and upload it, and runtime for whole program depends on the Number of epochs in the code. I have used 2 epoches and the total runtime for the code is abouts 5hrs.
 
 
## Author
**Kavya Jain** - d3molisher
