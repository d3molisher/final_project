# Histopath Histopathologic Cancer Detection logic

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
