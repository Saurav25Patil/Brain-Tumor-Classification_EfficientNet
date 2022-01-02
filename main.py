# importing libraries
#!pip install --quiet --no-index --find-links ../input/pip-download-torchio/ --requirement ../input/pip-download-torchio/requirements.txt
#!pip install --quiet torchio
import os
import csv
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import torchio as tio
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from dicom_to_nifti import convert
from split_dataset import split_patients
from train import train_model
from predict import prediction

#if True limits to 10 patients

demo = True
scan_types=['FLAIR','T1w','T1wCE','T2w']

def main():
    #Convert DICOM images to nifti
    convert(demo=demo,scan_types=scan_types)
    
    #Split the dataset using Nibabel
    split_patients()

    #Train the model
    train_model()

    #prediction
    prediction()

if __name__=="__main__":
    main()

