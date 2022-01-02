import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from pathlib import Path

data_dir  = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
out_dir   = '/kaggle/working/processed'

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def add_batch_channel(volume):
    """Process validation data by adding a channel."""
    volume = tf.expand_dims(volume, axis=-1)
    volume = tf.expand_dims(volume, axis=0)
    return volume

def process_scan(filepath):
    scan = read_nifti_file(filepath)
    volume = add_batch_channel(scan)
    return volume

def split_patients():
    labels_df = pd.read_csv(data_dir+'train_labels.csv', index_col=0)

    # split patients
    patients = os.listdir(f'{out_dir}/train')
    from sklearn.model_selection import train_test_split
    train, validation = train_test_split(patients, test_size=0.3, random_state=42)
    print(f'{len(patients)} total patients.\n   {len(train)} in the train split.\n   {len(validation)} in the validation split')

    scan_types  = ['FLAIR','T1w','T1wCE','T2w']
    splits_dict = {'train':train, 'validation':validation}

    for scan_type in scan_types:
        print(f'{scan_type} start')
        for split_name, split_list in splits_dict.items():
            print(f'   {split_name} start')
            label_list = []
            filepaths = []
            for patient in split_list:
                label = labels_df._get_value(int(patient), 'MGMT_value')
                label = add_batch_channel(label)
                label_list.append(label)
                filepath  = f'{out_dir}/train/{patient}/{scan_type}/{scan_type}.nii.gz'
                filepaths.append(filepath)

            features = np.array([process_scan(filepath) for filepath in filepaths if filepath])
            labels = np.array(label_list, dtype=np.uint8)
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            
            # save dataset   
            tf_data_path = f'./datasets/{scan_type}_{split_name}_dataset'
            tf.data.experimental.save(dataset, tf_data_path, compression='GZIP')
            with open(tf_data_path + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
                pickle.dump(dataset.element_spec, out_)
            print(f'   {split_name} done')
        print(f'{scan_type} done')