import os
import csv
import nibabel as nib
import torchio as tio
import tensorflow as tf
from pathlib import Path

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

def prediction():
    data_dir   = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
    test_dir   = f'{data_dir}test'
    patients = os.listdir(test_dir)
    print(f'Total patients: {len(patients)}\n\n')

    out_dir  = '/kaggle/working/processed'

    scan_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    scan_types = ['T1wCE']

    for scan_type in scan_types:
        f = open(f'/kaggle/working/submission.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(['BraTS21ID','MGMT_value'])
        for patient in patients:
            # dicom to nifiti
            scan_src  = f'{test_dir}/{patient}/{scan_type}/'
            scan_dest = f'{out_dir}/test/{patient}/{scan_type}/'
            Path(scan_dest).mkdir(parents=True, exist_ok=True)
            image = tio.ScalarImage(scan_src)  # subclass of Image
            transforms = [
                tio.ToCanonical(),
                tio.Resample(1),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.CropOrPad((128,128,64)),
                tio.RescaleIntensity((-1, 1)),
            ]
            transform = tio.Compose(transforms)
            preprocessed = transform(image)
            filepath = f'{scan_dest}/{scan_type}.nii.gz'
            preprocessed.save(filepath)
            
            # process_scan
            case = process_scan(filepath)

            # tf model
            model = tf.keras.models.load_model(f'./models/{scan_type}/')

            # get prediction
            prediction = model.predict(case)
            
            # write prediction
            print(f'{patient},{prediction[0][0]}')
            writer.writerow([patient, prediction[0][0]])

        f.close()