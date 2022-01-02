import os
import pandas as pd
import torchio as tio
from pathlib import Path


def convert(demo,scan_types):
    data_dir   = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
    out_dir    = './processed'

    for dataset in ['train']:
        dataset_dir = f'{data_dir}{dataset}'
        patients = os.listdir(dataset_dir)
        if demo:
            patients = patients[:10]
        
        # Remove cases the competion host said to exclude 
        # https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/262046
        if '00109' in patients: patients.remove('00109')
        if '00123' in patients: patients.remove('00123')
        if '00709' in patients: patients.remove('00709')
        
        print(f'Total patients in {dataset} dataset: {len(patients)}')

        count = 0
        for patient in patients:
            count = count + 1
            print(f'{dataset}: {count}/{len(patients)}')

            for scan_type in scan_types:
                scan_src  = f'{dataset_dir}/{patient}/{scan_type}/'
                scan_dest = f'{out_dir}/{dataset}/{patient}/{scan_type}/'
                Path(scan_dest).mkdir(parents=True, exist_ok=True)
                image = tio.ScalarImage(scan_src)
                transforms = [
                    tio.ToCanonical(),
                    tio.Resample(1),
                    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                    tio.CropOrPad((128,128,64)),
                    tio.RescaleIntensity((-1, 1)),
                ]
                transform = tio.Compose(transforms)
                preprocessed = transform(image)
                preprocessed.save(f'{scan_dest}/{scan_type}.nii.gz')