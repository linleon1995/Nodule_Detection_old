# Exploring data analysis of LUNA16 dataset
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from medical_image_process import load_itk

sys.path.append("..")
from modules.data import dataset_utils
from modules.visualize import vis_utils


def main():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    ct_scan, origin, spacing = load_itk(filename)

    raw_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16'
    mask_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\seg-lungs-LUNA16'
    raw_files = dataset_utils.get_files(raw_path, ['mhd'], recursive=True)
    mask_files = dataset_utils.get_files(mask_path, ['mhd'], recursive=True)

    print(f"Number of CT scans (raw image): {len(raw_files)-len(mask_files)}")
    print(f"Number of CT scans (mask): {len(mask_files)}")

    # Display sample in 2d

    # nodule size
    
    print(3)

if __name__ == '__main__':
    main()