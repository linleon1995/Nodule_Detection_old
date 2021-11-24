
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append("..")
from modules.data import dataset_utils
from modules.visualize import vis_utils
from medical_image_process import load_itk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
from pprint import pprint
from eda_utils import process_asus_nodules_label, display_compare


class ASUS_modules_statistics(object):
    def __init__(self, data_path):
        self.dir_list = dataset_utils.get_files(data_path, ['1m00'], recursive=False, get_dirs=True)
        self.num_cases = len(self.dir_list)

        pprint(self.dir_list)
        print(f"Case number: {self.num_cases}")
        
    def basic_information(self):
        vis_utils.plot_histogram()
        

class ASUS_modules_visualize(object):
    def __init__(self, root):
        self.root = root
        # data_dir = dataset_utils.get_files(root, 'raw', recursive=False, get_dirs=True)[0]
        # path = dataset_utils.get_files(data_dir, 'mhd')[0]
        # label_dir = dataset_utils.get_files(root, 'mask', recursive=False, get_dirs=True)[0]
        # label_path = dataset_utils.get_files(label_dir, 'mhd')[0]


        # self.ct_scan, self.origin, self.spacing = load_itk(path)
        # self.masks, self.mask_origin, self.mask_spacing = load_itk(label_path)
        # if label_path is not None:
        #     assert self.ct_scan.shape == self.masks.shape, f'input shape {self.ct_scan.shape} is not matching with masks shape {self.masks.shape}'
        # self.num_slice = self.ct_scan.shape[0]
        
    def show_2d(self, idx=None):
        if idx is None:
            # idx = np.random.randint(0, self.num_slice)
            for idx in range(self.num_slice):
                if np.sum(self.masks[idx]) > 0:
                    break
        else:
            assert idx < self.num_slice and idx > 0

        print(f"Display slice: {idx}")
        self.show_image_and_mask(self.ct_scan[idx], self.masks[idx])
        
    def show_image_and_mask(self, image, mask=None):
        fig, ax = plt.subplots()
        ax.imshow(image, 'gray')
        if mask is not None:
            ax.imshow(mask, alpha=0.2)
        plt.show()
        plt.close(fig)
        pass

    def show_3d(self):
        vis_utils.plot_3d(self.ct_scan[:50])

    def process_mask(self):
        data_dir = dataset_utils.get_files(self.root, '1m00', recursive=False, get_dirs=True)
        for _dir in data_dir:
            mask_dir = dataset_utils.get_files(_dir, 'mask', get_dirs=True)[0]
            mask_path = dataset_utils.get_files(mask_dir, 'mhd')[0]
            ct_scan, _, _ = load_itk(mask_path)
            process_asus_nodules_label(ct_scan, save_name=os.path.split(mask_dir)[1], save_path=rf'C:\Users\test\Desktop\Leon\Weekly\1126\label')

    def batch_compare(self):
        data_dir = dataset_utils.get_files(self.root, '1m00', recursive=False, get_dirs=True)
        for _dir in data_dir:
            img_dir = dataset_utils.get_files(_dir, 'raw', get_dirs=True)
            mask_dir = dataset_utils.get_files(_dir, 'mask', get_dirs=True)
            if len(img_dir) > 0 and len(mask_dir) > 0:
                img_dir = img_dir[0]
                mask_dir = mask_dir[0]
                img_path = dataset_utils.get_files(img_dir, 'mhd')
                mask_path = dataset_utils.get_files(mask_dir, 'mhd')
                if len(img_path) > 0 and len(mask_path) > 0:
                    img_path = img_path[0]
                    mask_path = mask_path[0]
                    ct_scan, _, _ = load_itk(img_path)
                    ct_scan_mask, _, _ = load_itk(mask_path)
                    if ct_scan_mask.shape[0] == ct_scan.shape[0]:
                        display_compare(ct_scan, 
                                        ct_scan_mask, 
                                        save_path=rf'C:\Users\test\Desktop\Leon\Weekly\1126\compare2', \
                                        save_name=os.path.split(mask_dir)[1],
                                        alpha=0.2,)
                    else:
                        print("---UM:", os.path.split(mask_dir)[1])
                    # process_asus_nodules_label(ct_scan, save_name=os.path.split(mask_dir)[1], save_path=rf'C:\Users\test\Desktop\Leon\Weekly\1126\label')


if __name__ == '__main__':
    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign\0017207\0017207mask'
    # files = dataset_utils.get_files(data_path, 'mhd')
    # print(files)

    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0001\1m0001raw mhd\1.2.826.0.1.3680043.2.1125.1.62517356555972253265149798269717921.mhd'
    label_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0001\1m0001mask mhd\1.2.826.0.1.3680043.2.1125.1.37415644189337809125763467895792497.mhd'

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0012\1m0012raw mhd\1.2.826.0.1.3680043.2.1125.1.62098011105456587698466395992736761.mhd'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0012\1m0012mask mhd\1.2.826.0.1.3680043.2.1125.1.37055484888799566174135558418616650.mhd'
    label_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0012\1m0012mask mhd\1.2.826.0.1.3680043.2.1125.1.37055484888799566174135558418616650.mhd'

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant\1m0010'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    data_vis = ASUS_modules_visualize(path)
    data_vis.batch_compare()
    # data_vis.process_mask()

    # data_stats = ASUS_modules_statistics(data_path)