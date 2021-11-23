
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from modules.data import dataset_utils
from medical_image_process import load_itk


class ASUS_modules_statistics(object):
    def __init__(self):
        pass

    def __call__(self, filename):
        self.ct_scan, self.origin, self.spacing = load_itk(filename)


class ASUS_modules_visualize(object):
    def __init__(self, path, label_path=None):
        self.ct_scan, self.origin, self.spacing = load_itk(path)
        self.masks, self.mask_origin, self.mask_spacing = load_itk(label_path)
        if label_path is not None:
            assert self.ct_scan.shape == self.masks.shape
        self.z = self.ct_scan.shape[0]
        
    def show_2d(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.z)
        else:
            assert idx < self.z and idx > 0

        for idx in range(self.z):
            if np.sum(self.masks[idx]) > 0:
                break


        print(f"Display slice: {idx}")
        self.show_image_and_mask(self.ct_scan[idx], self.masks[idx])
        
    def show_image_and_mask(self, image, mask=None):
        fig, ax = plt.subplots()
        ax.imshow(image, 'gray')
        if mask is not None:
            ax.imshow(mask, alpha=0.2)
        # plt.imshow()
        plt.show()
        # plt.close(fig)
        pass

    def show_3d(self):
        pass


if __name__ == '__main__':
    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign\0017207\0017207mask'
    # files = dataset_utils.get_files(data_path, 'mhd')
    # print(files)

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign\0017207\0017207raw mhd\1.2.826.0.1.3680043.2.1125.1.90165150408318304766748088748668071.mhd'
    label_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign\0017207\0017207mask\1.2.826.0.1.3680043.2.1125.1.3379098776305243303064809158540599.mhd'
    data_vis = ASUS_modules_visualize(path, label_path)
    data_vis.show_2d()

    # data_vis.show_2d(show=True, save=False)
    # data_vis.sow_3d(show=True, save=True)
