import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# TODO: to a single object

def process_hu_value(img, low_cut_off=None, high_cut_off=None):
    if low_cut_off is not None:
        img = np.where(img<low_cut_off, low_cut_off, img)

    if high_cut_off is not None:
        img = np.where(img>high_cut_off, high_cut_off, img)
    return img


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def sample_slice_from_ct(path):
    ct_scan, _, _ = load_itk(path)
    slice_idx = np.random.choice(np.arange(ct_scan.shape[0]))
    return ct_scan[slice_idx]


def convert_npy_to_pil(path):
    array = np.load(path)
    image = Image.fromarray(array)
    return image