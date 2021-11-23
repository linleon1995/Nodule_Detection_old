import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt



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



if __name__ == '__main__':
    # filename = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    # ct_scan, origin, spacing = load_itk(filename)
    # for scan in ct_scan:
    #     plt.imshow(scan, 'gray')
    #     plt.show()
    # print(3)
    pass