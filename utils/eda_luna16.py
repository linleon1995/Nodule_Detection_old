# Exploring data analysis of LUNA16 dataset
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from dataset.data_utils import load_itk

sys.path.append("..")
from modules.data import dataset_utils
from modules.visualize import vis_utils


def test(lidc_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from skimage.measure import find_contours

    import pylidc as pl
    from pylidc.utils import consensus

    img = np.load(rf'C:\Users\test\Desktop\Leon\Projects\LIDC-IDRI-Preprocessing\data\Image\LIDC-IDRI-0002\0002_NI000_slice020.npy')
    mask = np.load(rf'C:\Users\test\Desktop\Leon\Projects\LIDC-IDRI-Preprocessing\data\Mask\LIDC-IDRI-0002\0002_MA000_slice020.npy')
    plt.imshow(img)
    plt.imshow(mask, alpha=0.2)
    plt.show()

    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id).first()
    vol = scan.to_volume()

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    anns = nods[0]

    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    cmask,cbbox,masks = consensus(anns, clevel=0.5,
                                pad=[(20,20), (20,20), (0,0)])

    # Get the central slice of the computed bounding box.
    k = int(0.5*(cbbox[2].stop - cbbox[2].start))

    # Set up the plot.
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)

    # Plot the annotation contours for the kth slice.
    colors = ['r', 'g', 'b', 'y']
    for j in range(len(masks)):
        for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
            label = "Annotation %d" % (j+1)
            plt.plot(c[:,1], c[:,0], colors[j], label=label)

    # Plot the 50% consensus contour for the kth slice.
    for c in find_contours(cmask[:,:,k].astype(float), 0.5):
        plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')

    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    #plt.savefig("../images/consensus.png", bbox_inches="tight")
    plt.show()


def main():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\seg-lungs-LUNA16\1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\seg-lungs-LUNA16'
    d = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI\LIDC-IDRI-1009\01-01-2000-CT THORAX WCONTRAST-88101\3.000000-Recon 2 CHEST-38044\1-106.dcm'
    d = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI\LIDC-IDRI-0219\01-01-2000-CHEST PA  LATERAL-55820\3186.000000-74714\1-1.dcm'
    image = load_itk(d)
    plt.imshow(image[0][0])
    plt.show()


    for f_idx, f in enumerate(dataset_utils.get_files(filename, ['mhd'])):
        ct_scan, origin, spacing = load_itk(f)
        print(f_idx, f)
        for idx, scan in enumerate(ct_scan):
            if np.sum(scan[scan==1]) > 0:
                plt.imshow(scan)
                plt.show()
    raw_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16'
    mask_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\seg-lungs-LUNA16'
    raw_files = dataset_utils.get_files(raw_path, ['mhd'], recursive=True)
    mask_files = dataset_utils.get_files(mask_path, ['mhd'], recursive=True)

    print(f"Number of CT scans (raw image): {len(raw_files)-len(mask_files)}")
    print(f"Number of CT scans (mask): {len(mask_files)}")

    # Display sample in 2d

    # nodule size


if __name__ == '__main__':
    # main()
    test('LIDC-IDRI-0079')