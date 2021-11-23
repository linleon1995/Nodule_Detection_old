
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
import cv2
import os


def process_asus_nodules_label(mask_scans, save_name='mask', save_path=None):
    for mask_idx, mask in enumerate(mask_scans):
        if np.sum(mask) > 0:
            
            contour_list = process_label(mask)
            # plt.imshow(mask)
            # plt.show()
            # Save new contour
            
            # plt.imshow(c_list[0], 'gray', alpha=0.4)
            # plt.imshow(c_list[0], 'gray')
            # plt.show()
            

            # Save comparision
            # def get_contours(mask, color):
            #     _, contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     mask_for_show = 255 * np.uint8(np.tile(mask[...,np.newaxis], (1,1,3)))
            #     mask_with_contours = mask_for_show.copy()
            #     cv2.drawContours(mask_with_contours, contours, -1, color, thickness=1)  
            #     return mask_with_contours

            # mask_with_contours = get_contours(mask, color=(255,0,0))
            # new_mask_with_contours = get_contours(new_mask, color=(0,255,0))

            mask_for_show = 255 * np.uint8(np.tile(mask[...,np.newaxis], (1,1,3)))
            z = np.zeros_like(mask_for_show)

            _, contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(z, contours, -1, (255,0,0), thickness=1)
            # print(new_mask, contour_list)
            if len(contour_list) > 0:
                if save_path:
                    for idx, c in enumerate(contour_list):
                        print(mask_idx, idx)
                        fig, ax = plt.subplots(1, 1, figsize=(10,10))
                        ax.imshow(c, 'gray')
                        fig.savefig(os.path.join(save_path, f'{save_name}_{mask_idx:03d}_{idx:03d}.png'))
                        plt.close(fig)
                new_mask = sum(contour_list)
                _, new_contours, new_hierarchy = cv2.findContours(np.uint8(new_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(z, new_contours, -1, (0,255,0), thickness=1) 

            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            ax.imshow(z, 'gray')
            fig.savefig(os.path.join(save_path, f'{save_name}_{mask_idx:03d}_compare.png'))
            plt.close(fig)
            # fig.show()


def process_label(mask):
    # Get original mask information (contour)
    
    _, contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Surpress labeling noise
    kernel = np.ones((7,7), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    denoised = erosion * mask
    denoised[denoised>0] = 1

    # Get new mask information (contour)
    # denoised[:,:,:2] = 0
    c_list = []
    for i in range(len(contours)):
        temp = mask.copy()
        cv2.drawContours(temp, contours, i, (2), thickness=cv2.FILLED)
        # mask_with_c = temp * mask_with_c
        c = 1 - (mask-temp+1)
        intersection = denoised * c
        num_pixel = np.sum(intersection)
        if num_pixel > 0:
            print(num_pixel)
            c_list.append(c)
    
    return c_list