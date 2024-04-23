#!/usr/bin/env python3
"""Resize cropped images to half to generate data that fits the resolution for the Google Glass
"""

import os
import cv2

SRC_DIR_BASENAME = 'cropped_images'


def main():
    # Scan image folders under 'cropped_images/'
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), SRC_DIR_BASENAME)
    image_folders = [os.path.abspath(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    image_folders.sort()
    class_num = len(image_folders)
    if class_num <= 0:
        return

    for img_dir in image_folders:
        img_list = [os.path.abspath(f.path) for f in os.scandir(img_dir) if f.is_file()]
        print("Resizing {} images in {}".format(len(img_list), img_dir))
        for img_path in img_list:
            im = cv2.imread(img_path)
            im_resized = cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[1] / 2)))
            im_out_path = img_path + "-2.jpg"
            cv2.imwrite(im_out_path, im_resized)
            im_resized = cv2.resize(im, (int(im.shape[1] / 3), int(im.shape[1] / 3)))
            im_out_path = img_path + "-3.jpg"
            cv2.imwrite(im_out_path, im_resized)
            # Optionally, remove the originally-sized images to reduce the sample size
            os.remove(img_path)


if __name__ == "__main__":
    main()
