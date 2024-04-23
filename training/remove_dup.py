#!/usr/bin/env python3
#
#  Copyright (c) 2018-2020 Carnegie Mellon University
#  All rights reserved.
#
# Based on work by Junjue Wang.
#
#
"""Remove similar frames based on a perceptual hash metric
"""

import os
import glob
import argparse
import shutil
import imagehash
from PIL import Image
import numpy as np


DIFF_THRESHOLD = 1
SRC_DIR_BASENAME = 'cropped_images'


def check_diff(image_hash, base_image_hash, threshold):
    if base_image_hash is None:
        return True
    if image_hash - base_image_hash >= threshold:
        return True

    return False


def check_diff_complete(image_hash, base_image_list, threshold):
    if len(base_image_list) <= 0:
        return True
    for i in base_image_list:
        if not check_diff(image_hash, i, threshold):
            return False
    return True


def main():
    # Scan image folders under 'cropped_images/'
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), SRC_DIR_BASENAME)
    image_folders = [os.path.abspath(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    image_folders.sort()
    class_num = len(image_folders)
    if class_num <= 0:
        return

    base_image_list = []
    dup_count = 0
    total_orig_count = 0

    for img_dir in image_folders:
        img_list = [os.path.abspath(f.path) for f in os.scandir(img_dir) if f.is_file()]
        class_orig_count = len(img_list)
        class_dup_count = 0
        total_orig_count += class_orig_count
        print(img_dir, class_orig_count)
        for img_path in img_list:
            im = Image.open(img_path)
            a = np.asarray(im)
            im = Image.fromarray(a)
            image_hash = imagehash.phash(im)
            if check_diff_complete(image_hash, base_image_list, DIFF_THRESHOLD):
                base_image_list.append(image_hash)
            else:
                dup_count += 1
                class_dup_count += 1
                os.unlink(img_path)
        print("Removed", class_dup_count, "of", class_orig_count)
    print("Total: removed", dup_count, "of", total_orig_count)


if __name__ == "__main__":
    main()
