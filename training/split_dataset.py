import os
import shutil
import argparse
import random

parser = argparse.ArgumentParser(description='Split images into training set and validation set')
parser.add_argument('input_dir', metavar='<input_dir>',
                    help='path to input image dataset before splitting')
parser.add_argument('train_dir', metavar='<train_dir>',
                    help='path to output training set directory')
parser.add_argument('val_dir', metavar='<val_dir>',
                    help='path to output validation set directory')
parser.add_argument('test_dir', metavar='<test_dir>',
                    help='path to output test set directory')
parser.add_argument('--val_ratio', default=0.1, type=float, metavar='<ratio>',
                    help='split ratio of the validation set, default to 0.1')
parser.add_argument('--test_ratio', default=0.1, type=float, metavar='<ratio>',
                    help='split ratio of the test set, default to 0.1')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='randomly select images when splitting')


def main():
    args = parser.parse_args()
    # Scan image folders for each class under the input directory
    input_dir = args.input_dir
    class_folders = [os.path.abspath(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    class_folders.sort()
    if len(class_folders) <= 0:
        return
    assert args.val_ratio + args.test_ratio < 1.0

    # Create output directories if not exist
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    # Copy images to training and validation set directories
    for class_dir in class_folders:
        train_class_dir = os.path.join(args.train_dir, os.path.basename(class_dir))
        val_class_dir = os.path.join(args.val_dir, os.path.basename(class_dir))
        test_class_dir = os.path.join(args.test_dir, os.path.basename(class_dir))
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        class_images = [os.path.abspath(f.path) for f in os.scandir(class_dir) if f.is_file()]
        class_images.sort()
        if args.shuffle:
            random.shuffle(class_images)
        num_val_images = int(len(class_images) * args.val_ratio)
        num_test_images = int(len(class_images) * args.test_ratio)
        num_train_images = len(class_images) - num_val_images - num_test_images
        train_images = class_images[:num_train_images]
        val_images = class_images[num_train_images:num_train_images + num_val_images]
        test_images = class_images[num_train_images + num_val_images:]
        for img in train_images:
            shutil.copy(img, train_class_dir)
        for img in val_images:
            shutil.copy(img, val_class_dir)
        for img in test_images:
            shutil.copy(img, test_class_dir)


if __name__ == '__main__':
    main()
