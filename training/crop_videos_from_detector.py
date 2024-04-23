import argparse
import os

import cv2

from PIL import Image

import tensorflow as tf
import numpy as np


DETECTOR_ONES_SIZE = (1, 480, 640, 3)
THRESHOLD = 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('object_detector_path', type=str)
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    object_detector = tf.saved_model.load(args.object_detector_path)
    ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
    object_detector(ones)

    count = 0
    for label_class in os.listdir(args.input_path):
        video_dir = os.path.join(args.input_path, label_class)
        video_files = os.listdir(video_dir)
        write_dir = os.path.join(args.output_path, label_class)
        os.makedirs(write_dir, exist_ok=True)

        for video_file in video_files:
            video_capture = cv2.VideoCapture(video_file)
            frame = video_capture.read()[1]
            while frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                detections = object_detector(np.expand_dims(frame, 0))
                scores = detections['detection_scores'][0].numpy()
                boxes = detections['detection_boxes'][0].numpy()

                for score, box in zip(scores, boxes):
                    if score < THRESHOLD:
                        continue

                    # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
                    ymin, xmin, ymax, xmax = box

                    # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
                    (left, right, top, bottom) = (
                        xmin * im_width, xmax * im_width,
                        ymin * im_height, ymax * im_height)

                    image = Image.fromarray(frame)
                    cropped = image.crop((left, top, right, bottom))
                    cropped.save(
                        os.path.join(write_dir, '{}.jpg'.format(count)))

                count += 1
                frame = video_capture.read()[1]
                raise Exception


if __name__ == '__main__':
    main()
