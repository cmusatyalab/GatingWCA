import math
import os.path

import cv2

import numpy as np
import csv


def detect_objects(frame):
    # Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("mask", mask)
    objects_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            objects_contours.append(cnt)

    return objects_contours


def size_measuring(xmi_, ymi_, xma_, yma_, frame):
    img = frame
    xmi = math.ceil(float(xmi_))
    ymi = math.ceil(float(ymi_))
    xma = math.ceil(float(xma_))
    yma = math.ceil(float(yma_))
    print("box: x[{}, {}] y[{}, {}]".format(xmi, xma, ymi, yma))

    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if len(corners) == 0:
        print("WARNING: Show full aruco marker please!!!!!!!!")
        return img, -1, -100

    int_corners = np.int0(corners[0])
    # reshape the bounding of aruco marker
    four_corners = int_corners.reshape((4, 2))

    minx = min(four_corners[:, 0])
    maxx = max(four_corners[:, 0])
    miny = min(four_corners[:, 1])
    maxy = max(four_corners[:, 1])
    print("aruco: x[{}, {}] y[{}, {}]".format(minx, maxx, miny, maxy))
    x_overlap = (minx >= xmi and minx <= xma) or (maxx >= xmi and maxx <= xma)
    y_overlap = (miny >= ymi and miny <= yma) or (maxy >= ymi and maxy <= yma)
    if x_overlap and y_overlap:
        print("WARNING: False positive!!!")
        return img, -2, -100

    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 20

    # crop the part in bounding box
    cropped_image = img[ymi:yma, xmi:xma]

    contours = detect_objects(cropped_image)
    a = cv2.drawContours(cropped_image, [c for c in contours], -1, (0, 0, 255), 2)

    # Draw objects boundaries
    x_midle = img.shape[0] / 2
    y_midle = img.shape[1] / 2
    size_ob = -100
    for cnt in contours:
        # print(cnt.shape)
        # Get rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio

        # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.circle(cropped_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        size_ob = max(object_height, object_width)
        cv2.putText(img, "Height {} cm".format(round(size_ob, 1)), (int(x_midle - 50), int(y_midle + 50)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        # print(object_height, "height")
        # print(object_width, "width")

    return img, 1, size_ob


def main():
    size_measuring(0, 0, 0, 0, None)


if __name__ == '__main__':
    main()
