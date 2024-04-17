import math
import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def dist(p, q):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))


def dot(p, q):
    return sum(px * qx for px, qx in zip(p, q))


def get_hand_state(landmarks, shape):
    """
    Returns a dictionary of current hand state, including hand orientation, finger open/closed,
    thumb orientation, and finger y-ordering
    :param shape: Shape of the frame: (height, width, _)
    :param landmarks: List of 21 hand landmarks
    :return: Dictionary of current hand state
    """
    hand_state = {}
    x0 = landmarks[WRIST].x * shape[1]
    y0 = landmarks[WRIST].y * shape[0]
    x9 = landmarks[MIDDLE_FINGER_MCP].x * shape[1]
    y9 = landmarks[MIDDLE_FINGER_MCP].y * shape[0]

    # Set Landmark 0 as origin with x-axis pointing horizontally towards the right,
    # returns the angle between the x-axis and the ine joining Landmark 0 and Landmark 9,
    # ranges from -180 to 180 degrees.
    hand_state["orientation"] = math.degrees(math.atan2(y0 - y9, x9 - x0))

    x1 = landmarks[THUMB_CMC].x * shape[1]
    y1 = landmarks[THUMB_CMC].y * shape[0]
    x2 = landmarks[THUMB_MCP].x * shape[1]
    y2 = landmarks[THUMB_MCP].y * shape[0]
    x3 = landmarks[THUMB_IP].x * shape[1]
    y3 = landmarks[THUMB_IP].y * shape[0]
    x4 = landmarks[THUMB_TIP].x * shape[1]
    y4 = landmarks[THUMB_TIP].y * shape[0]
    d01 = dist([x0, y0], [x1, y1])
    d02 = dist([x0, y0], [x2, y2])
    d03 = dist([x0, y0], [x3, y3])
    d04 = dist([x0, y0], [x4, y4])
    hand_state["thumb_open"] = d04 > d03 and d03 > d02 and d02 > d01

    x5 = landmarks[INDEX_FINGER_MCP].x * shape[1]
    y5 = landmarks[INDEX_FINGER_MCP].y * shape[0]
    v04 = (x4 - x0, y4 - y0)
    v05 = (x5 - x0, y5 - y0)
    v09 = (x9 - x0, y9 - y0)
    hand_state["thumb_index_angle"] = math.degrees(
        math.acos(dot(v04, v05) / math.sqrt(dot(v04, v04)) / math.sqrt(dot(v05, v05))))
    hand_state["thumb_middle_angle"] = math.degrees(
        math.acos(dot(v04, v09) / math.sqrt(dot(v04, v04)) / math.sqrt(dot(v09, v09))))

    x6 = landmarks[INDEX_FINGER_PIP].x * shape[1]
    y6 = landmarks[INDEX_FINGER_PIP].y * shape[0]
    x7 = landmarks[INDEX_FINGER_DIP].x * shape[1]
    y7 = landmarks[INDEX_FINGER_DIP].y * shape[0]
    x8 = landmarks[INDEX_FINGER_TIP].x * shape[1]
    y8 = landmarks[INDEX_FINGER_TIP].y * shape[0]
    d06 = dist([x0, y0], [x6, y6])
    d07 = dist([x0, y0], [x7, y7])
    d08 = dist([x0, y0], [x8, y8])
    hand_state["index_finger_closed"] = d06 > d07 and d07 > d08

    x10 = landmarks[MIDDLE_FINGER_PIP].x * shape[1]
    y10 = landmarks[MIDDLE_FINGER_PIP].y * shape[0]
    x11 = landmarks[MIDDLE_FINGER_DIP].x * shape[1]
    y11 = landmarks[MIDDLE_FINGER_DIP].y * shape[0]
    x12 = landmarks[MIDDLE_FINGER_TIP].x * shape[1]
    y12 = landmarks[MIDDLE_FINGER_TIP].y * shape[0]
    d010 = dist([x0, y0], [x10, y10])
    d011 = dist([x0, y0], [x11, y11])
    d012 = dist([x0, y0], [x12, y12])
    hand_state["middle_finger_closed"] = d010 > d011 and d011 > d012

    x14 = landmarks[RING_FINGER_PIP].x * shape[1]
    y14 = landmarks[RING_FINGER_PIP].y * shape[0]
    x15 = landmarks[RING_FINGER_DIP].x * shape[1]
    y15 = landmarks[RING_FINGER_DIP].y * shape[0]
    x16 = landmarks[RING_FINGER_TIP].x * shape[1]
    y16 = landmarks[RING_FINGER_TIP].y * shape[0]
    d014 = dist([x0, y0], [x14, y14])
    d015 = dist([x0, y0], [x15, y15])
    d016 = dist([x0, y0], [x16, y16])
    hand_state["ring_finger_closed"] = d014 > d015 and d015 > d016

    x18 = landmarks[PINKY_PIP].x * shape[1]
    y18 = landmarks[PINKY_PIP].y * shape[0]
    x19 = landmarks[PINKY_DIP].x * shape[1]
    y19 = landmarks[PINKY_DIP].y * shape[0]
    x20 = landmarks[PINKY_TIP].x * shape[1]
    y20 = landmarks[PINKY_TIP].y * shape[0]
    d018 = dist([x0, y0], [x18, y18])
    d019 = dist([x0, y0], [x19, y19])
    d020 = dist([x0, y0], [x20, y20])
    hand_state["pinky_closed"] = d018 > d019 and d019 > d020

    landmark_y_order = np.argsort([pos.y for pos in landmarks])
    if landmark_y_order[0] == THUMB_TIP and landmark_y_order[1] == THUMB_IP:
        hand_state["thumb_orientation"] = "up"
    elif landmark_y_order[-1] == THUMB_TIP and landmark_y_order[-2] == THUMB_IP:
        hand_state["thumb_orientation"] = "down"
    else:
        hand_state["thumb_orientation"] = "unknown"

    if y4 < y6 and y6 < y10 and y10 < y14 and y14 < y18:
        hand_state["finger_y_order"] = "up"
    elif y4 > y6 and y6 > y10 and y10 > y14 and y14 > y18:
        hand_state["finger_y_order"] = "down"
    else:
        hand_state["finger_y_order"] = "unknown"

    return hand_state


def get_thumb_state(hand_landmark, shape):
    hand_state = get_hand_state(hand_landmark, shape)
    # print(hand_state)
    thumb_state = ""

    # Check if thumb open and other fingers closed
    if hand_state["thumb_open"] and hand_state["index_finger_closed"] and \
            hand_state["middle_finger_closed"] and hand_state["ring_finger_closed"] and \
            hand_state["pinky_closed"]:
        # Check if the thumb-index-angle is appropriate
        if hand_state["thumb_middle_angle"] <= 90 and \
                hand_state["thumb_middle_angle"] > hand_state["thumb_index_angle"] and \
                hand_state["thumb_index_angle"] > 15:
            # Check if the hand orientation is within range
            if hand_state["orientation"] > 120 or hand_state["orientation"] < -150 or \
                    (hand_state["orientation"] < 60 and hand_state["orientation"] > -30):
                if hand_state["thumb_orientation"] == "up" and hand_state["finger_y_order"] == "up":
                    thumb_state = "thumbs up"
                elif hand_state["thumb_orientation"] == "down" and hand_state["finger_y_order"] == "down":
                    thumb_state = "thumbs down"
    return thumb_state
