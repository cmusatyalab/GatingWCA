package edu.cmu.cs.wca;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;

public class ThumbsUpDetection {
    public Hands hands;

    public ThumbsUpDetection(MainActivity mainActivity) {
        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(2)
                .setRunOnGpu(true)
                .build();
        hands = new Hands(mainActivity, handsOptions);
    }

    public static boolean detectThumbsUp(HandsResult result) {
        HashMap<String, Object> handState = getHandState(result);
        if ((Boolean)handState.get("thumb_open") &&
                (Boolean)handState.get("index_finger_closed") &&
                (Boolean)handState.get("middle_finger_closed") &&
                (Boolean)handState.get("ring_finger_closed") &&
                (Boolean)handState.get("pinky_closed")) {

            if ((Double)handState.get("thumb_middle_angle") <= 90 &&
                    (Double)handState.get("thumb_middle_angle") > (Double)handState.get("thumb_index_angle") &&
                    (Double)handState.get("thumb_index_angle") > 15) {

                if ((Double)handState.get("orientation") > 120 ||
                        (Double)handState.get("orientation") < -150 ||
                        ((Double)handState.get("orientation") < 60 &&
                                (Double)handState.get("orientation") > -30)) {

                    if (handState.get("thumb_orientation").equals("up") &&
                            handState.get("finger_y_order").equals("up")) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static double dist2D(double x1, double y1, double x2, double y2) {
        return Math.hypot(Math.abs(y2 - y1), Math.abs(x2 - x1));
    }

    private static double dot2D(double v1x, double v1y, double v2x, double v2y) {
        return v1x * v2x + v1y * v2y;
    }

    private static double vectorAngleDegree(double v1x, double v1y, double v2x, double v2y) {
        return Math.toDegrees(Math.acos(dot2D(v1x, v1y, v2x, v2y) /
                Math.sqrt(dot2D(v1x, v1y, v1x, v1y)) /
                Math.sqrt(dot2D(v2x, v2y, v2x, v2y))));
    }

    private static HashMap<String, Object> getHandState(HandsResult result) {
        int width = result.inputBitmap().getWidth();
        int height = result.inputBitmap().getHeight();
        List<LandmarkProto.NormalizedLandmark> handLandmark =
                result.multiHandLandmarks().get(0).getLandmarkList();

        HashMap<String, Object> handState = new HashMap<>();

        double x0 = handLandmark.get(HandLandmark.WRIST).getX() * width;
        double y0 = handLandmark.get(HandLandmark.WRIST).getY() * height;
        double x9 = handLandmark.get(HandLandmark.MIDDLE_FINGER_MCP).getX() * width;
        double y9 = handLandmark.get(HandLandmark.MIDDLE_FINGER_MCP).getY() * height;
        handState.put("orientation", Math.toDegrees(Math.atan2(y0 - y9, x9 - x0)));

        double x1 = handLandmark.get(HandLandmark.THUMB_CMC).getX() * width;
        double y1 = handLandmark.get(HandLandmark.THUMB_CMC).getY() * height;
        double x2 = handLandmark.get(HandLandmark.THUMB_MCP).getX() * width;
        double y2 = handLandmark.get(HandLandmark.THUMB_MCP).getY() * height;
        double x3 = handLandmark.get(HandLandmark.THUMB_IP).getX() * width;
        double y3 = handLandmark.get(HandLandmark.THUMB_IP).getY() * height;
        double x4 = handLandmark.get(HandLandmark.THUMB_TIP).getX() * width;
        double y4 = handLandmark.get(HandLandmark.THUMB_TIP).getY() * height;
        double d01 = dist2D(x0, y0, x1, y1);
        double d02 = dist2D(x0, y0, x2, y2);
        double d03 = dist2D(x0, y0, x3, y3);
        double d04 = dist2D(x0, y0, x4, y4);
        handState.put("thumb_open", d04 > d03 && d03 > d02 && d02 > d01);

        double x5 = handLandmark.get(HandLandmark.INDEX_FINGER_MCP).getX() * width;
        double y5 = handLandmark.get(HandLandmark.INDEX_FINGER_MCP).getY() * height;
        double v04x = x4 - x0;
        double v04y = y4 - y0;
        double v05x = x5 - x0;
        double v05y = y5 - y0;
        double v09x = x9 - x0;
        double v09y = y9 - y0;
        handState.put("thumb_index_angle", vectorAngleDegree(v04x, v04y, v05x, v05y));
        handState.put("thumb_middle_angle", vectorAngleDegree(v04x, v04y, v09x, v09y));

        double x6 = handLandmark.get(HandLandmark.INDEX_FINGER_PIP).getX() * width;
        double y6 = handLandmark.get(HandLandmark.INDEX_FINGER_PIP).getY() * height;
        double x7 = handLandmark.get(HandLandmark.INDEX_FINGER_DIP).getX() * width;
        double y7 = handLandmark.get(HandLandmark.INDEX_FINGER_DIP).getY() * height;
        double x8 = handLandmark.get(HandLandmark.INDEX_FINGER_TIP).getX() * width;
        double y8 = handLandmark.get(HandLandmark.INDEX_FINGER_TIP).getY() * height;
        double d06 = dist2D(x0, y0, x6, y6);
        double d07 = dist2D(x0, y0, x7, y7);
        double d08 = dist2D(x0, y0, x8, y8);
        handState.put("index_finger_closed", d06 > d07 && d07 > d08);

        double x10 = handLandmark.get(HandLandmark.MIDDLE_FINGER_PIP).getX() * width;
        double y10 = handLandmark.get(HandLandmark.MIDDLE_FINGER_PIP).getY() * height;
        double x11 = handLandmark.get(HandLandmark.MIDDLE_FINGER_DIP).getX() * width;
        double y11 = handLandmark.get(HandLandmark.MIDDLE_FINGER_DIP).getY() * height;
        double x12 = handLandmark.get(HandLandmark.MIDDLE_FINGER_TIP).getX() * width;
        double y12 = handLandmark.get(HandLandmark.MIDDLE_FINGER_TIP).getY() * height;
        double d010 = dist2D(x0, y0, x10, y10);
        double d011 = dist2D(x0, y0, x11, y11);
        double d012 = dist2D(x0, y0, x12, y12);
        handState.put("middle_finger_closed", d010 > d011 && d011 > d012);

        double x14 = handLandmark.get(HandLandmark.RING_FINGER_PIP).getX() * width;
        double y14 = handLandmark.get(HandLandmark.RING_FINGER_PIP).getY() * height;
        double x15 = handLandmark.get(HandLandmark.RING_FINGER_DIP).getX() * width;
        double y15 = handLandmark.get(HandLandmark.RING_FINGER_DIP).getY() * height;
        double x16 = handLandmark.get(HandLandmark.RING_FINGER_TIP).getX() * width;
        double y16 = handLandmark.get(HandLandmark.RING_FINGER_TIP).getY() * height;
        double d014 = dist2D(x0, y0, x14, y14);
        double d015 = dist2D(x0, y0, x15, y15);
        double d016 = dist2D(x0, y0, x16, y16);
        handState.put("ring_finger_closed", d014 > d015 && d015 > d016);

        double x18 = handLandmark.get(HandLandmark.PINKY_PIP).getX() * width;
        double y18 = handLandmark.get(HandLandmark.PINKY_PIP).getY() * height;
        double x19 = handLandmark.get(HandLandmark.PINKY_DIP).getX() * width;
        double y19 = handLandmark.get(HandLandmark.PINKY_DIP).getY() * height;
        double x20 = handLandmark.get(HandLandmark.PINKY_TIP).getX() * width;
        double y20 = handLandmark.get(HandLandmark.PINKY_TIP).getY() * height;
        double d018 = dist2D(x0, y0, x18, y18);
        double d019 = dist2D(x0, y0, x19, y19);
        double d020 = dist2D(x0, y0, x20, y20);
        handState.put("pinky_closed", d018 > d019 && d019 > d020);

        Integer[] landmarkYOrder = new Integer[handLandmark.size()];
        for (int i = 0; i < landmarkYOrder.length; i++) {
            landmarkYOrder[i] = i;
        }
        // A brief implementation of argsort
        Arrays.sort(landmarkYOrder, (i1, i2) -> Float.compare(handLandmark.get(i1).getY(), handLandmark.get(i2).getY()));
        String thumbOrientation = "unknown";
        if (landmarkYOrder[0] == HandLandmark.THUMB_TIP && landmarkYOrder[1] == HandLandmark.THUMB_IP) {
            thumbOrientation = "up";
        } else if (landmarkYOrder[landmarkYOrder.length - 1] == HandLandmark.THUMB_TIP && landmarkYOrder[landmarkYOrder.length - 2] == HandLandmark.THUMB_IP) {
            thumbOrientation = "down";
        }
        handState.put("thumb_orientation", thumbOrientation);

        String fingerYOrder = "unknown";
        if (y4 < y6 && y6 < y10 && y10 < y14 && y14 < y18) {
            fingerYOrder = "up";
        } else if (y4 > y6 && y6 > y10 && y10 > y14 && y14 > y18) {
            fingerYOrder = "down";
        }
        handState.put("finger_y_order", fingerYOrder);

        return handState;
    }
}
