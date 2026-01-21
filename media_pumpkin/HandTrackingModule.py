"""
Hand Tracking Module
By: SmugPumpkins
"""

import math

import cv2 as cv
import mediapipe as mp

from media_pumpkin import BoundingBox, justify_center, align_top
from media_pumpkin.Utils import stack_text, HAlign, VAlign


class Hand:
    def __init__(self, landmarks, original_landmarks, box : BoundingBox, side):
        """
        Stores hand data for later use.
        :param landmarks: The [x,y,z] pixel coordinates of landmarks
        :param original_landmarks: The actual landmarks of the hand as provided by mediapipe
        :param box: The bounding box of the hand
        :param side: The side of the hand ("Left" or "Right")
        """
        self.landmarks = landmarks
        self.original_landmarks = original_landmarks
        self.box = box
        self.center = self.box.center
        self.side = side
        self.thumb = self.landmarks[4]
        self.index = self.landmarks[8]
        self.middle = self.landmarks[12]
        self.ring = self.landmarks[16]
        self.pinky = self.landmarks[20]
        self.wrist = self.landmarks[0]
        self.connection_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
        self.landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self.default_connection_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
        self.default_landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self.flags = self.finger_flags()

    def landmark_distance(self, landmark_index_1, landmark_index_2, image=None):
        """
        Finds the distance in pixels between 2 specified landmarks.
        :param landmark_index_1: The landmark index for the first point
        :param landmark_index_2: The landmark index for the second point
        :param image: If not None, draws a line between the 2 points on the specified image
        :return: The distance between 2 points
        """
        landmark_1 = self.landmarks[landmark_index_1]
        landmark_2 = self.landmarks[landmark_index_2]
        distance : float = math.dist(landmark_1, landmark_2)
        if image is not None:
            pass
        return distance


    def finger_flags(self):
        """
        Finds which fingers are up and returns them as a binary list in the order of thumb, index, middle, ring, pinky.
        :return: A list of binary values representing whether a finger is up or down
        """
        # Initialize empty list for finger flags
        fingers = []

        # Distance for thumb to be considered open
        distance_threshold = 0.2

        # Vector math to determine whether landmarks 1→2 are closely aligned with 2→3
        angle_a = angle_3d(self.landmarks[1], self.landmarks[2])
        angle_b = angle_3d(self.landmarks[2], self.landmarks[3])
        thumb_angle_distance = math.dist(angle_a, angle_b)

        # Append thumb value to fingers
        if thumb_angle_distance < distance_threshold:
            fingers.append(1)
        else:
            fingers.append(0)

        # Landmark indices for index_tip, middle_tip, ring_tip, and pinky_tip
        finger_indices = [8, 12, 16, 20]

        # Compare the distance between the tip and the wrist to the distance between the knuckle and the wrist
        for index in finger_indices:
            tip_distance = math.dist(self.landmarks[index], self.landmarks[0])
            knuckle_distance = math.dist(self.landmarks[index - 2], self.landmarks[0])
            # Append each finger value to fingers
            if tip_distance > knuckle_distance:
                fingers.append(1)
            else:
                fingers.append(0)

        # Return list of flags
        return fingers

    def fingers_up(self):
        """
        Provides a list of the fingers that are up in English.
        """
        # Initialize empty list for finger flags
        fingers = []
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for flag, name in zip(self.flags, finger_names):
            # Append fingers
            if flag > 0:
                fingers.append(name)

        # Return list of fingers that are up
        return fingers

    def fingers_down(self):
        """
        Provides a list of the fingers that are down in English.
        """
        # Initialize empty list for finger flags
        fingers = []
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for flag, name in zip(self.flags, finger_names):
            # Append fingers
            if flag < 1:
                fingers.append(name)
        # Return list of fingers that are up
        return fingers

    def draw(self, image):
        """
        Draws the hand skeleton on the specified image.
        :param image: The target image for the drawing.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            self.original_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            self.landmark_style,
            self.connection_style
        )

    def debug(self, image, skeleton=True, bounding_box=True, center=True, side=True, fingers=True, flags=True):
        """
        Draws the requested debug information. Defaults to all debug information.
        """
        height, width, _ = image.shape
        debug_text_size = 1.2
        debug_font = cv.FONT_HERSHEY_PLAIN
        debug_thickness = 1
        if skeleton:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                self.original_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                self.default_landmark_style,
                self.default_connection_style
            )
        if bounding_box:
            self.box.draw(image)
        if center:
            center_text = f"Center (x:{self.center[0]}, y:{self.center[1]})"
            cv.circle(
                image,
                self.center,
                10,
                (0,0,0),
                -1
            )
            stack_text(
                image,
                [center_text],
                self.center,
                debug_font,
                debug_text_size * 1.5,
                debug_thickness * 2,
                (0,0,50),
                HAlign.LEFT,
                VAlign.BOTTOM
            )
        if side:
            x_center = justify_center(self.side, debug_font, debug_text_size, debug_thickness)
            y_top = align_top(self.side, debug_font, debug_text_size, debug_thickness)
            x_position = x_center + self.wrist[0]
            y_position = y_top + self.box.opposite[1]
            cv.putText(
                image,
                self.side,
                (x_position,y_position),
                debug_font,
                debug_text_size * 1.5,
                (0,0,0),
                debug_thickness * 2
            )

        lines = []
        if flags:
            flag_text = f"Flags: {self.flags}"
            lines.append(flag_text)
        if fingers:
            lines.append("Fingers:")
            for finger in self.fingers_up():
                lines.append(finger)
        if self.side == "Left":
            stack_text(
                image,
                lines,
                (0,0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0,0,0)
            )
        else:
            stack_text(
                image,
                lines,
                (width, 0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0, 0, 0),
                h_align=HAlign.RIGHT
            )

    def set_connection_style(self, color=None, thickness=None):
        """
        Modifies the style of the hand connections when it is drawn.
        :param color: The BGR color of the connections
        :param thickness: The thickness of the connector lines in pixels
        """

        old_color, old_thickness, old_radius = None, None, None
        old_values = self.connection_style.values()
        for old_value in old_values:
            old_color = old_value.color
            old_thickness = old_value.thickness
            old_radius = old_value.circle_radius
            break

        if color is not None:
            new_color = color
        else:
            new_color = old_color
        if thickness is not None:
            new_thickness = thickness
        else:
            new_thickness = old_thickness
        self.connection_style = mp.solutions.drawing_styles.DrawingSpec(new_color, new_thickness, old_radius)

    def set_landmarks_style(self, color=None, thickness=None, radius=None):
        """
        Modifies the style of the hand landmarks when drawn.
        :param color: The BGR color of the landmarks
        :param thickness: The thickness of the outlines of the circles?
        :param radius: The radius of the landmarks
        """
        old_color = None
        old_thickness = None
        old_radius = None
        old_values = self.landmark_style.values()
        for old_value in old_values:
            old_color = old_value.color
            old_thickness = old_value.thickness
            old_radius = old_value.circle_radius
            break

        if color is not None:
            new_color = color
        else:
            new_color = old_color
        if thickness is not None:
            new_thickness = thickness
        else:
            new_thickness = old_thickness
        if radius is not None:
            new_radius = radius
        else:
            new_radius = old_radius
        self.landmark_style = mp.solutions.drawing_styles.DrawingSpec(new_color, new_thickness, new_radius)

def angle_3d(p1, p2):
    """
    Returns the normalized 3D vector of 2 points.
    :param p1: The origin point
    :param p2: The offset point
    :return: Normalized 3D point representing the angle between 2 3D points
    """
    # vector 1 → 2
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    # vector magnitude (length)
    magnitude = math.sqrt(x * x + y * y + z * z)
    # avoid division by zero
    if magnitude == 0:
        return 0, 0, 0
    # normalized vector (unit length)
    return x / magnitude, y / magnitude, z / magnitude


def find_distance(p1, p2, image=None, color=(255, 0, 255), scale=5):
    """
    Find the distance between two landmarks input should be (x1,y1) (x2,y2)
    :param p1: Point1 (x1,y1)
    :param p2: Point2 (x2,y2)
    :param image: Image to draw output on. If no image input output img is None
    :param color: The color used to draw the line connecting the 2 points
    :param scale: Controls line thicknesses and size of circles
    :return: Distance between the points
             Image with output drawn
             Line information
    """

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if image is not None:
        cv.circle(image, (x1, y1), scale, color, cv.FILLED)
        cv.circle(image, (x2, y2), scale, color, cv.FILLED)
        cv.line(image, (x1, y1), (x2, y2), color, max(1, scale // 3))
        cv.circle(image, (cx, cy), scale, color, cv.FILLED)

    return length, info, image

class HandDetector:
    def __init__(
        self,
        max_hands=2,
        static_mode=False,
        model_complexity=1,
        detection_con=0.5,
        min_track_con=0.5,
        flip_side=False
    ):
        self.flip_side = flip_side

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_con,
            min_tracking_confidence=min_track_con
        )

    def find_hands(self, image):
        image_height, image_width, _ = image.shape

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        all_hands = []

        if not results.multi_hand_landmarks:
            return all_hands

        for hand_side, hand_landmarks in zip(
            results.multi_handedness,
            results.multi_hand_landmarks
        ):
            landmark_points = []
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                px = int(lm.x * image_width)
                py = int(lm.y * image_height)
                pz = int(lm.z * image_width)

                landmark_points.append([px, py, pz])
                x_list.append(px)
                y_list.append(py)

            bounding_box = BoundingBox(
                (min(x_list) - 10, min(y_list) - 10),
                (max(x_list) + 10, max(y_list) + 10)
            )

            label = hand_side.classification[0].label
            if self.flip_side:
                side = "Left" if label == "Right" else "Right"
            else:
                side = label

            all_hands.append(
                Hand(
                    landmark_points,
                    hand_landmarks,
                    bounding_box,
                    side
                )
            )

        return all_hands


def main():
    # Initialize the webcam to capture video
    cap = cv.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    hand_detector = HandDetector( 2)
    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv.flip(img, 1)
        # Find hands in the current frame
        hands = hand_detector.find_hands(img)

        # Methods for each hand
        for hand in hands:
            hand.debug(img)

        # Display the image in a window
        cv.imshow("Image", img)

        # Wait for 1 ms to show this frame, then continue to the next frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
