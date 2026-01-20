"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import math

import cv2 as cv
import mediapipe as mp



class Hand:
    def __init__(self, landmarks_pixels, landmarks, bounding_box, center, side):
        """
        Stores hand data for later use.
        :param landmarks_pixels: The [x,y,z] pixel coordinates of landmarks
        :param landmarks: The actual landmarks of the hand as provided by mediapipe
        :param bounding_box: The bounding box of the hand
        :param center: The [x, y] center of the hand
        :param side: The side of the hand ("Left" or "Right")
        """
        self.landmarks_pixels = landmarks_pixels
        self.landmarks = landmarks
        self.bounding_box = bounding_box
        self.center = center
        self.side = side


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
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, static_mode=False, max_hands=2, model_complexity=1, detection_con=0.5, min_track_con=0.5):

        """
        :param static_mode: In static mode, detection is done on each image: slower
        :param max_hands: Maximum number of hands to detect
        :param model_complexity: Complexity of the hand landmark model: 0 or 1.
        :param detection_con: Minimum Detection Confidence Threshold
        :param min_track_con: Minimum Tracking Confidence Threshold
        """
        self.results = None
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.min_track_con = min_track_con
        self.mediapipe_hands = mp.solutions.hands
        self.hands = self.mediapipe_hands.Hands(static_image_mode=self.static_mode,
                                                max_num_hands=self.max_hands,
                                                model_complexity=self.model_complexity,
                                                min_detection_confidence=self.detection_con,
                                                min_tracking_confidence=self.min_track_con)

        self.mediapipe_draw = mp.solutions.drawing_utils
        self.tip_indices = [4, 8, 12, 16, 20]
        self.knuckle_indices = [2, 6, 10, 14, 18]
        self.thumb_indices = [1, 2, 3]
        self.fingers = []
        self.landmarks = []

    def find_hands(self, image, flip_side=False):
        """
        Finds hands in a BGR image.
        :param image: Image to find the hands in.
        :param flip_side: Switches left and right sides when True. Use when processed image is not being flipped already.
        :return: Image with or without drawings
        """
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb = cv.resize(image_rgb, (256, 144))
        self.results = self.hands.process(image_rgb)
        all_hands = []
        image_height, image_width, image_channels = image.shape
        if self.results.multi_hand_landmarks:
            for hand_side, hand_landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):

                ## landmark_list
                landmark_points = []
                landmarks = hand_landmarks
                x_list = []
                y_list = []
                for index, landmark in enumerate(hand_landmarks.landmark):
                    pixel_x, pixel_y, pixel_z = int(landmark.x * image_width), int(landmark.y * image_height), int(landmark.z * image_width)
                    landmark_points.append([pixel_x, pixel_y, pixel_z])
                    x_list.append(pixel_x)
                    y_list.append(pixel_y)

                ## bounding_box
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                bounding_box_width, bounding_box_height = x_max - x_min, y_max - y_min
                bounding_box = x_min, y_min, bounding_box_width, bounding_box_height
                center_x, center_y = bounding_box[0] + (bounding_box[2] // 2), \
                         bounding_box[1] + (bounding_box[3] // 2)

                ## flip_side
                if flip_side:
                    if hand_side.classification[0].label == "Right":
                        side = "Left"
                    else:
                        side = "Right"
                else:
                    side = hand_side.classification[0].label

                hand = Hand(landmark_points, landmarks, bounding_box, (center_x,center_y), side)
                all_hands.append(hand)

        return all_hands

    def draw_hand_debug(self, image, hand : Hand, show_skeleton=True, show_bounding_box=True, show_side=True):
        """
        Draws debug information for a hand on a specified image
        :param image: The image to draw the debug information on
        :param hand: The hand data to draw the debug information for
        :param show_skeleton: Flag for drawing the hand skeleton
        :param show_bounding_box: Flag for drawing the bounding box
        :param show_side: Flag for drawing the hand's side
        :return: Image with debug information drawn onto it
        """

        # TODO: Add the following options:
        # Display hand landmark indices
        # Display bounding box width and height
        # Display center x,y position

        # Hand skeleton
        if show_skeleton:
            self.mediapipe_draw.draw_landmarks(
                image,
                hand.landmarks,
                self.mediapipe_hands.HAND_CONNECTIONS
            )
        # Bounding box
        if show_bounding_box:
            cv.rectangle(
                image,
                (hand.bounding_box[0] - 20, hand.bounding_box[1] - 20),
                (hand.bounding_box[0] + hand.bounding_box[2] + 20,
                hand.bounding_box[1] + hand.bounding_box[3] + 20),
                (255, 0, 255),
                2
            )
        # Hand side ("Left" or "Right")
        if show_side:
            cv.putText(
                image,
                hand.side,
                (hand.bounding_box[0] - 30, hand.bounding_box[1] - 30),
                cv.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2
            )

        return image

    def fingers_up_absolute(self, hand : Hand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :param hand: The hand object to get the fingers for
        :return: List of which fingers are up as a list of binary numbers
        """
        fingers = []
        hand_side = hand.side
        landmarks = hand.landmarks_pixels
        if self.results.multi_hand_landmarks:

            # Thumb
            if hand_side == "Right":
                if landmarks[self.tip_indices[0]][0] > landmarks[self.tip_indices[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if landmarks[self.tip_indices[0]][0] < landmarks[self.tip_indices[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for index in range(1, 5):
                if landmarks[self.tip_indices[index]][1] < landmarks[self.tip_indices[index] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def fingers_up(self, hand : Hand):
        """
        Finds how many fingers are open and returns in a list.
        :param hand: The hand to find the fingers up on.
        :return: List of which fingers are up
        """
        fingers = []
        landmarks = hand.landmarks_pixels
        if self.results.multi_hand_landmarks:
            # Thumb
            dist_threshold = 0.2  # How sensitive it should be for an open vs a closed thumb

            # Get the angle vectors of landmark 1-2 and landmark 2-3
            angle_a = angle_3d(landmarks[self.thumb_indices[0]], landmarks[self.thumb_indices[1]])
            angle_b = angle_3d(landmarks[self.thumb_indices[1]], landmarks[self.thumb_indices[2]])

            # Get the distance between the ends of the two angle vectors
            thumb_angle_distance = math.dist(angle_a, angle_b)

            # When thumb is open, the distance is less than the threshold
            if thumb_angle_distance < dist_threshold:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for index in range(1, 5):
                # Compare the distance between the tip and the wrist to the distance between the knuckle and the wrist
                tip_distance = math.dist(landmarks[self.tip_indices[index]], landmarks[0])
                knuckle_distance = math.dist(landmarks[self.knuckle_indices[index]], landmarks[0])
                if tip_distance > knuckle_distance:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def landmark_distance(self, hand_1 : Hand, landmark_index_1, landmark_index_2, hand_2 : None | Hand = None):
        """
        Gets the distance between 2 landmarks in pixels from the landmark indices of the hand.
        :param hand_1: Hand that the landmarks are on
        :param landmark_index_1: Index of the first landmark
        :param landmark_index_2: Index of the second landmark
        :param hand_2: If needing a landmark from a second hand, this is the second hand
        :return: Distance in pixels between the 2 landmarks
        """
        distance = None
        if self.results.multi_hand_landmarks:
            if hand_2 is None:
                landmark_1 = hand_1.landmarks_pixels[landmark_index_1]
                landmark_2 = hand_1.landmarks_pixels[landmark_index_2]
                distance = math.dist(landmark_1, landmark_2)
            else:
                landmark_1 = hand_1.landmarks_pixels[landmark_index_1]
                landmark_2 = hand_2.landmarks_pixels[landmark_index_2]
                distance = math.dist(landmark_1, landmark_2)
        return distance

    # TODO: Add methods for hand_angle, point_angle, and finger_bend, and changing hand skeleton colors for the debugger
def main():
    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(static_mode=False, max_hands=2, model_complexity=1, detection_con=0.5, min_track_con=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv.flip(img, 1)
        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands = detector.find_hands(img, flip_side=False)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 : Hand = hands[0]  # Get the first hand detected
            landmark_list_1 = hand1.landmarks_pixels  # List of 21 landmarks for the first hand
            detector.draw_hand_debug(img, hand1)

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingers_up(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

            # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = find_distance(landmark_list_1[8][0:2], landmark_list_1[12][0:2], img, color=(255, 0, 255),
                                                       scale=10)

            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 : Hand= hands[1]
                landmark_list_2 = hand2.landmarks_pixels

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingers_up_absolute(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")

                # Calculate distance between the index fingers of both hands and draw it on the image
                length, info, img = find_distance(landmark_list_1[8][0:2], landmark_list_2[8][0:2], img, color=(255, 0, 0),
                                                           scale=10)

            print(" ")  # New line for better readability of the printed output

        # Display the image in a window
        cv.imshow("Image", img)

        # Wait for 1 ms to show this frame, then continue to the next frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
