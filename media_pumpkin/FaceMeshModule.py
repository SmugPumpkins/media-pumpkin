"""
Face Mesh Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import cv2 as cv
import mediapipe as mp
import math


def find_distance(p1, p2, image=None):
    """
    Find the distance between two landmarks based on their
    index numbers.
    :param p1: Point1
    :param p2: Point2
    :param image: Image to draw on.
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
        cv.circle(image, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(image, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(image, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return length,info, image
    else:
        return length, info


class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, static_mode=False, max_faces=2, min_detection_con=0.5, min_track_con=0.5):
        """
        :param static_mode: In static mode, detection is done on each image: slower
        :param max_faces: Maximum number of faces to detect
        :param min_detection_con: Minimum Detection Confidence Threshold
        :param min_track_con: Minimum Tracking Confidence Threshold
        """
        self.results = None
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_mode,
                                                    max_num_faces=self.max_faces,
                                                    min_detection_confidence=self.min_detection_con,
                                                    min_tracking_confidence=self.min_track_con)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, image, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param image: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb = cv.resize(image_rgb, (256, 144))
        self.results = self.face_mesh.process(image_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.draw_spec, self.draw_spec)
                face = []
                for index, landmark in enumerate(face_landmarks.landmark):
                    image_height, image_width, image_channels = image.shape
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    face.append([x, y])
                faces.append(face)
        return image, faces


def main():
    # Initialize the webcam
    # '2' indicates the third camera connected to the computer, '0' would usually refer to the built-in webcam
    cap = cv.VideoCapture(0)

    # Initialize FaceMeshDetector object
    # staticMode: If True, the detection happens only once, else every frame
    # maxFaces: Maximum number of faces to detect
    # minDetectionCon: Minimum detection confidence threshold
    # minTrackCon: Minimum tracking confidence threshold
    detector = FaceMeshDetector(static_mode=False, max_faces=2, min_detection_con=0.5, min_track_con=0.5)

    # Start the loop to continually get frames from the webcam
    while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: The current frame
        success, img = cap.read()
        img = cv.flip(img, 1)
        # Find face mesh in the image
        # img: Updated image with the face mesh if draw=True
        # faces: Detected face information
        img, faces = detector.find_face_mesh(img, draw=True)

        # Check if any faces are detected
        if faces:
            # Loop through each detected face
            for face in faces:
                # Get specific points for the eye
                # left_eye_up_point: Point above the left eye
                # left_eye_down_point: Point below the left eye
                left_eye_up_point = face[159]
                left_eye_down_point = face[23]

                # Calculate the vertical distance between the eye points
                # left_eye_vertical_distance: Distance between points above and below the left eye
                # info: Additional information (like coordinates)
                left_eye_vertical_distance, info = find_distance(left_eye_up_point, left_eye_down_point)

                # Print the vertical distance for debugging or information
                print(left_eye_vertical_distance)

        # Display the image in a window named 'Image'
        cv.imshow("Image", img)

        # Wait for 1 millisecond to check for any user input, keeping the window open
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
