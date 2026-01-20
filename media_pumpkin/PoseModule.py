"""
Pose Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""
import math

import cv2 as cv
import mediapipe as mp


class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(
        self,
        static_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):
        """
        :param static_mode: In static mode, detection is done on each image: slower
        :param model_complexity:
        :param smooth_landmarks:
        :param enable_segmentation:
        :param smooth_segmentation:
        :param detection_confidence: Minimum Detection Confidence Threshold
        :param tracking_confidence: Minimum Tracking Confidence Threshold
        """
        self.results = None
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mediapipe_draw = mp.solutions.drawing_utils
        self.mediapipe_pose = mp.solutions.pose
        self.pose = self.mediapipe_pose.Pose(static_image_mode=self.static_mode,
                                             model_complexity=self.model_complexity,
                                             smooth_landmarks=self.smooth_landmarks,
                                             enable_segmentation=self.enable_segmentation,
                                             smooth_segmentation=self.smooth_segmentation,
                                             min_detection_confidence=self.detection_confidence,
                                             min_tracking_confidence=self.tracking_confidence)

    def find_pose(self, image, draw=True):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param image: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb = cv.resize(image_rgb, (256, 144))
        self.results = self.pose.process(image_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mediapipe_draw.draw_landmarks(image, self.results.pose_landmarks, self.mediapipe_pose.POSE_CONNECTIONS)
        return image

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][0] - self.lmList[11][0]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][0] - ad
                x2 = self.lmList[15][0] + ad
            else:
                x1 = self.lmList[12][0] - ad
                x2 = self.lmList[11][0] + ad

            y2 = self.lmList[29][1] + ad
            y1 = self.lmList[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv.rectangle(img, bbox, (255, 0, 255), 3)
                cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        return self.lmList, self.bboxInfo

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
           Find the distance between two landmarks input should be (x1,y1) (x2,y2)
           :param p1: Point1 (x1,y1)
           :param p2: Point2 (x2,y2)
           :param img: Image to draw output on. If no image input output img is None
           :return: Distance between the points
                    Image with output drawn
                    Line information
           """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv.circle(img, (x1, y1), scale, color, cv.FILLED)
            cv.circle(img, (x2, y2), scale, color, cv.FILLED)
            cv.circle(img, (cx, cy), scale, color, cv.FILLED)

        return length, img, info

    def findAngle(self,p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        """
        Finds angle between three points.

        :param p1: Point1 - (x1,y1)
        :param p2: Point2 - (x2,y2)
        :param p3: Point3 - (x3,y3)
        :param img: Image to draw output on. If no image input output img is None
        :return:
        """

        # Get the landmarks
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if img is not None:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv.circle(img, (x1, y1), scale, color, cv.FILLED)
            cv.circle(img, (x1, y1), scale+5, color, max(1,scale//5))
            cv.circle(img, (x2, y2), scale, color, cv.FILLED)
            cv.circle(img, (x2, y2), scale+5, color, max(1,scale//5))
            cv.circle(img, (x3, y3), scale, color, cv.FILLED)
            cv.circle(img, (x3, y3), scale+5, color, max(1,scale//5))
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv.FONT_HERSHEY_PLAIN, 2, color, max(1,scale//5))
        return angle, img

    def angleCheck(self, myAngle, targetAngle, offset=20):
        return targetAngle - offset < myAngle < targetAngle + offset


def main():
    cap = cv.VideoCapture(0)

    # Initialize the PoseDetector class with the given parameters
    detector = PoseDetector(static_mode=False,
                            model_complexity=1,
                            smooth_landmarks=True,
                            enable_segmentation=False,
                            smooth_segmentation=True,
                            detection_confidence=0.5,
                            tracking_confidence=0.5)

    # Loop to continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        # Find the human pose in the frame
        img = detector.find_pose(img)
        img = cv.flip(img, 1)
        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

        # Check if any body landmarks are detected
        if lmList:
            # Get the center of the bounding box around the body
            center = bboxInfo["center"]

            # Draw a circle at the center of the bounding box
            cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)

            # Calculate the distance between landmarks 11 and 15 and draw it on the image
            length, img, info = detector.findDistance(lmList[11][0:2],
                                                      lmList[15][0:2],
                                                      img=img,
                                                      color=(255, 0, 0),
                                                      scale=10)

            # Calculate the angle between landmarks 11, 13, and 15 and draw it on the image
            angle, img = detector.findAngle(lmList[11][0:2],
                                            lmList[13][0:2],
                                            lmList[15][0:2],
                                            img=img,
                                            color=(0, 0, 255),
                                            scale=10)

            # Check if the angle is close to 50 degrees with an offset of 10
            isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                                 targetAngle=50,
                                                 offset=10)

            # Print the result of the angle check
            print(isCloseAngle50)

        # Display the frame in a window
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    main()
