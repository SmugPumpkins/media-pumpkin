"""
Face Detection Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import cv2 as cv
import mediapipe as mp

import media_pumpkin


class FaceDetector:
    """
    Find faces in realtime using the lightweight model provided in the mediapipe
    library.
    """

    def __init__(self, min_detection_con=0.5, model_selection=0):
        """
        :param min_detection_con: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_detection#min_detection_confidence.

        :param model_selection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        self.min_detection_con = min_detection_con
        self.model_selection = model_selection
        self.mediapipe_face_detection = mp.solutions.face_detection
        self.mediapipe_draw = mp.solutions.drawing_utils
        self.face_detection = self.mediapipe_face_detection.FaceDetection(min_detection_confidence=self.min_detection_con,
                                                                          model_selection=self.model_selection)
        self.results = None

    def find_faces(self, image, draw=True):
        """
        Find faces in an image and return the bounding info
        :param image: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.face_detection.process(image_rgb)
        bounding_box = []
        if self.results.detections:
            for marker, detection in enumerate(self.results.detections):
                if detection.score[0] > self.min_detection_con:
                    bounding_box_data = detection.location_data.relative_bounding_box
                    image_height, image_width, image_channels = image.shape
                    bounding = int(bounding_box_data.xmin * image_width), int(bounding_box_data.ymin * image_height), \
                        int(bounding_box_data.width * image_width), int(bounding_box_data.height * image_height)
                    cx, cy = bounding[0] + (bounding[2] // 2), \
                             bounding[1] + (bounding[3] // 2)
                    bounding_box_info = {"id": marker, "bounding": bounding, "score": detection.score, "center": (cx, cy)}
                    bounding_box.append(bounding_box_info)
                    if draw:
                        image = cv.rectangle(image, bounding, (255, 0, 255), 2)

                        cv.putText(image, f'{int(detection.score[0] * 100)}%',
                                   (bounding[0], bounding[1] - 20), cv.FONT_HERSHEY_PLAIN,
                                   2, (255, 0, 255), 2)
        return image, bounding_box


def main():
    # Initialize the webcam
    # '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
    cap = cv.VideoCapture(0)

    # Initialize the FaceDetector object
    # minDetectionCon: Minimum detection confidence threshold
    # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
    detector = FaceDetector(min_detection_con=0.5, model_selection=0)

    # Run the loop to continually get frames from the webcam
    while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: the captured frame
        success, img = cap.read()

        img = cv.flip(img, 1)

        # Detect faces in the image
        # img: Updated image
        # bounding_boxes: List of bounding boxes around detected faces
        img, bounding_boxes = detector.find_faces(img, draw=False)

        # Check if any face is detected
        if bounding_boxes:
            # Loop through each bounding box
            for bbox in bounding_boxes:
                # bbox contains 'id', 'bbox', 'score', 'center'

                # ---- Get Data  ---- #
                center = bbox["center"]
                x, y, w, h = bbox['bounding']
                score = int(bbox['score'][0] * 100)

                # ---- Draw Data  ---- #
                cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)
                media_pumpkin.putTextRect(img, f'{score}%', (x, y - 10))
                media_pumpkin.cornerRect(img, (x, y, w, h))

        # Display the image in a window named 'Image'
        cv.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
        cv.waitKey(1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
