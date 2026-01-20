"""
FPS Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import time
import cv2 as cv
import media_pumpkin


class FPS:
    """
    FPS class for calculating and displaying the Frames Per Second in a video stream.

    Attributes:
        previous_time (float): Previous time stamp.
        frame_times (list): List to keep track of frame times.
        average_count (int): Number of frames over which to average the FPS.
    """

    def __init__(self, average_count=30):
        """
        Initialize FPS class.

        :param average_count: Number of frames over which to average the FPS, default is 30.
        """
        self.previous_time = time.time()  # Initialize previous time to current time
        self.frame_times = []  # List to store the time taken for each frame
        self.average_count = average_count  # Number of frames to average over

    def update(self, image=None, pos=(20, 50), background_color=(255, 0, 255),
               text_color=(255, 255, 255), scale=3, thickness=3):
        """
        Update the frame rate and optionally display it on the image.

        :param image: Image to display FPS on. If None, just returns the FPS value.
        :param pos: Position to display FPS on the image.
        :param background_color: Background color of the FPS text.
        :param text_color: Text color of the FPS display.
        :param scale: Font scale of the FPS text.
        :param thickness: Thickness of the FPS text.
        :return: FPS value, and optionally the image with FPS drawn on it.
        """

        current_time = time.time()  # Get the current time
        delta = current_time - self.previous_time  # Calculate the time difference between the current and previous frame
        self.frame_times.append(delta)  # Append the time difference to the list
        self.previous_time = current_time  # Update previous time

        # Remove the oldest frame time if the list grows beyond avgCount
        if len(self.frame_times) > self.average_count:
            self.frame_times.pop(0)

        average_frame_time = sum(self.frame_times) / len(self.frame_times)  # Calculate the average frame time
        fps = 1 / average_frame_time  # Calculate FPS based on the average frame time

        # Draw FPS on image if img is provided
        if image is not None:
            media_pumpkin.putTextRect(image, f'FPS: {int(fps)}', pos,
                                      scale=scale, thickness=thickness, colorT=text_color,
                                      colorR=background_color, offset=10)
        return fps, image


def main():
    # Initialize the FPS class with an average count of 30 frames for smoothing
    fps_reader = FPS(average_count=30)

    # Initialize the webcam and set it to capture at 60 FPS
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 30)  # Set the frames per second to 30

    # Main loop to capture frames and display FPS
    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        img = cv.flip(img, 1)
        # Update the FPS counter and draw the FPS on the image
        # fps_reader.update returns the current FPS and the updated image
        fps, img = fps_reader.update(img, pos=(20, 50),
                                    background_color=(255, 0, 255), text_color=(255, 255, 255),
                                    scale=3, thickness=3)

        # Display the image with the FPS counter
        cv.imshow("Image", img)

        # Wait for 1 ms to show this frame, then continue to the next frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    main()
