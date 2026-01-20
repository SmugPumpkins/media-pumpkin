"""
Color Module
Finds color in an image based on hsv values
Can run as stand alone to find relevant hsv values

"""

import cv2 as cv
import numpy as np

import media_pumpkin


class ColorFinder:
    def __init__(self, track_bar=False):
        """
        :param track_bar: Whether to use OpenCV trackbars to dynamically adjust HSV values. Default is False.
        """
        self.track_bar = track_bar
        if self.track_bar:
            self.init_trackbars()

    def empty(self, a):
        """An empty function to pass as a parameter when creating trackbars."""
        pass

    def init_trackbars(self):
        """Initialize the OpenCV trackbars for dynamic HSV value adjustment."""
        cv.namedWindow("TrackBars")
        cv.resizeWindow("TrackBars", 640, 240)
        cv.createTrackbar("Hue Min", "TrackBars", 0, 179, self.empty)
        cv.createTrackbar("Hue Max", "TrackBars", 179, 179, self.empty)
        cv.createTrackbar("Sat Min", "TrackBars", 0, 255, self.empty)
        cv.createTrackbar("Sat Max", "TrackBars", 255, 255, self.empty)
        cv.createTrackbar("Val Min", "TrackBars", 0, 255, self.empty)
        cv.createTrackbar("Val Max", "TrackBars", 255, 255, self.empty)

    def get_trackbar_values(self):
        """
         Get the current HSV values set by the trackbars.

         :return: A dictionary containing the current HSV values from the trackbars.
         """
        hue_min = cv.getTrackbarPos("Hue Min", "TrackBars")
        saturation_min = cv.getTrackbarPos("Sat Min", "TrackBars")
        value_min = cv.getTrackbarPos("Val Min", "TrackBars")
        hue_max = cv.getTrackbarPos("Hue Max", "TrackBars")
        saturation_max = cv.getTrackbarPos("Sat Max", "TrackBars")
        value_max = cv.getTrackbarPos("Val Max", "TrackBars")

        hsv_values = {"hue_min": hue_min, "saturation_min": saturation_min, "value_min": value_min,
                   "hue_max": hue_max, "saturation_max": saturation_max, "value_max": value_max}
        print(hsv_values)
        return hsv_values

    def update(self, image, my_color=None):
        """
        Find a specified color in the given image.

        :param image: The image in which to find the color.
        :param my_color: The color to find, can be a string or None.

        :return: A tuple containing a mask image with only the specified color, and the original image masked to only show the specified color.
        """
        image_color = []
        mask = []

        if self.track_bar:
            my_color = self.get_trackbar_values()

        if isinstance(my_color, str):
            my_color = self.getColorHSV(my_color)

        if my_color is not None:
            image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            lower = np.array([my_color['hue_min'], my_color['saturation_min'], my_color['value_min']])
            upper = np.array([my_color['hue_max'], my_color['saturation_max'], my_color['value_max']])
            mask = cv.inRange(image_hsv, lower, upper)
            image_color = cv.bitwise_and(image, image, mask=mask)

        return image_color, mask

def main():
    # Create an instance of the ColorFinder class with trackBar set to True.
    myColorFinder = ColorFinder(track_bar=True)

    # Initialize the video capture using OpenCV.
    cap = cv.VideoCapture(0)

    # Set the dimensions of the camera feed to 640x480.
    cap.set(3, 640)
    cap.set(4, 480)

    # Custom color values for detecting orange.
    # 'hmin', 'smin', 'vmin' are the minimum values for Hue, Saturation, and Value.
    # 'hmax', 'smax', 'vmax' are the maximum values for Hue, Saturation, and Value.
    hsv_vals = {'hue_min': 10, 'saturation_min': 55, 'value_min': 215, 'hue_max': 42, 'saturation_max': 255,
                'value_max': 255}

    # Main loop to continuously get frames from the camera.
    while True:
        # Read the current frame from the camera.
        success, img = cap.read()
        if not success:
            break

        # Use the update method from the ColorFinder class to detect the color.
        # It returns the masked color image and a binary mask.
        img_orange, mask = myColorFinder.update(img, hsv_vals)

        # Stack the original image, the masked color image, and the binary mask.
        img_stack = media_pumpkin.stackImages([img, img_orange, mask], 3, 1)

        # Show the stacked images.
        cv.imshow("Image Stack", img_stack)

        # Break the loop if the 'q' key is pressed.
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image Stack", cv.WND_PROP_VISIBLE) < 1:
            break
        if cv.getWindowProperty("TrackBars", cv.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
