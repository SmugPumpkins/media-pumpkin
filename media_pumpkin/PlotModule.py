import math
import time

import cv2 as cv
import numpy as np


class LivePlot:
    """
    A class for real-time plotting in OpenCV.
    """

    def __init__(self, width=640, height=480, y_limit=(0, 100),
                 interval=0.001, invert=True, char='Y'):
        """
        Initialize the LivePlot object.

        :param width: Width of the plotting window
        :param height: Height of the plotting window
        :param y_limit: Y-axis limits [y_min, y_max]
        :param interval: Time interval for updating the plot
        :param invert: Whether to invert the y-axis
        :param char: A character to display on the plot for annotation
        """

        self.y_limit = y_limit
        self.width = width
        self.height = height
        self.invert = invert
        self.interval = interval
        self.char = char[0]
        self.image_plot = np.zeros((self.height, self.width, 3), np.uint8)
        self.image_plot[:] = 225, 225, 225
        self.y = 0
        self.y_points = []
        self.x_points = [x for x in range(0, 100)]
        self.p_time = 0

    def update(self, y, color=(255, 0, 255)):
        """
        Update the plot with a new y-value.

        :param y: The new y-value to plot
        :param color: RGB color for the plot line

        :return: Updated image of the plot
        """

        # Check if enough time has passed for an update
        if time.time() - self.p_time > self.interval:
            self.image_plot[:] = 225, 225, 225  # Refresh
            self.draw_background()  # Draw static parts
            cv.putText(self.image_plot, str(y), (self.width - 125, 50), cv.FONT_HERSHEY_PLAIN, 3, (150, 150, 150), 3)

            # Interpolate y-value to plot height
            if self.invert:
                self.y = int(np.interp(y, self.y_limit, [self.height, 0]))
            else:
                self.y = int(np.interp(y, self.y_limit, [0, self.height]))

            self.y_points.append(self.y)
            if len(self.y_points) == 100:
                self.y_points.pop(0)

            # Draw plot lines
            for i in range(2, len(self.y_points)):
                x1 = int((self.x_points[i - 1] * (self.width // 100)) - (self.width // 10))
                y1 = self.y_points[i - 1]
                x2 = int((self.x_points[i] * (self.width // 100)) - (self.width // 10))
                y2 = self.y_points[i]
                cv.line(self.image_plot, (x1, y1), (x2, y2), color, 2)

            self.p_time = time.time()

        return self.image_plot

    def draw_background(self):
        """
        Draw the static background elements of the plot.
        """

        cv.rectangle(self.image_plot, (0, 0), (self.width, self.height), (0, 0, 0), cv.FILLED)
        cv.line(self.image_plot, (0, self.height // 2), (self.width, self.height // 2), (150, 150, 150), 2)

        # Draw grid lines and y-axis labels
        for x in range(0, self.width, 50):
            cv.line(self.image_plot, (x, 0), (x, self.height), (50, 50, 50), 1)
        for y in range(0, self.height, 50):
            cv.line(self.image_plot, (0, y), (self.width, y), (50, 50, 50), 1)
            y_label = int(self.y_limit[1] - ((y / 50) * ((self.y_limit[1] - self.y_limit[0]) / (self.height / 50))))
            cv.putText(self.image_plot, str(y_label), (10, y), cv.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)

        cv.putText(self.image_plot, self.char, (self.width - 100, self.height - 25), cv.FONT_HERSHEY_PLAIN, 5, (150, 150, 150), 5)


def main():
    plot = LivePlot(width=1200, y_limit=[-100, 100], interval=0.01)
    x = 0
    while True:

        x += 1
        if x == 360: x = 0
        image_plot = plot.update(int(math.sin(math.radians(x)) * 100))

        cv.imshow("Image", image_plot)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    main()
