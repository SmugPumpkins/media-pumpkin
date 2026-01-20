import cv2 as cv
import numpy as np
import time



class PID:
    def __init__(self, pid_values, target_value, axis=0, limit=None):
        self.pid_values = pid_values
        self.target_value = target_value
        self.axis = axis
        self.proportional_error = 0
        self.limit = limit
        self.integral = 0
        self.proportional_time = 0

    def update(self, current_value):
        # Current Value - Target Value
        t = time.time() - self.proportional_time
        error = current_value - self.target_value
        proportion = self.pid_values[0] * error
        self.integral = self.integral + (self.pid_values[1] * error * t)
        derivative = (self.pid_values[2] * (error - self.proportional_error)) / t

        result = proportion + self.integral + derivative

        if self.limit is not None:
            result = float(np.clip(result, self.limit[0], self.limit[1]))
        self.proportional_error = error
        self.proportional_time = time.time()

        return result

    def draw(self, image, center):
        image_height, image_width, _ = image.shape
        if self.axis == 0:
            cv.line(image, (self.target_value, 0), (self.target_value, image_height), (255, 0, 255), 1)
            cv.line(image, (self.target_value, center[1]), (center[0], center[1]), (255, 0, 255), 1, 0)
        else:
            cv.line(image, (0, self.target_value), (image_width, self.target_value), (255, 0, 255), 1)
            cv.line(image, (center[0], self.target_value), (center[0], center[1]), (255, 0, 255), 1, 0)

        cv.circle(image, tuple(center), 5, (255, 0, 255), cv.FILLED)

        return image


def main():
    from media_pumpkin.FaceDetectionModule import FaceDetector
    cap = cv.VideoCapture(0)
    detector = FaceDetector(min_detection_con=0.8)
    # For a 640x480 image center target is 320 and 240
    x_pid = PID([1, 0.000000000001, 1], 640 // 2)
    y_pid = PID([1, 0.000000000001, 1], 480 // 2, axis=1, limit=[-100, 100])

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img, faces = detector.find_faces(img)
        if faces:
            x, y, w, h = faces[0].bounding_box
            center_x, center_y = faces[0].center
            x_value = int(x_pid.update(center_x))
            y_value = int(y_pid.update(center_y))

            x_pid.draw(img, [center_x, center_y])
            y_pid.draw(img, [center_x, center_y])

            cv.putText(img, f'x:{x_value} , y:{y_value} ', (x, y - 100), cv.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

        cv.imshow("Image", img)
        # Wait for 1 ms to show this frame, then continue to the next frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
