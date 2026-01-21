from media_pumpkin.HandTrackingModule import HandDetector
import cv2 as cv

# Initialize the webcam to capture video
cap = cv.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
hand_detector = HandDetector(2)

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

cap.release()
cv.destroyAllWindows()