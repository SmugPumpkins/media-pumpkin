import cv2
import media_pumpkin

# Initialize camera capture
cap = cv2.VideoCapture(2)

imgPNG = media_pumpkin.downloadImageFromUrl(
    url='https://github.com/cvzone/cvzone/blob/master/Results/cvzoneLogo.png?raw=true',
    keepTransparency=True)

while True:
    # Read image frame from camera
    success, img = cap.read()

    imgOverlay = media_pumpkin.overlayPNG(img, imgPNG, pos=[-30, 50])
    imgOverlay = media_pumpkin.overlayPNG(img, imgPNG, pos=[200, 200])
    imgOverlay = media_pumpkin.overlayPNG(img, imgPNG, pos=[500, 400])

    cv2.imshow("imgOverlay", imgOverlay)
    cv2.waitKey(1)