from media_pumpkin.PlotModule import LivePlot
from media_pumpkin.FaceDetectionModule import FaceDetector
import cv2
import media_pumpkin
import math

cap = cv2.VideoCapture(2)
detector = FaceDetector(min_detection_con=0.85, model_selection=0)

xPlot = LivePlot(width=1200, y_limit=[0, 500], interval=0.01)
sinPlot = LivePlot(width=1200, y_limit=[-100, 100], interval=0.01)
xSin=0



while True:
    success, img = cap.read()

    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.find_faces(img, draw=False)
    val = 0
    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            val = center[0]
            # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            media_pumpkin.putTextRect(img, f'{score}%', (x, y - 10))
            media_pumpkin.cornerRect(img, (x, y, w, h))

    xSin += 1
    if xSin == 360: xSin = 0
    imgPlotSin = sinPlot.update(int(math.sin(math.radians(xSin)) * 100))
    imgPlot = xPlot.update(val)


    cv2.imshow("Image Plot", imgPlot)
    cv2.imshow("Image Sin Plot", imgPlotSin)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
