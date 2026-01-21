"""
Supporting Functions for Computer vision using OpenCV
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import copy
import urllib.request

import cv2
import cv2 as cv
import numpy as np
from typing import Tuple

from enum import Enum, auto

class HAlign(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

class VAlign(Enum):
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()



class BoundingBox:
    def __init__(self, top_left_corner : Tuple[int, int], bottom_right_corner : Tuple[int, int]):
        x, y = top_left_corner
        x2, y2 = bottom_right_corner
        w = abs(x2 - x)
        h = abs(y2 - y)
        # Integer pixel center of the bounding box
        self.center = (
            (x + x2) // 2,
            (y + y2) // 2
        )
        self.box = (x, y, w, h)
        # Dimensions in pixels
        self.width = w
        self.height = h
        self.size = (w, h)
        self.origin = (x, y)
        self.opposite = (x2, y2)

    def draw(self, image, color=(0,127,0), thickness=2):
        cv.rectangle(
            image,
            self.origin,
            self.opposite,
            color,
            thickness
        )


def stackImages(_imgList, cols, scale):
    """
    Stack Images together to display in a single window
    :param _imgList: list of images to stack
    :param cols: the num of img in a row
    :param scale: bigger~1+ ans smaller~1-
    :return: Stacked Image
    """
    imgList = copy.deepcopy(_imgList)

    # Get dimensions of the first image
    width1, height1 = imgList[0].shape[1], imgList[0].shape[0]

    # make the array full by adding blank img, otherwise the openCV can't work
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

    # Create a blank image with dimensions of the first image
    imgBlank = np.zeros((height1, width1, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

    # resize the images to be the same as the first image and apply scaling
    for i in range(cols * rows):
        imgList[i] = cv.resize(imgList[i], (width1, height1), interpolation=cv.INTER_AREA)
        imgList[i] = cv.resize(imgList[i], (0, 0), None, scale, scale)

        if len(imgList[i].shape) == 2:  # Convert grayscale to color if necessary
            imgList[i] = cv.cvtColor(imgList[i], cv.COLOR_GRAY2BGR)

    # put the images in a board
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver


def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv.line(img, (x, y), (x + l, y), colorC, t)
    cv.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv.line(img, (x, y1), (x + l, y1), colorC, t)
    cv.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img


def findContours(img, imgPre, minArea=1000, maxArea=float('inf'), sort=True,
                 filter=None, drawCon=True, c=(255, 0, 0), ct=(255, 0, 255),
                 retrType=cv.RETR_EXTERNAL, approxType=cv.CHAIN_APPROX_NONE):
    """
    Finds Contours in an image.
    Sorts them based on area
    Can use filtration to get based on x corner points
    e.g. filter = [3,4] will return triangles and rectangles both

    :param img: Image on which we want to draw.
    :param imgPre: Image on which we want to find contours.
    :param minArea: Minimum Area to detect as valid contour.
    :param maxArea: Maximum Area to detect as valid contour.
    :param sort: True will sort the contours by area (biggest first).
    :param filter: List of filters based on the corner points e.g. [3, 4, 5].
                   If None, no filtering will be done.
    :param drawCon: Draw contours boolean.
    :param c: Color to draw the contours.
    :param ct: Color for Text
    :param retrType: Retrieval type for cv.findContours (default is cv.RETR_EXTERNAL).
    :param approxType: Approximation type for cv.findContours (default is cv.CHAIN_APPROX_NONE).

    :return: Found contours with [contours, Area, BoundingBox, Center].
    """
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv.findContours(imgPre, retrType, approxType)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if minArea < area < maxArea:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            if filter is None or len(approx) in filter:
                if drawCon:
                    cv.drawContours(imgContours, cnt, -1, c, 3)
                    x, y, w, h = cv.boundingRect(approx)
                    cv.putText(imgContours, str(len(approx)), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, ct, 2)
                cx, cy = x + (w // 2), y + (h // 2)
                cv.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                cv.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv.FILLED)
                conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True)

    return imgContours, conFound


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    """
     Overlay a PNG image with transparency onto another image using alpha blending.
     The function handles out-of-bound positions, including negative coordinates, by cropping
     the overlay image accordingly. Edges are smoothed using alpha blending.

     :param imgBack: The background image, a NumPy array of shape (height, width, 3) or (height, width, 4).
     :param imgFront: The foreground PNG image to overlay, a NumPy array of shape (height, width, 4).
     :param pos: A list specifying the x and y coordinates (in pixels) at which to overlay the image.
                 Can be negative or cause the overlay image to go out-of-bounds.
     :return: A new image with the overlay applied, a NumPy array of shape like `imgBack`.
     """
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape

    x1, y1 = max(pos[0], 0), max(pos[1], 0)
    x2, y2 = min(pos[0] + wf, wb), min(pos[1] + hf, hb)

    # For negative positions, change the starting position in the overlay image
    x1_overlay = 0 if pos[0] >= 0 else -pos[0]
    y1_overlay = 0 if pos[1] >= 0 else -pos[1]

    # Calculate the dimensions of the slice to overlay
    wf, hf = x2 - x1, y2 - y1

    # If overlay is completely outside background, return original background
    if wf <= 0 or hf <= 0:
        return imgBack

    # Extract the alpha channel from the foreground and create the inverse mask
    alpha = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
    inv_alpha = 1.0 - alpha

    # Extract the RGB channels from the foreground
    imgRGB = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

    # Alpha blend the foreground and background
    for c in range(0, 3):
        imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha

    return imgBack


def rotateImage(imgInput, angle, scale=1, keepSize=False):
    """
    Rotates an image around it's center while optionally keeping the original image dimensions.

    :param imgInput: The input image to be rotated. Should be an ndarray.
    :param angle: The angle by which the image is to be rotated. Should be a float.
    :param scale: A scaling factor that allows the image to be scaled while rotating. Default is 1. Optional.
    :param keepSize: If True, keeps the dimensions of the rotated image the same as the input.
                     If False, adjusts dimensions to fit the entire rotated image. Default is False. Optional.

    :return: The rotated image as an ndarray.

    Example:
        rotated_img = rotateImage(img, 90, keepSize=True)
    """
    # Get the dimensions of the input image (height and width)
    h, w = imgInput.shape[:2]

    # Calculate the center of the original image
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=scale)

    if keepSize:
        new_w = w
        new_h = h
    else:
        # Calculate the new dimensions of the image
        abs_cos = abs(rotate_matrix[0, 0])
        abs_sin = abs(rotate_matrix[0, 1])

        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the new dimensions
        rotate_matrix[0, 2] += new_w / 2 - center[0]
        rotate_matrix[1, 2] += new_h / 2 - center[1]

    # Perform the actual rotation and return the image
    imgOutput = cv.warpAffine(src=imgInput, M=rotate_matrix, dsize=(new_w, new_h))

    return imgOutput

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv.rectangle(img, (x1, y1), (x2, y2), colorR, cv.FILLED)
    if border is not None:
        cv.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


def align_top(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return 20 + baseline
def align_middle(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    center_y = - (text_height // 2)
    return center_y
def align_bottom(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return -(text_height + 20)
def justify_right(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return 20
def justify_left(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return -(text_width + 20)
def justify_center(text, font, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    center_x = - (text_width // 2)
    return center_x

def measure_text_block(lines, font, font_scale, thickness, margin=20):
    sizes = [cv.getTextSize(t, font, font_scale, thickness) for t in lines]

    widths  = [w for (w, h), _ in sizes]
    heights = [h for (w, h), _ in sizes]
    baselines = [b for (_, _), b in sizes]

    block_width  = max(widths)
    block_height = sum(heights) + sum(baselines) + margin * (len(lines) - 1)

    return block_width, block_height, sizes


def block_x_offset(block_width, align: HAlign, margin=15):
    match align:
        case HAlign.LEFT:
            return margin
        case HAlign.CENTER:
            return -block_width // 2
        case HAlign.RIGHT:
            return -(block_width + margin)


def block_y_offset(block_height, align: VAlign, margin=15):
    match align:
        case VAlign.TOP:
            return margin
        case VAlign.MIDDLE:
            return -block_height // 2
        case VAlign.BOTTOM:
            return -(block_height + margin)



def stack_text(
    image,
    lines,
    origin,
    font,
    font_scale,
    thickness,
    color,
    h_align: HAlign = HAlign.LEFT,
    v_align: VAlign = VAlign.TOP,
    margin=5
):
    block_w, block_h, sizes = measure_text_block(
        lines, font, font_scale, thickness, margin
    )

    ox = origin[0] + block_x_offset(block_w, h_align)
    oy = origin[1] + block_y_offset(block_h, v_align)

    # baseline for first line (inside top margin)
    first_h = sizes[0][0][1]
    y_cursor = oy + first_h

    for text, ((w, h), baseline) in zip(lines, sizes):

        # consistent justification for all lines
        match h_align:
            case HAlign.LEFT:
                x = ox
            case HAlign.CENTER:
                x = ox + (block_w - w) // 2
            case HAlign.RIGHT:
                x = ox + (block_w - w)

        cv.putText(
            image,
            text,
            (x, y_cursor),
            font,
            font_scale,
            color,
            thickness,
            cv.LINE_AA
        )

        # move DOWN for next line
        y_cursor += h + baseline + margin



def downloadImageFromUrl(url, keepTransparency=False):
    """
    Download an image from a given URL and return it as an OpenCV image.

    :param url: The URL of the image to download
    :param keep_transparency: Whether to keep the alpha channel (transparency) in the image (default: False)
    :return: The downloaded image in OpenCV format
    """
    # Download the image using urllib
    url_response = urllib.request.urlopen(url)

    # Convert the downloaded bytes to a numpy array
    image_data = np.asarray(bytearray(url_response.read()), dtype=np.uint8)

    # Decode the image data
    if keepTransparency:
        image = cv.imdecode(image_data, cv.IMREAD_UNCHANGED)
    else:
        image = cv.imdecode(image_data, cv.IMREAD_COLOR)

    return image


def main():
    cap = cv.VideoCapture(2)

    # ------ downloadImageFromUrl ------#
    imgPNG = downloadImageFromUrl(
        url='https://github.com/cvzone/cvzone/blob/master/Results/cvzoneLogo.png?raw=true',
        keepTransparency=True)

    imgShapes = downloadImageFromUrl(
        url='https://github.com/cvzone/cvzone/blob/master/Results/shapes.png?raw=true')

    while True:
        success, img = cap.read()

        # ------ putTextRect ------- #
        img, bbox = putTextRect(img, "CVZone", (50, 50),
                                scale=3, thickness=3,
                                colorT=(255, 255, 255), colorR=(255, 0, 255),
                                font=cv.FONT_HERSHEY_PLAIN, offset=10,
                                border=5, colorB=(0, 255, 0))
        # ------ cornerRect ------- #
        img = cornerRect(img, (200, 200, 300, 200),
                         l=30, t=5, rt=1,
                         colorR=(255, 0, 255), colorC=(0, 255, 0))

        # ------ rotateImage ------- #
        imgRotated60 = rotateImage(img, 60, scale=1, keepSize=False)
        imgRotated60KeepSize = rotateImage(img, 60, scale=1, keepSize=True)
        cv.imshow("imgRotated60", imgRotated60)
        cv.imshow("imgRotated60KeepSize", imgRotated60KeepSize)

        # ------ overlayPNG ------- #
        imgOverlay = overlayPNG(img, imgPNG, pos=[-30, 100])
        cv.imshow("imgOverlay", imgOverlay)

        # ------ findContours ------- #
        imgCanny = cv.Canny(imgShapes, 50, 150)
        imgDilated = cv.dilate(imgCanny, np.ones((5, 5), np.uint8), iterations=1)
        imgContours, conFound = findContours(imgShapes, imgDilated, minArea=1000, sort=True,
                                             filter=None, drawCon=True, c=(255, 0, 0), ct=(255, 0, 255),
                                             retrType=cv.RETR_EXTERNAL, approxType=cv.CHAIN_APPROX_NONE)
        imgContoursFiltered, conFoundFiltered = findContours(imgShapes, imgDilated, minArea=1000, sort=True,
                                                             filter=[3, 4], drawCon=True, c=(255, 0, 0),
                                                             ct=(255, 0, 255),
                                                             retrType=cv.RETR_EXTERNAL,
                                                             approxType=cv.CHAIN_APPROX_NONE)
        cv.imshow("imgContours", imgContours)
        cv.imshow("imgContoursFiltered", imgContoursFiltered)

        # ------ stackImages ------- #
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgList = [img, imgGray]
        imgStacked = stackImages(imgList, 2, 0.8)
        cv.imshow("stackedImg", imgStacked)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
