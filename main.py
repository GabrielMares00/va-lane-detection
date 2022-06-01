import pickle
import cv2
import math
import matplotlib as mpl
from matplotlib.pyplot import gray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from moviepy.editor import VideoFileClip

lastBestKnownPosSlope = [1, 1]
lastBestKnownNegSlope = [-1, -1]


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = 255 * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, img


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    global lastBestKnownPosSlope
    global lastBestKnownNegSlope

    posSlopePoints = []
    negSlopePoints = []
    bestPosLineFitSlopeInt = []
    bestNegLineFitSlopeInt = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Find the Slope ((y2-y1)/(x2-x1))
            # If slope > 0, it should be the right lane line & if slope < 0, it should be left lane
            slope = ((y2 - y1) / (x2 - x1))
            if 0.85 > slope > 0.55:
                if not math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2):
                    posSlopePoints.append([x1, y1])
                    posSlopePoints.append([x2, y2])
            elif -0.80 < slope < -0.5:
                if not math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2):
                    negSlopePoints.append([x1, y1])
                    negSlopePoints.append([x2, y2])

    # print("Positive slope line points: ", posSlopePoints)
    # print("Negative slope line points: ", negSlopePoints)

    posSlopeXs = []
    posSlopeYs = []
    negSlopeXs = []
    negSlopeYs = []

    posSlopeXs = [pair[0] for pair in posSlopePoints]
    posSlopeYs = [pair[1] for pair in posSlopePoints]
    negSlopeXs = [pair[0] for pair in negSlopePoints]
    negSlopeYs = [pair[1] for pair in negSlopePoints]

    try:
        # Get the best line fit through the available points and store the slope & intercept of this line for both left & right lanes
        bestPosLineFitSlopeInt = np.polyfit(posSlopeXs, posSlopeYs, 1)
    except Exception as e:
        if lastBestKnownPosSlope is not None:
            bestPosLineFitSlopeInt = lastBestKnownPosSlope
        else:
            bestPosLineFitSlopeInt = [0, 0]
    try:
        bestNegLineFitSlopeInt = np.polyfit(negSlopeXs, negSlopeYs, 1)
    except Exception as e:

        if lastBestKnownNegSlope is not None:
            bestNegLineFitSlopeInt = lastBestKnownNegSlope
        else:
            bestNegLineFitSlopeInt = [0, 0]

    lastBestKnownPosSlope = bestPosLineFitSlopeInt
    lastBestKnownNegSlope = bestNegLineFitSlopeInt

    # Once we have line which is the best available fit through all the lines, extend this line to the ROI mask edges
    # Extended Left lane line bottom co-ordinates
    leftby = imgheight  # Y coordinate from ROI mask
    # X coordinate = (y-c)/m
    leftbx = (leftby - bestNegLineFitSlopeInt[1]) / bestNegLineFitSlopeInt[0]

    # Extended Left lane line top co-ordinates
    leftty = 0.62 * imgheight  # Y coordinate from ROI mask
    # X coordinate = (y-c)/m
    lefttx = (leftty - bestNegLineFitSlopeInt[1]) / bestNegLineFitSlopeInt[0]

    # Extended right lane line bottom co-ordinates
    rightby = imgheight  # Y coordinate from ROI mask
    # X coordinate = (y-c)/m
    rightbx = (rightby - bestPosLineFitSlopeInt[1]) / bestPosLineFitSlopeInt[0]

    # Extended right lane line top co-ordinates
    rightty = 0.62 * imgheight  # Y coordinate from ROI mask
    # X coordinate = (y-c)/m
    righttx = (rightty - bestPosLineFitSlopeInt[1]) / bestPosLineFitSlopeInt[0]

    cv2.line(img, (int(leftbx), int(leftby)),
             (int(lefttx), int(leftty)), color, thickness)
    #    plt.figure()
    #    plt.imshow(img)  #For debug
    cv2.line(img, (int(rightbx), int(rightby)),
             (int(righttx), int(rightty)), color, thickness)
    # plt.imshow(img)  #For debug


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, img2):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # print(lines)
    if lines is None:
        # cv2.imshow("Hough Lines",img)
        # cv2.imshow("Original Image",img2)
        # cv2.waitKey(0)
        lines = []
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def process_static_image(image):
    # import Camera Calibration Parameters
    dist_pickle = "wide_dist_pickle.p"
    with open(dist_pickle, mode="rb") as f:
        CalData = pickle.load(f)
    mtx, dist = CalData["mtx"], CalData["dist"]
    # undistort image for better accuracy
    undist_img = cv2.undistort(image, mtx, dist, None, mtx)
    # Stil confused if it worked
    image = undist_img

    gray = grayscale(image)

    # define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    # edges = canny(blur_hsv, low_threshold, high_threshold)
    edges = canny(blur_gray, low_threshold, high_threshold)

    # defining a four sided polygon to create a mask
    global imgheight
    global imgwidth
    imgheight = image.shape[0]
    imgwidth = image.shape[1]
    vertices = np.array([[(0.05 * imgwidth, imgheight), (0.48 * imgwidth, 0.62 * imgheight),
                          (0.55 * imgwidth, 0.62 * imgheight), (0.95 * imgwidth, imgheight)]], dtype=np.int32)
    masked_edges, img = region_of_interest(edges, vertices)

    # define the Hough transform parameters
    rho = 2
    theta = np.pi / 180
    threshold = 40
    min_line_len = 10
    max_line_gap = 50

    hough_lines_img = hough_lines(
        masked_edges, rho, theta, threshold, min_line_len, max_line_gap, img)
    line_marked_img = weighted_img(hough_lines_img, image)
    return line_marked_img

def video_init():
    output = 'lane_detect_output\\processedVideo.mp4'
    to_be_processed_video = VideoFileClip("roadlaneimg\\video4.mp4")
    clip = to_be_processed_video.fl_image(process_static_image)
    clip.write_videofile(output, audio=False)
    

video_init()
