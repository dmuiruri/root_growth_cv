# -*- coding: utf-8 -*-

import cv2
import math

# Contours is a curve joining all the continuous points (along the boundary), having same color or intensity. 
# Function find_contours search contours, order them and return the dictionary rootID:{file name, center coordinates, radius, diameter}
def find_contours(filename, image, mask, tip_size):
    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    
    # Sort contours from left to right
    cnts = sort_contours(cnts)[0]

    # loop over the contours and collect coordinates to the dictionary
    coordinates = {}
    imageID = 0
    tip_size = tip_size/2
    for c in cnts:
    # get the bright spot of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if int(radius) <= tip_size:  
            pass
        else:
            coordinates[imageID] = [filename, (int(cX), int(cY)), int(radius), int(radius)*2]
            imageID += 1
    return coordinates

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

# open original color image, resize it and cut borders
def processImage(filename):
    # Read color image (original)
    im = cv2.imread(filename)

    # New image size. With this 1 pixel = 0,1mm, when image is A4 size.
    resize_dim = (2970, 2100)

    # Resize image
    im = cv2.resize(im, resize_dim, interpolation = cv2.INTER_AREA)

    # Cut borders from image
    im = im[50:2100-50, 50:2970-50]
    return im
    
# draw circle around roots and mark center of the circle
def draw_circles_around(image, coordinates, radius, color):
    for row in range(len(coordinates)):
        cv2.circle(image, (coordinates[row][0],coordinates[row][1]), radius[row], color, 3)
        cv2.circle(image, (coordinates[row][0],coordinates[row][1]), 0, color, 3)


# add root ID according to center location 
def add_Text(image, coordinates, radius, color):
    i = 0
    for row in range(len(coordinates)):
        i += 1
        x, y, r = coordinates[row][0], coordinates[row][1], radius[row]
        if y < 50:
            cv2.putText(image, "#{}".format(i), (x+r, y+(3*r)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)
        elif x > 2800:
            cv2.putText(image, "#{}".format(i), (x-(2*r), y+(2*r)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)
        else:
            cv2.putText(image, "#{}".format(i), (x+r, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)

# open image after segmentation and create binary image
def process_ML_image(filename):
    # Read image as unchanged 1-channel gray image
    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Convert the imager to binary
    ret, im_binary = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)

    # Make areas fatter by 1 pixel (join roots where is holes in edges)
    im_binary = cv2.dilate(im_binary, None, iterations=5)
    im_gray = cv2.dilate(im_gray, None, iterations=5)
    return im_binary, im_gray

# Auxiliary functions

def check_Radius(r_1, r_2):
    diff = r_2-r_1
    if diff >= 0:
        return True
    else:
        return False

# check_distance() function to calculate distance between two points    
def check_distance(crd1, crd2):
    distance = math.sqrt(((crd1[0]-crd2[0])**2)+((crd1[1]-crd2[1])**2))
    if distance < 30:
        return True
    else:
        return False

# get dataframe coordinates and radius (IMAGE #1)
def extract_data_DF(df):
    return (df[1], df[2])

def compareTips(im1_crd, im2_crd, im1_r, im2_r, im1_id, image, image1, image2):
    # Check if radius is bigger or not and get differentce (= root growing change)
    # if status is True --> same root by radious
    # Calculate distance between center points is less than 30 --> same root
    if check_Radius(im1_r, im2_r) and check_distance(im1_crd, im2_crd):
        image.loc[im1_id,'Root #ID'] = im1_id
        image.loc[im1_id,'Tip lenght #1'] = im1_r*2
        image.loc[im1_id,'(x1,y1)'] = im1_crd
        image.loc[im1_id,'Radius #1'] = im1_r
        image.loc[im1_id,'Tip length #2'] = im2_r*2
        image.loc[im1_id,'(x2,y2)'] = im2_crd
        image.loc[im1_id,'Radius #2'] = im2_r
        image.loc[im1_id,'Difference, mm'] = ((im2_r-im1_r)*2)*0.1
    else:
        pass

        