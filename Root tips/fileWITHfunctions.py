# -*- coding: utf-8 -*-

import cv2
import math
from datetime import datetime, timedelta
import glob
import numpy as np
import pandas as pd

# Final stage, draw root tip circle in original color image
def color_image(im2_o, result_DF):
    # Process original color images for visualization
    im2_original = processImage(im2_o)

    # Draw circles around root tips
    # Root tips from IMAGE #1 are green, root tips of combined image are red
    color_1 = (0,255,0)
    color_c = (0,0,255)
    draw_circles_around(im2_original, list(result_DF['(x1,y1)']),list(result_DF['Radius #1']), color_1)
    draw_circles_around(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

    # Numerate root tips
    add_Text(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

    return im2_original

# Final stage, last picture, draw root tip circle in original color image
def color_image_v2(im2_o, starting_df, result_DF):
    # Process original color images for visualization
    im2_original = processImage(im2_o)

    # Draw circles around root tips
    # Root tips from IMAGE #1 are green, root tips of combined image are red
    color_1 = (0,255,0)
    color_c = (0,0,255)
    draw_circles_around(im2_original, list(starting_df['(X,Y) coordinates']),list(starting_df['Radius']), color_1)
    draw_circles_around(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

    # Numerate root tips
    add_Text(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

    return im2_original

# If amount of images to be analized is bigger than 2
# Starting point, two images have to binarized
def aux_analysis_1(im1_s, im2_s, tip_size):
    print ("****", im1_s, im2_s, tip_size)

    im1_binary, im1_gray = process_ML_image(im1_s)
    im2_binary, im2_gray = process_ML_image(im2_s)

    # Calculate binary image colors, if roots exist it should be 2
    unique, counts = np.unique(im1_binary, return_counts=True)
    image_colors = len(dict(zip(unique, counts)))


    if image_colors == 2:
        # Create combined image from IMAGE #1 and IMAGE #2
        combined_image_b = im1_binary + im2_binary
        combined_image_g = im1_gray + im2_gray

        # Find countours from the binary images
        im1_COORDINATES = find_contours(im1_s, im1_binary, im1_gray, tip_size)
        combined_im_COORDINATES = find_contours('Combined image', combined_image_b, combined_image_g, tip_size)

        if bool(im1_COORDINATES): 
            # Create the dataframe from dictionaries 
            image1_df = pd.DataFrame.from_dict(im1_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])
            combined_im_df = pd.DataFrame.from_dict(combined_im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])
            
            # Create empty dataframe and fill it with final root ID, Day #1 root tip length, Day #1 + Day #2 root tip
            # lenght and calculated change.
            result_DF = pd.DataFrame(columns=['Root #ID', 'Tip lenght #1, mm', '(x1,y1)', 'Radius #1', 'Tip length #2, mm', '(x2,y2)', 'Radius #2','Difference, mm'])

            # Compare root tips and find same roots in image #1 and combined image
            for i in range(combined_im_df.shape[0]):
                # Get tip circle coodridnations and radius from combined image #1 + #2
                (combined_crd, combined_r) = extract_data_DF(combined_im_df.iloc[i,:])
                for j in range(image1_df.shape[0]): 
                    # Get tip circle coodridnations and radius from image #1
                    (im1_crd,im1_r) = extract_data_DF(image1_df.iloc[j,:])
                    compareTips(im1_crd, combined_crd, im1_r, combined_r, j, result_DF)
        else:
            print("No root tips of given size, try shorter period of time")
    else:
        print("No root tips in the image try shorter period of time")

    return combined_image_b, combined_image_g, image1_df, result_DF


# If amount of images to be analized is bigger than 2
# After first two image, only one image have to binarized
def aux_analysis_2(im1_binary, im1_gray, im2_s, tip_size, temp_df):
    print ("****",im2_s, tip_size)

    im2_binary, im2_gray = process_ML_image(im2_s)

    # Calculate binary image colors, if roots exist it should be 2
    unique, counts = np.unique(im2_binary, return_counts=True)
    image_colors = len(dict(zip(unique, counts)))

    if image_colors == 2:
        # Create combined image from IMAGE #1 and IMAGE #2
        combined_image_b = im1_binary + im2_binary
        combined_image_g = im1_gray + im2_gray

        # Find countours from the binary images of day #2
        combined_im_COORDINATES = find_contours('Combined image', combined_image_b, combined_image_g, tip_size)

        if bool(combined_im_COORDINATES): 
            # Create the dataframe from dictionaries 
            combined_im_df = pd.DataFrame.from_dict(combined_im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])

            # Create empty dataframe and fill it with final root ID, Day #1 root tip length, Day #1 + Day #2 root tip
            # lenght and calculated change.
            result_DF = pd.DataFrame(columns=['Root #ID', 'Tip lenght #1, mm', '(x1,y1)', 'Radius #1', 'Tip length #2, mm', '(x2,y2)', 'Radius #2','Difference, mm'])

            # Compare root tips and find same roots in image #1 and combined image
            for i in range(combined_im_df.shape[0]):
                # Get tip circle coodridnations and radius from combined image #1 + #2
                (combined_crd, combined_r) = extract_data_DF(combined_im_df.iloc[i,:])
                for j in range(temp_df.shape[0]): 
                    # Get tip circle coodridnations and radius from image #1
                    extract_data_temp_DF(temp_df.iloc[j,:])
                    (im1_crd,im1_r) = extract_data_temp_DF(temp_df.iloc[j,:])
                    compareTips(im1_crd, combined_crd, im1_r, combined_r, j, result_DF)
        else:
            print("No root tips of given size in the image, try shorter period of time")
    else:
        print("No root tips in the image, try shorter period of time")

    return combined_image_b, combined_image_g, result_DF

    # cv2.imshow('image',combined_image_b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# If amount of images to be analized is bigger than 2
# After first two image, only one image have to binarized
def aux_analysis_3(im1_binary, im1_gray, im2_s, tip_size, temp_df, im2_o, starting_df):

    im2_binary, im2_gray = process_ML_image(im2_s)

    # Calculate binary image colors, if roots exist it should be 2
    unique, counts = np.unique(im2_binary, return_counts=True)
    image_colors = len(dict(zip(unique, counts)))

    if image_colors == 2:
        # Create combined image from IMAGE #1 and IMAGE #2
        combined_image_b = im1_binary + im2_binary
        combined_image_g = im1_gray + im2_gray

        # Find countours from the binary images of day #2
        combined_im_COORDINATES = find_contours('Combined image', combined_image_b, combined_image_g, tip_size)

        if bool(combined_im_COORDINATES): 
            # Create the dataframe from dictionaries 
            combined_im_df = pd.DataFrame.from_dict(combined_im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])

            # Create empty dataframe and fill it with final root ID, Day #1 root tip length, Day #1 + Day #2 root tip
            # lenght and calculated change.
            result_DF = pd.DataFrame(columns=['Root #ID', 'Tip lenght #1, mm', '(x1,y1)', 'Radius #1', 'Tip length #2, mm', '(x2,y2)', 'Radius #2','Difference, mm'])

            # Compare root tips and find same roots in image #1 and combined image
            for i in range(combined_im_df.shape[0]):
                # Get tip circle coodridnations and radius from combined image #1 + #2
                (combined_crd, combined_r) = extract_data_DF(combined_im_df.iloc[i,:])
                for j in range(temp_df.shape[0]): 
                    # Get tip circle coodridnations and radius from image #1
                    extract_data_temp_DF(temp_df.iloc[j,:])
                    (im1_crd,im1_r) = extract_data_temp_DF(temp_df.iloc[j,:])
                    compareTips(im1_crd, combined_crd, im1_r, combined_r, j, result_DF)

            # print(result_DF)

            im_original = color_image_v2(im2_o, starting_df, result_DF)

            # Reindex root tips starting from #1 
            result_DF['Root #ID'] = result_DF['Root #ID'].apply(lambda x: x + 1)
            result_DF['Root #ID'] = result_DF['Root #ID'].astype('Int64')

            # Save dataframe to csv-file
            result_DF.to_csv('OPTION 2_TEST.csv', sep=';', index=False)

            # Save processed image
            cv2.imwrite("OPTION 2_TEST.png", im_original)
        else:
            print("No root tips of given size in the image, try shorter period of time")
    else:
        print("No root tips in the image, try shorter period of time")



# Two image analysis
def main_analysis(im1_s, im2_s, im2_o, tip_size):
    print (im1_s, im2_s, im2_o)

    im1_binary, im1_gray = process_ML_image(im1_s)
    im2_binary, im2_gray = process_ML_image(im2_s)

    # Calculate binary image colors, if roots exist it should be 2
    unique, counts = np.unique(im1_binary, return_counts=True)
    image_colors = len(dict(zip(unique, counts)))

    if image_colors == 2:
        # Create combined image from IMAGE #1 and IMAGE #2
        combined_image_b = im1_binary + im2_binary
        combined_image_g = im1_gray + im2_gray

        # Find countours from the binary images
        im1_COORDINATES = find_contours(im1_s, im1_binary, im1_gray, tip_size)
        combined_im_COORDINATES = find_contours('Combined image', combined_image_b, combined_image_g, tip_size)

        if bool(im1_COORDINATES) and bool(combined_im_COORDINATES): 

            # Create the dataframe from dictionaries 
            image1_df = pd.DataFrame.from_dict(im1_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])
            combined_im_df = pd.DataFrame.from_dict(combined_im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])

            # Create empty dataframe and fill it with final root ID, Day #1 root tip length, Day #1 + Day #2 root tip
            # lenght and calculated change.
            result_DF = pd.DataFrame(columns=['Root #ID', 'Tip lenght #1, mm', '(x1,y1)', 'Radius #1', 'Tip length #2, mm', '(x2,y2)', 'Radius #2','Difference, mm'])

            print(combined_im_df)
            print(image1_df)
            
            # Compare root tips and find same roots in image #1 and combined image
            for i in range(combined_im_df.shape[0]):
                # Get tip circle coodridnations and radius from combined image #1 + #2
                (combined_crd, combined_r) = extract_data_DF(combined_im_df.iloc[i,:])
                for j in range(image1_df.shape[0]): 
                    # Get tip circle coodridnations and radius from image #1
                    (im1_crd,im1_r) = extract_data_DF(image1_df.iloc[j,:])
                    print(combined_crd, combined_r, im1_crd,im1_r)
                    compareTips(im1_crd, combined_crd, im1_r, combined_r, j, result_DF)

            print(result_DF)

            im_original = color_image(im2_o, result_DF)

            # Reindex root tips starting from #1 
            result_DF['Root #ID'] = result_DF['Root #ID'].apply(lambda x: x + 1)
            result_DF['Root #ID'] = result_DF['Root #ID'].astype('Int64')

            # Save dataframe to csv-file
            result_DF.to_csv('OPTION 1_TEST.csv', sep=';', index=False)

            # Save processed image
            cv2.imwrite("OPTION 1_TEST.png", im_original)
        else:
            print("No root tips of given size")
    else:
        print("No root tips in image #1")

# If time period is bigger than 1, collect image filename to list
def create_file_list(im1_s, period, seg_files):
    days_list = create_days(im1_s, period)
    image_list = match_days(days_list, seg_files)
    return image_list
    
# Match day with filename and return image_list
def match_days(days_list, seg_files):
    # Extract path name from image filename
    image_list = []
    seg_path = seg_files + "*.jpg"
    seg_file_list = glob.glob(seg_path)
    for d in days_list:
        day = d.strftime('%Y.%m.%d')
        for f in seg_file_list:
            if day in f:
                image_list.append(f)
            else:
                pass
    return image_list
        
# Check if files exsist and return list of file need to be analyzed
def check_file(f, f_list):
    for i in range(len(f_list)):
        if f in f_list[i]:
            return True
        else:
            pass
    return False

# Create datetime object to calculate period of time
def extract_date(f):
    date = f.split('_')[9]
    date_DT = datetime.strptime(date, '%Y.%m.%d')
    return date_DT

# Create days list based on period of time and filename of image #1
def create_days(im1_s, period):
    days_list = []
    print ("im1_s", im1_s)
    print("period of time", period)
    start_day = extract_date(im1_s)
    days_list.append(start_day)
    print("First day: ", start_day)
    next_day = start_day
    while period > 0:
        next_day = next_day + timedelta(days=1)
        days_list.append(next_day)
        period -=1
    return days_list

# Check files, dates and return period of time
def check_date(im1_s, im2_s, im2_o):

    # Extract path name from image filename
    seg_path = im1_s.split('.')[:-1][0]+"*.jpg"
    orig_path  = im2_o.split('.')[:-1][0]+"*.jpg"

    seg_file_list = glob.glob(seg_path)
    orig_file_list = glob.glob(orig_path)

    # Check if files exist and calculate timedelta
    if check_file(im1_s.split('/')[-1],seg_file_list) and check_file(im2_s.split('/')[-1],seg_file_list)  and check_file(im2_o.split('/')[-1], orig_file_list):
        print("All 3 files exist")
        # Calculate timeperiod
        start_day = extract_date(im1_s)
        end_day = extract_date(im2_s)
        timeperiod = (end_day-start_day).days
    else:
        print("Some files are missing")
        timeperiod = 0
    return timeperiod

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
            cv2.putText(image, "#{}".format(i), (x-(2*r), y+(3*r)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=3)
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

# Check radius, r_2 should be bigger or equal --> same root tip, otherwise --> different root tip or other error in image
def check_Radius(r_1, r_2):
    diff = r_2-r_1
    if diff >= 0:
        return True
    else:
        return False

# check_distance() function to calculate distance between two points !!! 30 is 3mm, but was found empirically --> CAN BE ADJUSTED   
def check_distance(crd1, crd2):
    distance = math.sqrt(((crd1[0]-crd2[0])**2)+((crd1[1]-crd2[1])**2))
    if distance < 30:
        print(f"Distance between points: {distance} --> same root")
        return True
    else:
        return False

# get dataframe coordinates and radius (IMAGE #1)
def extract_data_DF(df):
    return (df[1], df[2])

# get dataframe coordinates and radius from dataframe of combined image
def extract_data_temp_DF(df):
    return (df[5], df[6])

# Compare two dataframe and fill resuld_DF dataframe where are only root tips from combined image that exist in image #1
def compareTips(im1_crd, im_crd, im1_r, im_r, im1_id, image):
    # Check if radius is bigger or not and get differentce (= root growing change)
    # if status is True --> same root by radious
    # Calculate distance between center points is less than 30 --> same root
    if check_Radius(im1_r, im_r) and check_distance(im1_crd, im_crd):
        image.loc[im1_id,'Root #ID'] = im1_id
        image.loc[im1_id,'Tip lenght #1, mm'] = (im1_r*2)*0.1
        image.loc[im1_id,'(x1,y1)'] = im1_crd
        image.loc[im1_id,'Radius #1'] = im1_r
        image.loc[im1_id,'Tip length #2, mm'] = (im_r*2)*0.1
        image.loc[im1_id,'(x2,y2)'] = im_crd
        image.loc[im1_id,'Radius #2'] = im_r
        image.loc[im1_id,'Difference, mm'] = ((im_r-im1_r)*2)*0.1
    else:
        print("No root tips to compare")

        