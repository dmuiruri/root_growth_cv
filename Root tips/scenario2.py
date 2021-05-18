# Image analysisis of two images
# Identify roots from binary images and collect found root tip informatation as location coordinates and length 
# (= diameter) into dictionary
# User can adjust how big are the tips they are searching.

# import functions 
from fileWITHfunctions import color_image, aux_analysis_1, aux_analysis_2,aux_analysis_3, main_analysis, create_file_list, check_date, find_contours, processImage, draw_circles_around, add_Text, process_ML_image, extract_data_DF, compareTips, check_file, extract_date

import pandas as pd
import sys
import cv2
import glob
from datetime import datetime
import numpy as np

# python scenario2.py hydescan1_T001_L001_2020.07.15_033029_362_DYY-prediction.jpg  hydescan1_T001_L001_2020.07.16_033029_363_DYY-prediction.jpg hydescan1_T001_L001_2020.07.15_033029_362_DYY.jpg 10  

""" 
STEP 1:
Read images as unchanged 1-channel gray image and convert them to binary

STEP 2:
Find contours and collect root tip information into the dictionaries

STEP 3: 
Create dataframes for further comparison. 
1. Check root tip diameters: diameter of combined image should be more or same.
2. Calculate distance between centre points, if it's less than 30 then root tips from IMAGE #1 and
combined image are same. 
3. Create a dataframe with root tips that have been found from IMAGE #1 and combine image (IMAGE #1 and IMAGE #2) 

STEP 4: 
Draw tips that exist in IMAGE #1 and in combined image

STEP 5:
Save dataframe to csv. Create image with root tips. 
"""



def main(im1_s, im2_s, im2_o, tip_size):

    period = check_date(im1_s, im2_s, im2_o)

    # Period is less than 1 --> mistake
    if period <1:
        print("Check filenames or order")
    # Period is 1 --> simple analysis of two images, OPTION 1
    elif period == 1:
        print("Analyze day #1 and day #2")
        main_analysis(im1_s, im2_s, im2_o, im_original, tip_size)
    # Period is more than 1 day --> OPTION 2
    else:
        print(f"Looking for tips from {period} days")
        image_list = create_file_list(im1_s, period, seg_files)
        counter = 2
        # Analyze firs pair of images and get combined image for further analysis and temporary dataframe
        # nex_day_image is binary combine image of day #1 and #2
        combined_image_b, combined_image_g, starting_df, temp_df = aux_analysis_1(image_list[0], image_list[1], tip_size)
        while counter < len(image_list):
            combined_image_b, combined_image_g, temp_df = aux_analysis_2(combined_image_b, combined_image_g, image_list[counter], tip_size, temp_df)
            counter += 1
            if counter == len(image_list)-1:
                print("Counter is ", counter, image_list[counter])
                # Call the third versio of image analysis since this is last file
                aux_analysis_3(combined_image_b, combined_image_g, image_list[counter], tip_size, temp_df, im2_o, starting_df)

                break
            else:
                pass



if __name__ == "__main__":

    if len(sys.argv)==5:
        # First image is processed image after root segmentation, IMAGE #1
        im1_s = sys.argv[1]

        # First image is processed image after root segmentation, IMAGE #2
        # Second image is original image for root tip drawing, IMAGE #2
        im2_s = sys.argv[2]
        im2_o = sys.argv[3]

        # Size of the root tip for analysis1.py 
        # 1 pixel = 0,1 mm
        tip_size = int(sys.argv[4])/0.1

        # Folders with images
        seg_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_result/'
        orig_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_data/'

        # Create correct pathnames
        im1_s = seg_files + im1_s
        im2_s = seg_files + im2_s
        im2_o = orig_files + im2_o

        if tip_size>1:
            print(f'Searching tips with diameter over: {int(tip_size)}px')
            main(im1_s, im2_s, im2_o, tip_size)
        else:
            print("Root tip size is too small")
    else:
        print("Check filenames or tip size")