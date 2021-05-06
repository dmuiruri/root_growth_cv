# Image analysis of separate image
# Identify roots from a binary image and collect found root tip information as location coordinates and length (= diameter)
# into csv-file
# User can adjust how big are the tips they are searching. 

# import functions 
from fileWITHfunctions import find_contours, processImage, draw_circles_around, add_Text, process_ML_image

import pandas as pd
import numpy as np
import sys
import cv2

""" 
STEP 1:
Read the image as unchanged 1-channel gray image and convert it to binary

STEP 2:
Find contours and collect root tip information into the dictionary

STEP 3: 
Read original color image, resize it and draw found tips

STEP 4: 
Create dataframe and save it to csv-file
"""
def main(im_s, im_o, tip_size):

    # create binary image and grayscale image to find contours
    im_binary, im_gray = process_ML_image(im_s)

    # Process original color image for visualization
    im_original = processImage(im_o)

    # Calculate binary image colors, if root tips exist it should be 2
    unique, counts = np.unique(im_binary, return_counts=True)
    image_colors = len(dict(zip(unique, counts)))

    if image_colors == 2:
        # Find countours from the binary image and colletc them to the dictionary
        im_COORDINATES = find_contours(im_s, im_binary, im_gray, tip_size)
        # If root tips of given size exist

        if bool(im_COORDINATES): 
            # Create dataframe from dictionary 
            image_df = pd.DataFrame.from_dict(im_COORDINATES, orient='index', columns=['Image','Location (x,y)', 'Radius','Diameter'])

            # Draw circles around root tips
            # Circle and text color is green
            color = (0,255,0)
            draw_circles_around(im_original, list(image_df['Location (x,y)']),list(image_df['Radius']), color)

            # Numerate root tips
            add_Text(im_original, list(image_df['Location (x,y)']), list(image_df['Radius']), color)

            # Generate csv-file name
            csv_file = 'OPTION 0_TEST.csv'

            # Reindex dataframe 
            index_list = range(1,len(im_COORDINATES)+1)
            image_df['ImageID']=index_list
            image_df['Tip length, mm']=image_df['Diameter']*0.1
            image_df[['Image','Location (x,y)', 'ImageID', 'Tip length, mm']].to_csv(csv_file, sep=';', index=False, float_format='%.1f')


            print(image_df)


            # Save processed image
            cv2.imwrite('OPTION 0_TEST.png', im_original)
            print(f'Image with identified root tips is created, number of roots: {len(im_COORDINATES)}')
        else:
            print("No root tips")
    else:
        print("Empty image, no root tips")

if __name__ == "__main__":

    if len(sys.argv)==4:
        # The first image is the processed image after the segmentation step
        # The second image is the original image for the root tip drawing purpose
        im_s = sys.argv[1]
        im_o = sys.argv[2]

        # Size of the root tip for analysis1.py 
        # 1 pixel = 0,1 mm
        tip_size = int(sys.argv[3])/0.1

        # Folders with images
        seg_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_result/'
        orig_files = '20210428_root_growth_cv-main/DL_use_model/data/prediction_data/'

        # Create correct pathnames
        im_s = seg_files + im_s
        im_o = orig_files + im_o

        if tip_size>1:
            print(f'Searching tips with diameter over: {int(tip_size)}px')
            main(im_s, im_o, tip_size)
        else:
            print("Root tip size is too small")
    else:
        print("Root tip size or filename is missing")









