# Image analysis of two images
# Identify roots from binary images and collect found root tip informatation as location coordinates and length 
# (= diameter) into dictionary
# User can adjust how big are the tips they are searching.

# import functions
from root_tips.fileWITHfunctions import find_contours, processImage, draw_circles_around, add_Text, process_ML_image, extract_data_DF, compareTips

import logging
import pandas as pd
import sys
import cv2


# python scenario2.py test_prediction_1.jpg test_original_1.jpg test_prediction_2.jpg test_original_2.jpg 10

# First image is processed image after root segmentation, IMAGE #1
# Second image is original image for root tip drawing, IMAGE #1
##im1_s = sys.argv[1]

# First image is processed image after root segmentation, IMAGE #2
# Second image is original image for root tip drawing, IMAGE #2
##im2_s = sys.argv[2]
##im2_o = sys.argv[3]

# Size of the root tip for analysis
# 1 pixel = 0,1 mm
##tip_size = int(sys.argv[4])/0.1

##print(f'Searching tips with diameter over: {int(tip_size)}px')

def root_tip_analysis(im1_s, im2_s, im2_o, tip_size, img_dir, csv_dir, img_date):
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
    tip_size = int(tip_size)/0.1    # 1 pixel = 0,1 mm

    im1_binary, im1_gray = process_ML_image(im1_s)
    im2_binary, im2_gray = process_ML_image(im2_s)

    # Create combined image from IMAGE #1 and IMAGE #2
    combined_image_b = im1_binary + im2_binary
    combined_image_g = im1_gray + im2_gray

    # Find countours from the binary images
    im1_COORDINATES = find_contours(im1_s, im1_binary, im1_gray, tip_size)
    combined_im_COORDINATES = find_contours('Combined image', combined_image_b, combined_image_g, tip_size)

    # Create the dataframe from dictionaries
    image1_df = pd.DataFrame.from_dict(im1_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])
    combined_im_df = pd.DataFrame.from_dict(combined_im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])

    # Create empty dataframe and fill it with final root ID, Day #1 root tip length, Day #1 + Day #2 root tip
    # lenght and calculated change.
    result_DF = pd.DataFrame(columns=['Root #ID', 'Tip length #1', '(x1,y1)', 'Radius #1', 'Tip length #2', '(x2,y2)', 'Radius #2','Difference, mm'])

    # Process original color images for visualization
    im2_original = processImage(im2_o)

    # Compare root tips and find same roots in image #1 and combined image
    for i in range(len(combined_im_COORDINATES )):
        # Get #ID, tip center location and radius of combined image
        im_id = i
        im_crd = combined_im_COORDINATES [i][1]
        im_r = combined_im_COORDINATES [i][2]
        # Get #ID, tip center location and radius of IMAGE #1
        for j in range(image1_df.shape[0]):
            # Get coordinates and radiou from dataframe
            (im1_crd,im1_r) = extract_data_DF(image1_df.iloc[j,:])
            im1_id = j
            try:
                compareTips(im1_crd, im_crd, im1_r, im_r, im1_id, result_DF, image1_df, combined_im_df)
            except:
                logging.exception('ERROR! PLACEHOLDER EXCEPT-PASS TO TEST UI')
                pass

    try:
        # Draw circles around root tips
        # Root tips from IMAGE #1 are green, root tips of combined image are red
        color_1 = (0,255,0)
        color_c = (0,0,255)
        draw_circles_around(im2_original, list(result_DF['(x1,y1)']),list(result_DF['Radius #1']), color_1)
        draw_circles_around(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

        # Identify root tips
        add_Text(im2_original, list(result_DF['(x2,y2)']),list(result_DF['Radius #2']), color_c)

        # Reindex root tips starting from #1
        result_DF['Root #ID'] = result_DF['Root #ID'].apply(lambda x: x + 1)
        result_DF['Root #ID'] = result_DF['Root #ID'].astype('Int64')

        # # Save dataframe to csv-file
        csv_out = f'{csv_dir}/{img_date}.csv'
        result_DF.to_csv(csv_out, sep=';', index=False)

        # # Save processed images
        img_out = f'{img_dir}/{img_date}.png'
        cv2.imwrite(img_out, im2_original)

        return result_DF
    except:
        logging.exception('ERROR! PLACEHOLDER EXCEPT-PASS TO TEST UI')
        return None
