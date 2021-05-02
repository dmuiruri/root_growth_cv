# Image analysis of separate Image
# Identify roots from a binary image and collect found root tip information as location coordinates and length (= diameter)
# into csv-file
# User can adjust how big are the tips they are searching. 

# import functions 
from fileWITHfunctions import find_contours, processImage, draw_circles_around, add_Text, process_ML_image

import pandas as pd
import sys
import cv2

# The first image is the processed image after the root segmentation
# The second image is the original image for the root tip drawing
im_s = sys.argv[1]
im_o = sys.argv[2]

# Size of the root tip for analysis1.py 
# 1 pixel = 0,1 mm
tip_size = int(sys.argv[3])/0.1

print(f'Searching tips with diameter over: {int(tip_size)}px')

""" 
STEP 1:
Read the image as unchanged 1-channel gray image and convert it to binary

STEP 2:
Open original color image and resize it.

STEP 3: 
Find contours and collect root tip information into the dictionary. 

STEP 4: 
Create dataframe, draw contours and save root tip information to csv-file

STEP 5:
Save image with marked tips
"""
im_binary, im_gray = process_ML_image(im_s)

# Process original color image for visualization
im_original = processImage(im_o)

# Find countours from the binary image
im_COORDINATES = find_contours(im_s, im_binary, im_gray, tip_size)

# Create the dataframe from dictionary 
image_df = pd.DataFrame.from_dict(im_COORDINATES, orient='index', columns=['Image','(X,Y) coordinates', 'Radius','Diameter'])

# Draw circles around root tips
# Circle and text color is green
color = (0,255,0)
draw_circles_around(im_original, list(image_df['(X,Y) coordinates']),list(image_df['Radius']), color)

# Identify root tips
add_Text(im_original, list(image_df['(X,Y) coordinates']), list(image_df['Radius']), color)

# Generate csv-file name
csv_file = im_s.split('.')[0] + '.csv'

# Reindex dataframe 
index_list = range(1,len(im_COORDINATES)+1)
image_df['ImageID']=index_list
image_df.to_csv(csv_file, sep=';', index=False)

# Save processed image
png_file = im_s.split('.')[0] + '.png'
cv2.imwrite(png_file, im_original)
