import cv2
import glob
import os
from tqdm import tqdm

# Read ans save folder
image_folder = "../data/train_images_and_masks"

# New image size. With this 1 pixel = 1mm, when image is A4 size.
resize_dim = (2670, 1890)

# Crop parameters
start_crop_h = 165
start_crop_w = 234
crop_h = 2975
crop_w = 4208

# Get image names
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Sort image names
images = sorted(images)

for image in tqdm(images):
    # Read image
    im = cv2.imread(os.path.join(image_folder, image))
    # Get image size
    im_h, im_w, im_dim = im.shape
    # If image is small, then do not crop
    if im_h > crop_h:
        # Cut borders from image
        im = im[start_crop_h:start_crop_h+crop_h, start_crop_w:start_crop_w+crop_w]
    # Resize image
    im = cv2.resize(im, resize_dim, interpolation = cv2.INTER_AREA)
    # Get image name
    im_name = os.path.join(image_folder, image)
    # Save image
    cv2.imwrite(im_name, im)

print("Image cropping and resizing ready!")