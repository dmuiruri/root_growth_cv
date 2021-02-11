#! /usr/bin/env python

"""This script contains helper functions for pre-processing image
files in a Computer Vision

"""
from skimage.filters import meijering
from skimage.filters import threshold_otsu, threshold_mean, threshold_isodata, threshold_li # li: too noisy
from skimage.filters import meijering, sato, frangi, hessian # ridge filters
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from skimage.filters.rank import enhance_contrast
from skimage import exposure
import os

def pre_process_img(img_file, filter_algo='frangi', sigmas_range=range(1,10,1), min_size=350):
  """
  Pre-process images.

  img_file: image file to be processed
  filter_algo: ridge detection algorithm
  sigmas: range of sigmas TODO: Not sure how this is actually applied
  min_size: minimum size of  objects to be eliminated from a binary image
  """
  # pick an image
  img_trimmed = color.rgb2gray(img_file)[600:3000, 200:4400] # trim edges
  img_rescaled = rescale(img_trimmed, 0.25, anti_aliasing=False)
  
  # improve contrast
  p1, p2 = np.percentile(img_rescaled, (2, 98))
  img_enhc = exposure.rescale_intensity(img_rescaled, in_range=(p1, p2))

  # detect ridges
  if filter_algo == 'frangi':
    print('frangi..')
    img_ridges = frangi(img_enhc, sigmas=sigmas_range, alpha=0.5,
                        beta=0.5, gamma=5, black_ridges=True, cval=0)
  elif filter_algo == 'meijering':
    print('meijering..')
    img_ridges = meijering(img_enhc, sigmas=sigmas_range, alpha=None,
                           black_ridges=True, mode='reflect', cval=0)
  elif filter_algo == 'sato':
    print('sato...')
    img_ridges = sato(img_enhc, sigmas=sigmas_range, black_ridges=True,
                      mode=None, cval=0)
  elif filter_algo == 'hessian':
    print('hessian...')
    img_ridges = hessian(img_enhc, sigmas=sigmas_range, scale_range=None,
                         scale_step=None, alpha=0.5, beta=0.5, gamma=15,
                         black_ridges=True, mode=None, cval=0)

  # binarize image
  thresh = threshold_otsu(img_ridges)
  img_bin = img_ridges > thresh

  # remove small objects
  img_cleaned = remove_small_objects(img_bin, min_size=370)

  return img_cleaned


if __name__ == '__main__':
    # Test code here
