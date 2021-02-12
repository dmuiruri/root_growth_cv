#! /usr/bin/env pythonA

"""This script contains helper functions for pre-processing image
files in a Computer Vision project, sample images can be found in
sample_images folder.

How to import the module: 
import preprocessor

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_mean, threshold_isodata, threshold_li # li: too noisy
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from skimage.filters.rank import enhance_contrast
from skimage import exposure, color, filters, io
from skimage.transform import rescale, resize, downscale_local_mean
from os import listdir

def pre_process_img(img_file, filter_algo='frangi', sigmas_range=range(1,10,1),
                    min_size=300, alpha=0.5, beta=0.5, gamma=5, mode='reflect'):
  """Pre-process images.

  img_file: image file to be processed
  filter_algo: frangi, meijering, sato, hessian
  sigmas: range of sigmas
  min_size: minimum size of  objects to be eliminated from a binary image

  Returns a binary image (rather its matrix representation). A
  relevant cmap parameter(cmap.cm.binary_r) for example can be used to
  plot the resulting image.

  Example:
  img_cv2 = cv2.imread('./<mypath>/mypic.jpeg')
  img_bin = preprocessor.pre_process_img(img_cv2)
  fig, ax = plt.subplots()
  ax.imshow(img_bin, cmap=plt.cm.binary_r)
  plt.show()

  """
  # pick an image
  if (type(img_file) != np.ndarray):
      raise TypeError('img file type should be a numpy ndarray')
  img_file = img_file[600:3000, 200:4400] # trim scanner edges
  img_trimmed = color.rgb2gray(img_file)
  img_rescaled = rescale(img_trimmed, 0.25, anti_aliasing=False)
  
  # improve contrast
  p1, p2 = np.percentile(img_rescaled, (2, 98))
  img_enhc = exposure.rescale_intensity(img_rescaled, in_range=(p1, p2))

  if filter_algo == 'frangi':
    # Defaults: alpha=0.5, beta=0.5, gamma=0.5
    print('frangi..')
    img_ridges = frangi(img_enhc, sigmas=sigmas_range, alpha=alpha,
                        beta=beta, gamma=gamma, black_ridges=True, cval=0)
  elif filter_algo == 'meijering':
    # Defaults: alpha=None, mode='reflect'
    print('meijering..')
    img_ridges = meijering(img_enhc, sigmas=sigmas_range, alpha=alpha,
                           black_ridges=True, mode=mode, cval=0)
  elif filter_algo == 'sato':
    print('sato...')
    img_ridges = sato(img_enhc, sigmas=sigmas_range, black_ridges=True,
                      mode=None, cval=0)
  elif filter_algo == 'hessian':
    # Defaults: alpha=0.5, beta=0.5, gamma=15, mode=None
    print('hessian...')
    img_ridges = hessian(img_enhc, sigmas=sigmas_range, scale_range=None,
                         scale_step=None, alpha=alpha, beta=beta, gamma=gamma,
                         black_ridges=True, mode=mode, cval=0)

  # binarize image
  thresh = threshold_otsu(img_ridges)
  img_bin = img_ridges > thresh

  # remove small objects
  img_cleaned = remove_small_objects(img_bin, min_size=min_size)

  return img_cleaned


if __name__ == '__main__':
    # Test code
    dir_path = './sample_images/'
    bin_imgs = list()
    orig_imgs = listdir(dir_path)

    fig, ax = plt.subplots(len(orig_imgs)-1, 2, figsize=(5, 5))
    # fig.subplots_adjust(hspace=0.05, wspace=0.025)
    for i, f in enumerate(orig_imgs):
        if f.endswith('.jpg'):
            orig_img = io.imread(dir_path+f)
            print(dir_path+f)
            bin_img = pre_process_img(orig_img)
            _ = ax[i,0].imshow(orig_img)
            _ = ax[i,1].imshow(bin_img, cmap=plt.cm.binary_r)
            fname = f.split('_')
            ax[i,0].set_title(fname[0] + '_' + fname[-2] + '_orig', fontsize=5)
            ax[i,1].set_title(fname[0] + '_' + fname[-2] + '_bin', fontsize=5)
    fig.tight_layout()
    plt.show()
