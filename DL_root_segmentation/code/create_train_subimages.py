from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import numpy as np
from dataloader import pad_pair_256


# Set paths
data_dir = Path('../data/train_images')
mask_dir = Path('../data/train_masks')
subimg_path = Path('../data/train_subimg')
subimg_path.mkdir(exist_ok=True, parents=True)
submask_path = Path('../data/train_submask')
submask_path.mkdir(exist_ok=True, parents=True)

# Sort paths in folder and print count
imgs = sorted(list(data_dir.glob('*.jpg')))
print('Number of images : ', len(imgs))
masks = sorted(list(mask_dir.glob('*.jpg')))
print('Number of masks : ', len(masks), "\n")

# Save idex info in a dictionary
info_dict = {k: v.parts[-1] for k, v in enumerate(imgs)}
print(info_dict)
with open('../data/info.pkl', 'wb') as handle:
    pickle.dump(info_dict, handle)
print('Index info saved to data/info.pkl ! \n')

# -- Create subimages and save to them --

# Create sub images
for idx, (mask_path, img_path) in enumerate(zip(masks, imgs)):
    mask = Image.open(mask_path)
    img = Image.open(img_path)
    new_img, new_mask = pad_pair_256(img, mask)
    new_img, new_mask = np.array(new_img), np.array(new_mask)
    # padded shape (2560, 2304)
    w, h, _ = new_img.shape
    for i in range(int(w/256)):
        for j in range(int(h/256)):
            subimg = new_img[i*256:(i+1)*256, j*256:(j+1)*256, :]
            subimg_fn = '{}/{}-{}-{}.png'.format(
                Path('../data/train_subimg').as_posix(), idx, i, j)
            plt.imsave(subimg_fn, subimg)
            submask_fn = '{}/{}-{}-{}.png'.format(
                Path('../data/train_submask').as_posix(), idx, i, j)
            submask = new_mask[i*256:(i+1)*256, j*256:(j+1)*256]
            plt.imsave(submask_fn, submask, cmap='gray')
    print('No.{} sub_imgs and sub_masks created !'.format(idx))