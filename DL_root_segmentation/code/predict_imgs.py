# Read libraries
from pathlib import Path
from PIL import Image
import torch
import torchvision
from skimage.morphology import erosion
import matplotlib.pyplot as plt
import time

# Import scripts
from dataloader import pad_pair_256, normalize
from model import SegRoot

# Initialize paths and parameters
weights_path = "../weights/best_segnet-(8,5).pt" # Path to trained model weights
width=8 # Width of SegRoot
depth=5 # Depth of SegRoot
threshold=0.9 # threshold of the final binarization
read_dir = Path("../data/prediction_data") # Directory to test data
result_dir = "../data/prediction_result" # Directory to test data

def pad_256(img_path):
    image = Image.open(img_path)
    W, H = image.size
    img, _ = pad_pair_256(image, image)
    NW, NH = img.size
    img = torchvision.transforms.ToTensor()(img)
    img = normalize(img)
    return img, (H, W, NH, NW)

# Predict
def predict(model, test_img, device):
    # For testing do not use gradient
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    # test_img.shape = (3, 2304, 2560)
    test_img = test_img.unsqueeze(0)
    output = model(test_img)
    # output.shape = (1, 1, 2304, 2560)
    output = torch.squeeze(output)
    torch.cuda.empty_cache()
    return output


def predict_gen(model, img_path, thres, device, result_dir):
    # Get img dimensions
    img, dims = pad_256(img_path)
    H, W, NH, NW = dims
    # Img to device and predict
    img = img.to(device)
    prediction = predict(model, img, device)
    # Threshold prediction to make it binary
    prediction[prediction >= thres] = 1.0
    prediction[prediction < thres] = 0.0
    if device.type == "cpu":
        prediction = prediction.detach().numpy()
    else:
        prediction = prediction.cpu().detach().numpy()
    prediction = erosion(prediction)
    # reverse padding
    prediction = prediction[
        (NH - H) // 2 : (NH - H) // 2 + H, (NW - W) // 2 : (NW - W) // 2 + W
    ]

    # Set prediction image name and path
    new_file_name = img_path.parts[-1].split(".jpg")[0] + "-prediction.jpg"
    save_dir = result_dir + "/" + new_file_name

    # Save prediction image
    plt.imsave(save_dir, prediction, cmap="gray")
    print("{} generated!".format(new_file_name))


if __name__ == "__main__":
    # If available, use GPU, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegRoot(8, 5).to(device)
    if device.type == "cpu":
        print("No Cuda available, will use CPU")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Use Cuda CPU")
        model.load_state_dict(torch.load(weights_path))

    # Predict images
    img_paths = read_dir.glob("*.jpg")
    for img_path in img_paths:
        start_time = time.time()
        predict_gen(model, img_path, threshold, device, result_dir)
        end_time = time.time()
        print("{:.4f}s for one image".format(end_time - start_time))

