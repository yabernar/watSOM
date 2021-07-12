import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt



from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"

plot = None

categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
print("Categories :", categories)
for cat in categories:
    elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
    print("Elements of", cat, ":", elements)
    for elem in elements:
        print("Processing", elem, "...")
        current_path = os.path.join(cdnet_path, cat, elem)
        bkg = Image.open(os.path.join(current_path, "input", "bkg.jpg"))
        if plot is None:
            plot = plt.imshow(bkg)
        else:
            plot.set_data(bkg)
        plt.waitforbuttonpress()

