import os
import subprocess

import numpy as np
from PIL import Image
import cv2


from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
output_path = "/users/yabernar/GrosDisque/CDNET14/saliency"
Processing = SaliencyGenerator


categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
print("Categories :", categories)
for cat in categories:
    elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
    print("Elements of", cat, ":", elements)
    for elem in elements:
        print("Processing", elem, "...")
        current_path = os.path.join(cdnet_path, cat, elem)
        subprocess.run(["convert", "ROI.bmp", "ROI.png"], cwd=current_path)

