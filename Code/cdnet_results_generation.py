import os
import numpy as np
from PIL import Image
import cv2


from Code.saliency_generator import SaliencyGenerator

np.set_printoptions(threshold=np.inf)

cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
output_path = "/users/yabernar/GrosDisque/CDNET14/saliency_as"
Processing = SaliencyGenerator


categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
print("Categories :", categories)
for cat in categories:
    elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
    print("Elements of", cat, ":", elements)
    for elem in elements:
        print("Processing", elem, "...")
        os.makedirs(os.path.join(output_path, 'results', cat, elem), exist_ok=True)
        current_path = os.path.join(cdnet_path, cat, elem)
        roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
        temporal_roi = (int(roi_file[0]), int(roi_file[1]))
        mask_roi = Image.open(os.path.join(current_path, "ROI.png"))
        p = Processing(os.path.join(current_path, "input"), os.path.join(output_path, 'results', cat, elem),
                       os.path.join(output_path, 'supplements', cat, elem), temporal_roi, mask_roi)
        p.generate_all()
