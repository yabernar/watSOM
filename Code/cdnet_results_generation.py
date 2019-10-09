import os
import numpy as np

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
        os.makedirs(os.path.join(output_path, cat, elem), exist_ok=True)
        current_path = os.path.join(cdnet_path, cat, elem)
        roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
        temporal_roi = (int(roi_file[0]), int(roi_file[1]))
        Processing(os.path.join(current_path, "input"), os.path.join(output_path, cat, elem), temporal_roi)

