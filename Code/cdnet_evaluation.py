import os
import subprocess

import numpy as np

from PIL import Image

BLACK = 0
SHADOW = 50
OUTOFSCOPE = 85
UNKNOWN = 170
WHITE = 255

class Comparator:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.nbShadowErrors = 0

    def evaluate_all(self, cdnet_path, output_path, categories_list):
        fmeasures = []
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in list(categories[i] for i in categories_list):
            elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
            for elem in elements:
                os.makedirs(os.path.join(output_path, 'results', cat, elem), exist_ok=True)
                current_path = os.path.join(cdnet_path, cat, elem)
                roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
                temporal_roi = (int(roi_file[0]), int(roi_file[1]))
                mask_roi = np.asarray(Image.open(os.path.join(current_path, "ROI.png")), dtype=int)
                fmeasures.append(self.evaluate__folder_c(os.path.join(current_path), os.path.join(output_path, 'results', cat, elem)))
                # self.evaluate_video(os.path.join(current_path, "groundtruth"), os.path.join(output_path, 'results', cat, elem), mask_roi,
                #                     range(temporal_roi[0], temporal_roi[1], step))
        print(fmeasures)
        return np.mean(np.asarray(fmeasures))

    def evaluate__folder_c(self, input_path, output_path, step=1):
        base = "../../"
        input_path_linux = base + input_path.replace('\\', "/")
        output_path_linux = base + output_path.replace('\\', "/")
        # subprocess.call(["bash", "-c", "cd ../../;ls"], cwd=os.path.join("Data", "cdnet_C_code"))
        subprocess.call(["bash", "-c", "./comparator "+input_path_linux+" "+output_path_linux+" "+str(step)], cwd=os.path.join("Data", "cdnet_C_code"))
        # subprocess.call(["bash", "-c", "./comparator "+input_path+" "+output_path], cwd="/Data/cdnet_C_code/", shell=True)
        f = open(os.path.join(output_path, "fpr.txt"))
        fmeasure = float(f.readline())
        precision = float(f.readline())
        recall = float(f.readline())
        return fmeasure, precision, recall

    def evaluate_video(self, input_path, output_path, roi, indexes):
        np.floor_divide(roi, roi.max(), out=roi)
        for img in indexes:
            self.evaluate_frame(input_path, output_path, roi, img)
        recall = self.tp / (self.tp + self.fn)
        precision = self.tp / (self.tp + self.fp)
        fmeasure = (2*precision*recall) / (precision*recall)
        print("Precision :", precision, "\tRecall:", recall, "\tF_measure:", fmeasure)
        self.__init__()

    def evaluate_frame(self, input_path, output_path, roi, index):
        gt = np.asarray(Image.open(os.path.join(input_path, "gt{0:06d}.png".format(index))), dtype=int)
        res = np.asarray(Image.open(os.path.join(output_path, "bin{0:06d}.png".format(index))), dtype=int)
        np.multiply(res, 510, out=res)
        a = np.add(gt, np.add(res, roi))
        a = np.bincount(a.flatten())
        self.tn += a[1] + a[51]
        if len(a) >= 256:
            self.fn += a[256]
        if len(a) >= 511:
            self.fn += a[511]
        if len(a) >= 561:
            self.fp += a[561]
            self.nbShadowErrors += a[561]
        if len(a) >= 766:
            self.tp += a[766]

        # for i, val in np.ndenumerate(res):
        #     # print(i, roi.shape, gt.shape, res.shape)
        #     if roi[i] != BLACK and gt[i] != UNKNOWN:
        #         if res[i] == WHITE:
        #             if gt[i] == WHITE: self.tp += 1
        #             else: self.fp += 1
        #         else:
        #             if gt[i] == WHITE: self.fn += 1
        #             else: self.tn += 1
        #     if gt[i] == SHADOW and res[i] == WHITE: self.nbShadowErrors += 1


if __name__ == '__main__':
    cdnet_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
    output_path = "/users/yabernar/GrosDisque/CDNET14/optimisation"
    cmp = Comparator()
    print(cmp.evaluate_all(cdnet_path, output_path, [1]))
