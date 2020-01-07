import os
import struct
import wave

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from PIL import Image


class SpokenDigitsDataset:
    def __init__(self, dataset_path, length=100):
        self.length = length
        self.data = []
        self.recording_folder = dataset_path
        self.create()

    def create(self):
        files = sorted([d for d in os.listdir(self.recording_folder) if os.path.isfile(os.path.join(self.recording_folder, d))], key=str.lower)
        for f in files:
            fs, data = wavfile.read(os.path.join(self.recording_folder, f))
            data = resample(data, self.length)
            data = (data / 65536) + 0.5
            data = data - min(data)
            data = data / max(data)
            self.data.append(data)
        self.data = np.asarray(self.data)

    def get_data(self):
        return self.data


if __name__ == '__main__':
    SpokenDigitsDataset("/users/yabernar/workspace/watSOM/Data/FSDD/recordings")
