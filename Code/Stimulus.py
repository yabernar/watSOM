from Code.Tools import *


class Stimulus:

    # amplitude of the stimulus
    # position is normalized between [-1,1], [0,0] corresponds to the center of the field
    # shape of the field
    # radius (normalised between 0 and 1)
    # sigma : spreading of the gaussian
    # period in ms
    def __init__(self, amplitude=1.0, position=(0.5, 0.5), shape=(width, height), radius=0.2, sigma=0.01,
                 period=1 * second):
        self.angle = 0.0
        self.dt = dt

        self.amplitude = amplitude
        self.position = position
        self.shape = shape
        self.radius = radius
        self.sigma = sigma
        self.period = period / self.dt  # number of steps for a period
        self.data = self.amplitude * gaussian(self.shape, self.sigma, self.position, normalized=False)

    #  shape type shall be a tuple
    def update_parameters(self, shape):
        self.shape = shape
        self.data = self.amplitude * gaussian(self.shape, self.sigma, self.position, normalized=False)

    def get_stimulus(self):
        return self.data

    def rotate_stimulus(self):
        # Update the angle
        self.angle += 1.0 / self.period
        # Compute the position of the center of the bubble after rotation by :
        # 1) Converting polar (radius, angles) coords to cartesian (x, y) coords
        # 2) Scaling by the position of the center for each dimension
        self.cx = self.position[0] + (self.radius * np.cos(2.0 * np.pi * self.angle))
        self.cy = self.position[1] + (self.radius * np.sin(2.0 * np.pi * self.angle))
        #             display ("cx = ",self.cx, "cy = ",self.cy)
        # Update the position of the gaussian bubble
        self.data = self.amplitude * gaussian(self.shape, self.sigma, (self.cx, self.cy), normalized=False)

    def get_data_with_random_noise(self, shape=(width, height), max_noise_amplitude=0.01):
        return self.normalize_data(self.data + max_noise_amplitude * (np.random.random(size=shape)))

    # threshold potentials in range [0,1]
    def normalize_data(self, data):
        normalized_data = data
        for index, value in np.ndenumerate(normalized_data):
            if value > 1.0:
                normalized_data[index] = 1.0
            elif value < 0.0:
                normalized_data[index] = 0.0
        return normalized_data
