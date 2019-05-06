from Code.Stimulus import *
from scipy import signal


class DNF:
    def __init__(self, inputs):

        self.nb_steps = nb_steps  # start at the beginning
        self.noise_amplitude = np.linspace(0, 0.5, 10)
        self.input_stimulus = Stimulus(amplitude=1.0, position=(0.5, 0.5), radius=0.4, sigma=0.03)
        self.squared_errors = []
        # Optimized parameters are : dt_tau h Ap sp Am sm gi
        self.dt = dt  # temporal discretisation (seconds).
        self.tau = inputs["tau_dt"] * dt  # timescale
        self.h = inputs["h"]  # resting level
        self.Ap = inputs["Ap"]  # Amplitude for 2D difference-of-gaussian
        self.Sp = inputs["Sp"]  # Spreading (sigma) for 2D difference-of-gaussian
        self.Am = inputs["Am"]  # Amplitude for 2D difference-of-gaussian
        self.Sm = inputs["Sm"]  # Spreading (sigma) for 2D difference-of-gaussian
        self.gi = inputs["gi"]

        self.width = width  # width of the field
        self.height = height  # height of the field
        self.dnf_shape = (width, height)
        self.kernel = np.zeros([self.width * 2, self.height * 2], dtype=float)
        self.convolution = np.zeros([self.width, self.height], dtype=float)
        self.potentials = np.zeros([self.width, self.height], dtype=float)
        self.in_stimulus = np.zeros([self.width, self.height], dtype=float)

        # Lateral interaction kernel K consists in a Difference of Gaussian
        # which is a sum of excitatory and inhibitory weights (lateral weights)
        # Here the interaction kernel is a DoG plus a global inhibition term to ensure monostable solution
        self.kernel = (self.Ap * gaussian((self.width * 2, self.height * 2), self.Sp)) - (
                    gi / (self.width * self.height)) - (self.Am * gaussian((self.width * 2, self.height * 2), self.Sm))

    def set_input(self, in_stimulus):
        self.in_stimulus = in_stimulus.copy()

    # threshold potentials in range [0,1]
    def normalize_potentials(self):
        #         display("normalize potentials")
        for index, value in np.ndenumerate(self.potentials):
            if value > 1.0:
                self.potentials[index] = 1.0
            elif value < 0.0:
                self.potentials[index] = 0.0

    def run(self):
        for noise in self.noise_amplitude:
            for _ in range(self.nb_steps):
                # set input with random noise for the DNF
                self.input_stimulus.rotate_stimulus()
                self.set_input(self.input_stimulus.get_data_with_random_noise(max_noise_amplitude=noise))

                # simulate the field dynamic
                self.convolution = signal.fftconvolve(self.potentials, self.kernel, mode='same')
                self.potentials += self.dt * (
                            -self.potentials + self.h + self.convolution + self.in_stimulus) / self.tau
                self.normalize_potentials()

                # compute the squared error for each sample
                self.squared_error()

    def squared_error(self):
        estimated_values = normalized_space_to_rate_code(self.potentials, self.dnf_shape)
        real_values = normalized_space_to_rate_code(self.input_stimulus.get_stimulus(), self.input_stimulus.shape)
        l1_norm = sum(absolute_error(real_values[i], estimated_values[i]) for i in range(len(estimated_values)))
        self.squared_errors.append(l1_norm ** 2)

    def mean_squared_error(self):
        return np.mean(self.squared_errors)
