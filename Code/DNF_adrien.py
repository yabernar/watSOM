import itertools
import os

import pandas as pd
from hyperopt import Trials, hp, fmin, tpe

from Code.Stimulus import *
from scipy import signal
import matplotlib.pyplot as plt
import multiprocessing as mp


class DNF:
    def __init__(self, inputs):

        self.nb_steps = nb_steps  # start at the beginning
        self.noise_amplitude = [0.3, 0.6, 1]
        self.input_stimulus = Stimulus(amplitude=1.0, position=(0.5, 0.5), radius=0.4, sigma=0.08)
        self.squared_errors = []
        self.difference_errors = []
        self.old_potentials = 0

        # Optimized parameters are : dt_tau h Ap sp Am sm gi
        self.dt = dt  # temporal discretisation (seconds).
        self.tau = inputs["tau_dt"] * dt  # timescale
        self.h = inputs["h"]  # resting level
        self.Ap = inputs["Ap"]  # Amplitude for 2D difference-of-gaussian
        self.Sp = inputs["Sp"]  # Spreading (sigma) for 2D difference-of-gaussian
        self.Am = inputs["Am"]  # Amplitude for 2D difference-of-gaussian
        self.Sm = inputs["Sm"]  # Spreading (sigma) for 2D difference-of-gaussian
        self.gi = inputs["gi"]

        self.width = inputs["size"][0]  # width of the field
        self.height = inputs["size"][1]  # height of the field
        self.dnf_shape = (width, height)
        self.kernel = np.zeros([self.width * 2, self.height * 2], dtype=float)
        self.convolution = np.zeros([self.width, self.height], dtype=float)
        self.potentials = np.zeros([self.width, self.height], dtype=float)
        self.in_stimulus = np.zeros([self.width, self.height], dtype=float)

        # Lateral interaction kernel K consists in a Difference of Gaussian
        # which is a sum of excitatory and inhibitory weights (lateral weights)
        # Here the interaction kernel is a DoG plus a global inhibition term to ensure monostable solution
        self.excitation_kernel = gaussian((self.width * 2, self.height * 2), self.Sp)
        self.excitation_kernel = np.divide(self.excitation_kernel, np.sum(self.excitation_kernel))
        self.kernel = (self.Ap * self.excitation_kernel) - (self.gi / (self.width * self.height)) - (self.Am * gaussian((self.width * 2, self.height * 2), self.Sm))

    def set_input(self, in_stimulus):
        self.in_stimulus = in_stimulus.copy()

    # threshold potentials in range [0,1]
    def normalize_potentials(self):
        # display("normalize potentials")
        for index, value in np.ndenumerate(self.potentials):
            if value > 1.0:
                self.potentials[index] = 1.0
            elif value < 0.0:
                self.potentials[index] = 0.0

    def run_once(self, input = None):
        # set input with random noise for the DNF
        if input is None:
            self.input_stimulus.rotate_stimulus()
            self.set_input(self.input_stimulus.get_data_with_random_noise(max_noise_amplitude=1))
        else:
            self.in_stimulus = input

        # simulate the field dynamic
        self.convolution = signal.fftconvolve(self.potentials, self.kernel, mode='same')
        # max = np.max(self.convolution)
        # if max > 0:
        #     self.convolution = np.divide(self.convolution, max)
        self.potentials += self.dt * (-self.potentials + self.h + self.convolution*0.5 + self.in_stimulus) / self.tau
        self.normalize_potentials()

        # self.pixels_diff_error()
        # print(self.squared_error())

    def run(self):
        for noise in self.noise_amplitude:
            for _ in range(self.nb_steps):
                # set input with random noise for the DNF
                self.input_stimulus.rotate_stimulus()
                self.set_input(self.input_stimulus.get_data_with_random_noise(max_noise_amplitude=noise))
                # simulate the field dynamic
                self.convolution = signal.fftconvolve(self.potentials, self.kernel, mode='same')
                self.potentials += self.dt * (-self.potentials + self.h + self.convolution + self.in_stimulus) / self.tau
                self.normalize_potentials()

                # compute the squared error for each sample
                self.pixels_diff_error()

    def pixels_diff_error(self):
        base = np.asarray(self.input_stimulus.get_stimulus())
        self.difference = np.abs(np.subtract(base, self.potentials))
        if np.max(self.potentials) < 0.1:
            # print(np.max(self.potentials))
            self.difference_errors.append(1)
        else:
            # self.difference_errors.append(np.mean(self.difference))
            potentials_mean = np.mean(self.potentials)
            total = np.mean(self.difference) + np.abs(potentials_mean - self.old_potentials)*10
            # print("Err : ", np.mean(self.difference), "and", np.abs(potentials_mean - self.old_potentials))
            self.old_potentials = potentials_mean
            self.difference_errors.append(total)
            # print("Total : ", total)

    def get_total_pixel_error(self):
        return np.mean(self.difference_errors)

    def squared_error(self):
        estimated_values = normalized_space_to_rate_code(self.potentials, self.dnf_shape)
        real_values = normalized_space_to_rate_code(self.input_stimulus.get_stimulus(), self.input_stimulus.shape)
        l1_norm = sum(absolute_error(real_values[i], estimated_values[i]) for i in range(len(estimated_values)))
        self.squared_errors.append(l1_norm ** 2)
        return l1_norm**2

    def mean_squared_error(self):
        return np.mean(self.squared_errors)


def show():
    # {'amplitude_excitation': 5.832715159828023, 'amplitude_inhibition': 2.720613585340117, 'global_inhibition': 24.22450711161237, 'h': -0.4210095795257387,
    # 'sigma_excitation': 0.0694045679294695, 'sigma_inhibition': 0.15299140668495112, 'tau_dt': 0.9998283112954799}

    inputs_DNF = {"tau_dt": 0.85,   # entre 0 et 1
                  "h": -0.42,        # entre -1 et 0
                  "Ap": 5.83,        # entre 1 et 10
                  "Sp": 0.0694,       # entre 0 et 1
                  "Am": 2.72,        # entre 1 et 10
                  "Sm": 0.1530,      # entre 0 et 1
                  "gi": 24}        # entre 0 et 500

    inputs_DNF = {"tau_dt": 0.8,   # entre 0 et 1
                  "h": -0.6,        # entre -1 et 0
                  "Ap": 1,        # entre 1 et 10
                  "Sp": 0.01,       # entre 0 et 1
                  "Am": 0,        # entre 1 et 10
                  "Sm": 0.05,      # entre 0 et 1
                  "gi": 2}        # entre 0 et 500

    dnf = DNF(inputs_DNF)
    plot = None
    while True:
        dnf.run_once()
        if plot is None:
            plot = []
            plt.subplot(2, 2, 1)
            plot.append(plt.imshow(dnf.in_stimulus))
            plt.subplot(2, 2, 2)
            plot.append(plt.imshow(dnf.input_stimulus.get_stimulus()))
            plt.subplot(2, 2, 3)
            plot.append(plt.imshow(dnf.difference))
            plt.subplot(2, 2, 4)
            plot.append(plt.imshow(dnf.potentials))
        else:
            plot[0].set_data(dnf.in_stimulus)
            plot[1].set_data(dnf.input_stimulus.get_stimulus())
            plot[2].set_data(dnf.difference)
            plot[3].set_data(dnf.potentials)
        plt.pause(0.02)
        plt.draw()


def evaluate(model, parameters, evaluation):
    np.random.seed()
    m = model(parameters)
    m.run()
    return evaluation(m)


def optimize(params):
    tau_dt, h, ampl_exci, sigma_exci, ampl_inhib, sigma_inhib, global_inhib = params
    inputs_DNF = {"tau_dt": tau_dt,   # entre 0 et 1
                  "h": h,        # entre -1 et 0
                  "Ap": ampl_exci,        # entre 1 et 10
                  "Sp": sigma_exci,       # entre 0 et 1
                  "Am": ampl_inhib,        # entre 1 et 10
                  "Sm": sigma_inhib,      # entre 0 et 1
                  "gi": global_inhib}        # entre 0 et 500
    nb_runs = 6
    pool = mp.Pool(nb_runs)
    a = pool.starmap(evaluate, zip(itertools.repeat(DNF, nb_runs), itertools.repeat(inputs_DNF, nb_runs), itertools.repeat(DNF.get_total_pixel_error, nb_runs)))
    pool.close()
    pool.join()
    return np.mean(a)


if __name__ == '__main__':
    show()
    tpe_trials = Trials()
    dnf_params = (hp.uniform('tau_dt', 0, 1), hp.uniform('h', -1, 0), hp.uniform('amplitude_excitation', 1, 10), hp.uniform('sigma_excitation', 0, 1),
                  hp.uniform('amplitude_inhibition', 1, 10), hp.uniform('sigma_inhibition', 0, 1), hp.uniform('global_inhibition', 0, 500))

    best = fmin(optimize, dnf_params, algo=tpe.suggest, trials=tpe_trials, max_evals=10000)

    print()
    print(best)

    full_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                'iteration': tpe_trials.idxs_vals[0]['tau_dt'],
                                'tau_dt': tpe_trials.idxs_vals[1]['tau_dt'],
                                'h': tpe_trials.idxs_vals[1]['h'],
                                'amplitude_excitation': tpe_trials.idxs_vals[1]['amplitude_excitation'],
                                'sigma_excitation': tpe_trials.idxs_vals[1]['sigma_excitation'],
                                'amplitude_inhibition': tpe_trials.idxs_vals[1]['amplitude_inhibition'],
                                'sigma_inhibition': tpe_trials.idxs_vals[1]['sigma_inhibition'],
                                'global_inhibition': tpe_trials.idxs_vals[1]['global_inhibition']})
    full_results.to_csv("/users/yabernar/GrosDisque/CDNET14/optimisation/basic_dnf.csv")
