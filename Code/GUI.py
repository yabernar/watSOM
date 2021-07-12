import matplotlib.pyplot as plt


class GUI:
    def __init__(self):
        self.plot_nbr = 4
        self.plot_current = 1
        self.models = []
        self.plots = []

    def add_plot(self, data):
        plt.subplot(2, 2, self.plot_current)
        self.plot_current += 1
        self.models.append(data)
        self.plots.append(data.display())

    def update(self):
        for i in range(len(self.plots)):

        plt.draw()



        if plot is None:
            plot = plt.imshow(data.reconstruct(som.get_reconstructed_data()))
        else:
            plot.set_data(data.reconstruct(som.get_neural_list(), size=som.neurons_nbr))
