def linear(progress):  # progress is between 0 and 1
    return progress


class Parameter:
    def __init__(self, start=1, end=0, nb_steps=1, evolution=linear):
        self.start = start
        self.end = end
        self.nb_steps = nb_steps
        self.evolution = evolution

        self.iteration = 0
        self.difference = self.start - self.end
        self.current = self.start

    def get(self):
        return self.current

    def execute(self):
        if self.iteration < self.nb_steps:
            self.iteration += 1
        self.current = self.start - self.difference * self.evolution(self.iteration/self.nb_steps)
