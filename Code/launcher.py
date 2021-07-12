import os


def single_choice(header, choices):
    print("\033[H\033[J")
    print(header)

    options_values = "azerty"
    options_set = {}
    for key in choices:
        letter = options_values[len(options_set)]
        print(letter+".", key)
        options_set[letter] = choices[key]
    while True:
        answer = input()
        if answer in options_set.keys():
            options_set[answer]()
        else:
            print("Wrong input :", answer)


def multiple_choice(header, choices):
    pass


def variable_set(header, ):
    pass


class NewExecution:
    def __init__(self):
        self.number = 0

    def status(self):
        hd = "Current number of new executions :" + str(self.number)
        choices = {"General settings": self.general_settings,
                   "Dataset settings": self.dataset_settings,
                   "Model settings": self.model_settings}
        single_choice(hd, choices)

    def general_settings(self):
        hd = "General settings :"
        choices = {"Set seed": self.set_seed}
        single_choice(hd, choices)

    def dataset_settings(self):
        hd = "Dataset settings :"
        choices = {"Generated dataset": self.generated_dataset,
                   "Image dataset": self.image_dataset}
        single_choice(hd, choices)

    def model_settings(self):
        hd = "Model settings :"
        choices = {}
        single_choice(hd, choices)


class Launcher:
    def __init__(self):
        self.execution = None

    def main_menu(self):
        hd = "Welcome to the WatSOM generation tool, options are as follows :"
        choices = {"Show status of stored results.": self.status,
                   "Add new executions": self.new_exec}
        single_choice(hd, choices)

    def status(self):
        print("Status not implemented yet")
        self.main_menu()

    def new_exec(self):
        self.execution = NewExecution()
        self.execution.status()
        self.main_menu()


if __name__ == '__main__':
    app = Launcher()
    app.main_menu()