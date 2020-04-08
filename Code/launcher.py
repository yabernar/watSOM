import os

class NewExecution:
    def __init__(self):
        number = 0

    def status(self):
        print("\033[H\033[J")
        print("Current number of new executions :", self.number)

        print("a. General settings")
        print("z. Dataset settings")
        print("e. Model settings")

    def general_settings(self):
        print("\033[H\033[J")
        print("General settings :")

        print("a. Set seed")

    def seed_param(self):


class Launcher:
    def __init__(self):
        new_exec = None

    def main_menu(self):
        print("\033[H\033[J")
        print("Welcome to the WatSOM generation tool, options are as follows :")
        print("a. Show status of stored results.")
        print("z. Add new executions")
        answer = input()
        if answer == "a":
            self.status()
        elif answer == "z":
            self.new_exec()
        else:
            self.main_menu()

    def status(self):
        print("\033[H\033[J")
        print("Status not implemented yet")
        self.main_menu()

    def new_exec(self):
        self.new_exec = NewExecution()
        self.new_exec.status()
        self.main_menu()

if __name__ == '__main__':
    app = Launcher()
    app.main_menu()