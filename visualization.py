import os
import sys
import numpy as np
import matplotlib
from PIL import Image
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

current_path = os.path.join("Data", "tracking", "dataset", "badWeather", "blizzard")
plot_types = None
current_image = 1


def onclick(event):
    global clicks
    clicks.append(event.xdata)


class Navigation(QtWidgets.QGroupBox):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.layout = QtWidgets.QGridLayout()
        self.category = QtWidgets.QComboBox()
        self.element = QtWidgets.QComboBox()

        self.fill_categories()
        self.fill_elements()
        self.category.currentIndexChanged.connect(self.fill_elements)
        self.element.currentIndexChanged.connect(self.update_path)
        button_stop = QtWidgets.QPushButton('Stop', self)
        button_start = QtWidgets.QPushButton('Start', self)

        self.layout.addWidget(QtWidgets.QLabel("Category", alignment=QtCore.Qt.AlignCenter), 0, 0)
        self.layout.addWidget(self.category, 1, 0)
        self.layout.addWidget(QtWidgets.QLabel("Element", alignment=QtCore.Qt.AlignCenter), 0, 1)
        self.layout.addWidget(self.element, 1, 1)
        self.layout.addWidget(button_start, 0, 2)
        self.layout.addWidget(button_stop, 0, 3)
        self.setLayout(self.layout)

    def fill_categories(self):
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
        for cat in categories:
            self.category.addItem(cat)

    def fill_elements(self):
        cdnet_path = os.path.join("Data", "tracking", "dataset")
        elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, self.category.currentText())) if os.path.isdir(os.path.join(cdnet_path, self.category.currentText(), d))], key=str.lower)
        self.element.clear()
        for elem in elements:
            self.element.addItem(elem)

    def update_path(self):
        global current_path
        current_path = os.path.join("Data", "tracking", "dataset", self.category.currentText(), self.element.currentText())
        self.app.display.update_plot()


class Display(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__()
        global plot_types
        plot_types = {"input_video": MplCanvas.input_video,
                        "background": MplCanvas.background,
                        "groundtruth": MplCanvas.groundtruth,
                        "statistics": MplCanvas.statistics}
        self.graphs = []
        self.graphs.append(MplCanvas("input_video"))
        self.graphs.append(MplCanvas("statistics"))
        self.graphs.append(MplCanvas("statistics"))
        self.graphs.append(MplCanvas("groundtruth"))
        self.graphs.append(MplCanvas("groundtruth"))
        self.graphs.append(MplCanvas("groundtruth"))

        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.graphs[0], 0, 0)
        self.layout.addWidget(self.graphs[1], 0, 1)
        self.layout.addWidget(self.graphs[2], 0, 2)
        self.layout.addWidget(self.graphs[3], 1, 0)
        self.layout.addWidget(self.graphs[4], 1, 1)
        self.layout.addWidget(self.graphs[5], 1, 2)
        self.setLayout(self.layout)

    def update_plot(self):
        for i in self.graphs:
            i.update_plot()


class MplCanvas(QtWidgets.QGroupBox):
    def __init__(self, subject):
        super().__init__()
        fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        self.subject = subject
        current = plot_types[self.subject](self)
        self.plot = self.axes.imshow(current)

        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self):
        current = plot_types[self.subject](self)
        self.plot.set_data(current)
        self.canvas.draw()

    # Content switch section
    def input_video(self):
        return Image.open(os.path.join(current_path, "input", "in{0:06d}.jpg".format(current_image)))

    def background(self):
        return Image.open(os.path.join(current_path, "bkg.jpg"))

    def groundtruth(self):
        return Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))

    def statistics(self):
        name = current_path.split(os.path.sep)[3::]
        return Image.open(os.path.join("Statistics", "cdnet_som", name[0]+"_"+name[1]+".png"))

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self._title = 'Prueba real-time'
        self.setWindowTitle(self._title)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        self.display = Display()
        self.navigation = Navigation(self)


        # self.table_clicks = QtWidgets.QTableWidget(0, 2)
        # self.table_clicks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


        # layouts

        layout = QtWidgets.QGridLayout(self._main)

        layout.addWidget(self.display, 0, 0)
        # layout.addWidget(self.table_clicks, 0, 1)
        layout.addWidget(self.navigation, 1, 0)

        layout.setColumnStretch(0, 2)
        layout.setRowStretch(0, 5)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    sys.exit(qapp.exec_())