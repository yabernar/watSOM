import os
import sys
import numpy as np
import matplotlib
from PIL import Image
from PyQt5.QtWidgets import QSlider

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *

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

        self.current_image = QDoubleSpinBox()
        self.current_image.setDecimals(0)
        self.current_image.setRange(1, 5000)
        self.current_image.valueChanged.connect(self.update_image)

        self.fps = QDoubleSpinBox()
        self.fps.setDecimals(0)
        self.fps.setRange(1, 60)
        self.fps.setValue(30)
        self.fps.valueChanged.connect(self.update_fps)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_tick)
        self.timer.setInterval(1000//int(self.fps.value()))

        button_stop = QtWidgets.QPushButton('Stop', self)
        button_start = QtWidgets.QPushButton('Start', self)

        button_start.clicked.connect(self.timer.start)
        button_stop.clicked.connect(self.timer.stop)


        self.layout.addWidget(QtWidgets.QLabel("Category", alignment=QtCore.Qt.AlignCenter), 0, 0)
        self.layout.addWidget(self.category, 1, 0)
        self.layout.addWidget(QtWidgets.QLabel("Element", alignment=QtCore.Qt.AlignCenter), 0, 1)
        self.layout.addWidget(self.element, 1, 1)
        self.layout.addWidget(button_start, 0, 2)
        self.layout.addWidget(button_stop, 1, 2)
        self.layout.addWidget(QtWidgets.QLabel("Current image", alignment=QtCore.Qt.AlignCenter), 0, 3)
        self.layout.addWidget(self.current_image, 0, 4)
        self.layout.addWidget(QtWidgets.QLabel("FPS", alignment=QtCore.Qt.AlignCenter), 1, 3)
        self.layout.addWidget(self.fps, 1, 4)


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

    def timer_tick(self):
        self.current_image.setValue(self.current_image.value()+1)

    def update_fps(self):
        self.timer.setInterval(1000//int(self.fps.value()))

    def update_image(self):
        global current_image
        current_image = int(self.current_image.value())
        self.app.display.update_plot()

    def update_path(self):
        global current_path, current_image
        current_path = os.path.join("Data", "tracking", "dataset", self.category.currentText(), self.element.currentText())
        roi_file = open(os.path.join(current_path, "temporalROI.txt"), "r").readline().split()
        temporal_roi = (int(roi_file[0]), int(roi_file[1]))
        self.current_image.setValue(temporal_roi[0])
        # current_image = int(self.current_image.value())
        # self.app.display.update_plot()


class Display(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__()
        global plot_types
        plot_types = {"input_video": MplCanvas.input_video,
                      "background": MplCanvas.background,
                      "groundtruth": MplCanvas.groundtruth,
                      "som_results": MplCanvas.som_result,
                      "som_topo": MplCanvas.som_Topo,
                      "som_vq": MplCanvas.som_VQ,
                      "som_saliency": MplCanvas.som_Saliency,
                      "gng_results": MplCanvas.gng_result,
                      "statistics": MplCanvas.statistics,
                      "statistics_gng": MplCanvas.statistics_gng}
        self.graphs = []
        self.graphs.append(MplCanvas("input_video"))
        self.graphs.append(MplCanvas("som_vq"))
        self.graphs.append(MplCanvas("som_topo"))
        self.graphs.append(MplCanvas("groundtruth"))
        self.graphs.append(MplCanvas("som_saliency"))
        self.graphs.append(MplCanvas("som_results"))

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
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.fig.add_subplot(111)
        self.subject = subject
        current = plot_types[self.subject](self)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.draw()
        self.axbackground = self.fig.canvas.copy_from_bbox(self.axes.bbox)

        self.plot = self.axes.imshow(current)


        toolbar = NavigationToolbar(self.canvas, self)


        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self):
        current = plot_types[self.subject](self)
        self.plot.set_data(current)
        self.fig.canvas.restore_region(self.axbackground)
        # redraw just the points
        self.axes.draw_artist(self.plot)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.axes.bbox)

        self.fig.canvas.flush_events()
        # self.canvas.draw()

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

    def statistics_gng(self):
        name = current_path.split(os.path.sep)[3::]
        return Image.open(os.path.join("Statistics", "cdnet_gng", name[0]+"_"+name[1]+".png"))

    def som_result(self):
        name = current_path.split(os.path.sep)[3::]
        try:
            img = Image.open(os.path.join("Results", "Visualizations", "SOM", name[0]+"_"+name[1]+"13n-21p-1", "results", name[0], name[1], "bin{0:06d}.png".format(current_image)))
        except FileNotFoundError:
            img = Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))
        return img

    def som_VQ(self):
        name = current_path.split(os.path.sep)[3::]
        try:
            img = Image.open(os.path.join("Results", "Visualizations", "SOM", name[0]+"_"+name[1]+"13n-21p-1", "results", "supplements", "difference", "dif{0:06d}.png".format(current_image)))
        except FileNotFoundError:
            img = Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))
        return img

    def som_Topo(self):
        name = current_path.split(os.path.sep)[3::]
        try:
            img = Image.open(os.path.join("Results", "Visualizations", "SOM", name[0]+"_"+name[1]+"13n-21p-1", "results", "supplements", "diff_winners", "win{0:06d}.png".format(current_image)))
        except FileNotFoundError:
            img = Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))
        return img

    def som_Saliency(self):
        name = current_path.split(os.path.sep)[3::]
        try:
            img = Image.open(os.path.join("Results", "Visualizations", "SOM", name[0]+"_"+name[1]+"13n-21p-1", "results", "supplements", "saliency", "sal{0:06d}.png".format(current_image)))
        except FileNotFoundError:
            img = Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))
        return img

    def gng_result(self):
        name = current_path.split(os.path.sep)[3::]
        try:
            img = Image.open(os.path.join("Results", "Visualizations", "GNG", name[0]+"_"+name[1]+"13n-21p-1", "results", name[0], name[1], "bin{0:06d}.png".format(current_image)))
        except FileNotFoundError:
            img = Image.open(os.path.join(current_path, "groundtruth", "gt{0:06}.png".format(current_image)))
        return img



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self._title = 'Prueba real-time'
        self.setWindowTitle(self._title)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        self.display = Display()
        self.navigation = Navigation(self)

        # layouts
        layout = QtWidgets.QGridLayout(self._main)

        layout.addWidget(self.display, 0, 0)
        layout.addWidget(self.navigation, 1, 0)

        layout.setColumnStretch(0, 2)
        layout.setRowStretch(0, 5)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    sys.exit(qapp.exec_())