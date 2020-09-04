import os
from tkinter import Menu
import matplotlib.pyplot as plt

from pyimg.config import constants
from pyimg.menus.operation_interface import UnaryImageOperation
from pyimg.models.image import ImageImpl


def plot_hist(image: ImageImpl):
    if image.channels > 1:
        color = ('red', 'green', 'blue')
        legends = ['Red Channel', 'Green Channel', 'Blue Channel', 'Total']
        #full_plot = plt.figure()

        for i in range(image.channels):
            plt.figure()
            plt.hist(image.array[:, :, i].ravel(), bins=constants.MAX_PIXEL_VALUE + 1, color=color[i])
            plt.xlabel('Intensity Value')
            plt.ylabel('Count')
            plt.legend([legends[i]])
            plt.title('Histogram for ' + legends[i])
            plt.show()

            #full_plot.hist(image.array[:, :, i].ravel(), bins=constants.MAX_PIXEL_VALUE + 1, color=color[i], alpha=0.5)

        #full_plot.xlabel('Intensity Value')
        #full_plot.ylabel('Count')
        #full_plot.legend(legends)
        #full_plot.show()
    else:
        plt.figure()
        plt.hist(image.array.ravel(), bins=constants.MAX_PIXEL_VALUE + 1)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend('Gray scale')
        plt.show()


class InfoImageMenu:
    """
    This class is a simple wrapper around tkinter menu object.
    it's self display as a menu in the top bar.
    """

    def __init__(self, menubar, image_io):
        info_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Info", menu=info_menu)

        info_menu.add_command(
            label="Histogram",
            command=UnaryImageOperation(image_io, 'Plot',
                                        lambda x: plot_hist(x)
                                        ).generate_interface
        )
