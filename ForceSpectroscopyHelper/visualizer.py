import os
import numpy as np

from .analyzer import GridAnalyzer
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


class NanonisGridVisualizer(object):

    def __init__(self, fileName, analyzer: GridAnalyzer, **kwargs):
        if "array_3d" not in kwargs:
            kwargs["array_3d"] = None

        self.apply_offset = False
        self.plot_addition_array = False
        self.plot_raw_array = True
        if "apply_offset" in kwargs:
            self.apply_offset = kwargs["apply_offset"]
        if "plot_addition_array" in kwargs:
            self.plot_addition_array = kwargs["plot_addition_array"]
        if "plot_raw_array" in kwargs:
            self.plot_raw_array = kwargs["plot_raw_array"]

        self.guiparam = NanonisGridVisualizer.gui_param(fileName, analyzer, kwargs["array_3d"])

        if "show_app" in kwargs:
            if kwargs["show_app"] is True:
                self.visual_app()
        else:
            self.visual_app()

    def visual_app(self):
        fileName = self.guiparam.fileName

        if self.apply_offset:
            data = self.guiparam.df_3d_offset_array
        else:
            data = self.guiparam.df_3d_array

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.canvas.set_window_title(fileName)
        # x, y = self.guiparam.analyzer.get_curve(point=(0, 0), fileName=fileName)
        # axes.flat[0].plot(x, y)

        # region Event Function
        def update_move(event):
            if event.xdata is None or event.ydata is None or event.inaxes is None:
                return

            if event.button == 1:
                if axes.flat[1] == event.inaxes:

                    if not self.guiparam.draw_mode:
                        color = np.random.rand(3,)
                        point = int(round(event.xdata)), int(round(event.ydata))
                        # axes.flat[0].cla()
                        x, y = self.guiparam.analyzer.get_curve(point=point, fileName=fileName)
                        if self.plot_raw_array:
                            axes.flat[0].plot(x, y, linestyle="-", c=color)
                        # slice [0:] cause error
                        if self.plot_addition_array:
                            x, y = self.guiparam.get_curve(point)
                            axes.flat[0].plot(x, y, linestyle="--", c=color)

                        plt.draw()

                    else:
                        if self.guiparam.gui_xy_data[0] == (None, None):
                            self.guiparam.gui_xy_data[0] = (event.xdata, event.ydata)
                        elif self.guiparam.gui_xy_data[1] == (None, None):
                            clr1_press(None)
                            self.guiparam.gui_xy_data[1] = (event.xdata, event.ydata)
                            draw_change(None)
                            axes.flat[1].plot([self.guiparam.gui_xy_data[0][0], self.guiparam.gui_xy_data[1][0]],
                                              [self.guiparam.gui_xy_data[0][1], self.guiparam.gui_xy_data[1][1]],
                                              color="white", linestyle="dashed")
                            # self.gui_xy_data = [(None, None), (None, None)]

        def update_im(var):
            im.set_data(data[:, :, int(var)])
            min = np.min(data[:, :, int(var)])
            max = np.max(data[:, :, int(var)])
            cbar.set_clim(vmin=min, vmax=max)
            cbar.draw_all()
            plt.draw()

        def clr0_press(var):
            axes.flat[0].cla()
            plt.draw()

        def clr1_press(var):
            if len(axes.flat[1].lines) == 0:
                return
            axes.flat[1].lines.pop(0)
            self.guiparam.gui_xy_data = [(None, None), (None, None)]
            plt.draw()

        def map1_press(var):
            self.__Show2DMap()

        def map2_press(var):
            self.__ShowTopoMap()

        def draw_change(var):
            if self.guiparam.draw_mode:
                self.guiparam.draw_mode = False
                drawButton.color = "red"
                drawButton.label.set_text("draw OFF")
            else:
                self.guiparam.draw_mode = True
                drawButton.color = "green"
                drawButton.label.set_text("draw ON")
            plt.draw()

        # endregion

        fig.canvas.mpl_connect("button_press_event", update_move)
        plt.subplots_adjust(bottom=0.25)
        im = axes.flat[1].imshow(data[:, :, 0])
        cbar = fig.colorbar(im)

        # widget position
        slider_ax = plt.axes([0.25, 0.1, 0.65, 0.05])

        clr0_button_ax = plt.axes([0.05, 0.95, 0.2, 0.05])
        clr1_button_ax = plt.axes([0.3, 0.95, 0.2, 0.05])
        draw_button_ax = plt.axes([0.55, 0.95, 0.2, 0.05])
        map1_button_ax = plt.axes([0.8, 0.95, 0.1, 0.05])
        map2_button_ax = plt.axes([0.9, 0.95, 0.1, 0.05])
        indexSlider = widgets.Slider(slider_ax, "z index", 0, data.shape[2] - 1,
                                     valinit=0, valstep=1)

        # widget defination
        indexSlider.on_changed(update_im)
        clr0Button = widgets.Button(clr0_button_ax, "clear curve")
        clr0Button.on_clicked(clr0_press)
        clr1Button = widgets.Button(clr1_button_ax, "clear draw")
        clr1Button.on_clicked(clr1_press)
        drawButton = widgets.Button(draw_button_ax, "draw OFF", color="red")
        drawButton.on_clicked(draw_change)
        map1Button = widgets.Button(map1_button_ax, "2D")
        map1Button.on_clicked(map1_press)
        map2Button = widgets.Button(map2_button_ax, "topo")
        map2Button.on_clicked(map2_press)
        plt.show()

    def __Show2DMap(self):
        fileName = self.guiparam.fileName
        x_count = abs(round(self.guiparam.gui_xy_data[0][0]) - round(self.guiparam.gui_xy_data[1][0]))
        y_count = abs(round(self.guiparam.gui_xy_data[0][1]) - round(self.guiparam.gui_xy_data[1][1]))

        count = int(np.min([x_count, y_count]))
        x = np.linspace((self.guiparam.gui_xy_data[0][0]), (self.guiparam.gui_xy_data[1][0]), count)
        y = np.linspace((self.guiparam.gui_xy_data[0][1]), (self.guiparam.gui_xy_data[1][1]), count)
        array = np.zeros(shape=(self.guiparam.analyzer.get_data_count(fileName), count))
        for i in range(0, count):
            array[:, i] = self.guiparam.analyzer.get_curve(point=(int(round(x[i])), int(round(y[i]))), fileName=fileName)[1]

        aspect = count / self.guiparam.analyzer.get_data_count(fileName)
        self.PopupImage(array, aspect)

    def __ShowTopoMap(self):
        self.PopupImage(self.guiparam.topo, 1)

    def PopupImage(self, image, aspect):
        f = plt.figure()
        plt.imshow(image, aspect=aspect)
        plt.colorbar()
        plt.show()

    class gui_param:
        def __init__(self, fileName, analyzer: GridAnalyzer, array_3d=None):
            if fileName is None:
                return

            self.fileName = fileName
            self.analyzer = analyzer
            self.topo = analyzer.topo(fileName)

            self.xy_count = self.analyzer.get_xy_count(fileName)
            self.z_count = self.analyzer.get_data_count(fileName)

            if array_3d is not None and np.array(array_3d).shape != (self.xy_count[0], self.xy_count[1], self.z_count):
                raise ValueError("3d array shape does not march to source file")

            if array_3d is not None:
                self.df_3d_array = np.array(array_3d)
            else:
                self.df_3d_array = np.array(analyzer.get_df_matrix(fileName))

            x, _ = self.analyzer.get_curve((0, 0), self.fileName)

            self.z_height_range = self.analyzer.get_height_range(fileName)

            self.grid_count = int((self.z_height_range[1] - self.z_height_range[0]) / (x[0] - x[1]))
            self.grid_z_array = np.linspace(self.z_height_range[0], self.z_height_range[1], self.grid_count)

            self.df_3d_offset_array = self.make_df_3d_offset_array(self.df_3d_array)

            # Cache data
            self.draw_mode = False
            self.gui_xy_data = [(None, None), (None, None)]

        def find_close_index(self, max_x_curve):
            return self.find_close_index_in_array(max_x_curve, self.grid_z_array)

        @staticmethod
        def find_close_index_in_array(max_x, array):
            _e = np.abs(array - max_x)
            return np.argmin(_e)

        def make_df_3d_offset_array(self, array_3d):
            result = np.zeros(shape=(self.xy_count[0], self.xy_count[1], self.grid_count))
            for i in range(0, self.xy_count[0]):
                for j in range(0, self.xy_count[1]):
                    x, y = self.analyzer.get_curve((i, j), self.fileName)
                    index = self.find_close_index(np.max(x))
                    # print(i, j, index)
                    # index-self.z_count, index
                    for k in range(0, self.grid_count):
                        result[j, i, k] = 0
                    for k in range(0, self.z_count):
                        result[j, i,index - k] = array_3d[j, i, k]
                    result[j, i, :] = np.flipud(result[j, i, :])

            return result

        @staticmethod
        def get_data_available_range(data):
            forward, backward = 0, 0
            for i in range(len(data)):
                if np.abs(data[i]) < 0.00001:
                    continue
                forward = i
                break
            data = np.flipud(data)
            for i in range(len(data)):
                if np.abs(data[i]) < 0.00001:
                    continue
                backward = len(data)-i
                break
            if forward > backward:
                raise ValueError("range error")
            return forward, backward

        def get_curve(self, point):
            x1 = np.flipud(self.grid_z_array)
            y1 = self.df_3d_offset_array[point[1], point[0], :]
            f, b = self.get_data_available_range(y1)
            if f == 0 and b != 0:
                return x1[:b], y1[:b]
            elif b == 0 and f != 0:
                return x1[f:], y1[f:]
            elif f == 0 and b == 0:
                return x1, y1
            else:
                return x1[f:b], y1[f:b]


