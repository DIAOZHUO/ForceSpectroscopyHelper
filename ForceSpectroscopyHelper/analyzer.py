from enum import Enum

import sys
import os
import pandas as pd
import numpy as np
from typing import List
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter

from ForceSpectroscopyHelper import *
from ForceSpectroscopyHelper.structures import *
import ForceSpectroscopyHelper.formula as formula
from .dulcineaMapCreator import DulcineaMapCreator

import nanonispy as nap
import matplotlib.pyplot as plt
from scipy import ndimage


class BaseAnalyzer:

    class FlattenMode(Enum):
        RawData = 1
        Average = 2
        LinearFit = 3
        PolyFit = 4
        ThreePointsFlatten = 5
        Off = 6

    topo_poly_order = 3
    gaussian_filter_sigma = 3

    def __init__(self, directory, fileNamesList=None):
        self.directory = directory
        self.directory_out = self.directory + '/Outputs/'
        self.fileNamesList = fileNamesList

        if not os.path.exists(self.directory_out):
            os.mkdir(self.directory_out)


    @staticmethod
    def flatten_plane(map, coef_x, coef_y):
        m = np.array(map)
        size = m.shape
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                m[i, j] -= j * coef_x + i * coef_y
        return m - np.min(m)


    @staticmethod
    def get_flatten_param(mapping_data, flatten=FlattenMode.LinearFit, flatten_param=None):
        size = Vector2(mapping_data.shape[0], mapping_data.shape[1])
        array_x = np.linspace(0, size.x, size.x)
        array_y = np.linspace(0, size.y, size.y)
        coef_x, coef_y = 0, 0

        if flatten == BaseAnalyzer.FlattenMode.Average:
            raise ValueError("please use fitting_map function for FlattenMode.Average")
        if flatten == BaseAnalyzer.FlattenMode.LinearFit:
            for i in range(0, size.y):
                coef = np.polyfit(array_x, mapping_data[i], 1)
                coef_x += coef[0]
                # plane_data[i, :] += np.polyval(np.array([coef[0], 0]), array_x)
            coef_x /= size.y
            for i in range(0, size.x):
                coef = np.polyfit(array_y, mapping_data[:, i], 1)
                coef_y += coef[0]
            coef_y /= size.x
            return coef_x, coef_y

        if flatten == BaseAnalyzer.FlattenMode.PolyFit:
            array_x = np.linspace(0, size.x, size.x)

            for i in range(0, size.y):
                coef = np.polyfit(array_x, mapping_data[i], BaseAnalyzer.topo_poly_order)
                return coef

        if flatten == BaseAnalyzer.FlattenMode.ThreePointsFlatten:
            array_x1 = np.linspace(flatten_param[0], flatten_param[1], flatten_param[1] - flatten_param[0])
            coef1 = np.polyfit(array_x1, mapping_data[flatten_param[2], flatten_param[0]:flatten_param[1]], 1)
            array_x2 = np.linspace(flatten_param[2], flatten_param[3], flatten_param[3] - flatten_param[2])
            coef2 = np.polyfit(array_x2, mapping_data[flatten_param[2]:flatten_param[3], flatten_param[0]], 1)
            return coef1, coef2
        else:
            raise ValueError("incorrect flatten mode param")

    """
        傾けるマップを補正する関数
        x,y pointにあるz情報を与えるか(pixel_data) 2D mappingを与える
    """
    @staticmethod
    def fitting_map(mapping_data, flatten=FlattenMode.Average, flatten_param=None):
        data = BaseAnalyzer.fitting_data(None, size=Vector2(mapping_data.shape[0], mapping_data.shape[1]),
                                         mapping_data=mapping_data, flatten=flatten, flatten_param=flatten_param)
        mapping_data = np.zeros(shape=mapping_data.shape)
        for it in data:
            mapping_data[it.x][it.y] = it.z
        return np.array(mapping_data)


    @staticmethod
    def fitting_data(pixel_data: List[Vector3], size: Vector2, mapping_data=None,
                     flatten=FlattenMode.Average, flatten_param=None) -> List[Vector3]:
        if mapping_data is None:
            # 二次元配列の確保
            mapping_data = np.zeros(shape=(size.x, size.y))
            for it in pixel_data:
                mapping_data[it.x][it.y] = it.z

        def Average(array):
            return np.nanmean(array)

        # 行,列についてすべての平均値を取って矯正
        if flatten == BaseAnalyzer.FlattenMode.Average:
            col_ave = []
            for col in mapping_data:
                col_ave.append(Average(col))

            col_ave_standard = Average(col_ave)

            row_ave = []
            for i in range(0, size.y):
                row_ave.append(Average([row[i] for row in mapping_data]))
            row_ave_standard = Average(row_ave)

            for i in range(0, size.x):
                for j in range(0, size.y):
                    mapping_data[i][j] += col_ave_standard - col_ave[i] + row_ave_standard - row_ave[j]

        # 傾きの矯正
        array_x = np.linspace(0, size.x, size.x)
        array_y = np.linspace(0, size.y, size.y)
        coef_x, coef_y = 0, 0
        if flatten == BaseAnalyzer.FlattenMode.LinearFit:
            coef1, coef2 = BaseAnalyzer.get_flatten_param(mapping_data, flatten=flatten, flatten_param=flatten_param)
            mapping_data = BaseAnalyzer.flatten_plane(mapping_data, coef1, coef2)

        if flatten == BaseAnalyzer.FlattenMode.PolyFit:

            for i in range(0, size.y):
                coef = np.polyfit(array_x, mapping_data[i], BaseAnalyzer.topo_poly_order)
                mapping_data[i, :] -= np.polyval(coef, array_x)

        if flatten == BaseAnalyzer.FlattenMode.ThreePointsFlatten:
            coef1, coef2 = BaseAnalyzer.get_flatten_param(mapping_data, flatten=flatten, flatten_param=flatten_param)

            for i in range(0, size.y):
                mapping_data[i, :] -= np.polyval(coef1, array_x)
            for i in range(0, size.x):
                mapping_data[:, i] -= np.polyval(coef2, array_y)

        # Smooth
        # if smooth is True:
        #     mapping_data = ndimage.gaussian_filter(mapping_data, BaseAnalyzer.gaussian_filter_sigma)

        # Vector3のインスタンスに矯正した実験データを入れる
        data = []
        max_value = sys.float_info.min
        min_value = sys.float_info.max

        for i in range(0, size.x):
            for j in range(0, size.y):
                depth_fitting = mapping_data[i][j]

                if depth_fitting > max_value:
                    max_value = depth_fitting
                if depth_fitting < min_value:
                    min_value = depth_fitting
                data.append(Vector3(i, j, depth_fitting))

        for v in data:
            v.z -= min_value

        # print("Finish Fitting Data ================================")
        return data

    @staticmethod
    def flatten_gui(mapping_data):
        fig, ax = plt.subplots()
        ax.imshow(mapping_data)
        coef_x, coef_y = BaseAnalyzer.get_flatten_param(mapping_data, BaseAnalyzer.FlattenMode.LinearFit)
        root = tkinter.Tk()
        root.withdraw()

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack()

        def set_text(e, text):
            e.delete(0, tkinter.END)
            e.insert(0, text)
            return

        def on_update_button(map):
            m = BaseAnalyzer.flatten_plane(map, float(X_Field.get()), float(Y_Field.get()))
            ax.imshow(m)
            canvas.draw()

        def on_ok_button():
            root.quit()

        X_Label = tkinter.Label(root, text="coef x")
        X_Label.pack()
        X_Field = tkinter.Entry(root)
        X_Field.pack()
        Y_Label = tkinter.Label(root, text="coef y")
        Y_Label.pack()
        Y_Field = tkinter.Entry(root)
        Y_Field.pack()
        set_text(X_Field, coef_x)
        set_text(Y_Field, coef_y)
        on_update_button(mapping_data)

        UpdateButton = tkinter.Button(root, text="update", command=lambda: on_update_button(mapping_data))
        UpdateButton.pack()
        OKButton = tkinter.Button(root, text="ok", command=lambda: on_ok_button())
        OKButton.pack()

        root.update()
        root.deiconify()
        root.mainloop()
        print("x_coef", float(X_Field.get()), "y_coef", float(Y_Field.get()))

    @staticmethod
    def create_texture(pixel_data: List[Vector3], size: Vector2, name="texture", dir=""):
        print("Texture Size", size)
        image = np.zeros((size.x, size.y, 3), np.uint8)

        maxDepth = sys.float_info.min
        minDepth = sys.float_info.max

        for it in pixel_data:
            if it.z > maxDepth:
                maxDepth = it.z
            if it.z < minDepth:
                minDepth = it.z

        for v in pixel_data:
            color = (v.z - minDepth) / (maxDepth - minDepth) * 255

            image.itemset((v.x, v.y, 0), color)
            image.itemset((v.x, v.y, 1), color)
            image.itemset((v.x, v.y, 2), color)

        plt.imshow(image)
        plt.savefig(os.path.join(dir, name + '.png'))


class GridAnalyzer(BaseAnalyzer):

    def __init__(self, directory, fileNamesList=None, **kwargs):
        super(GridAnalyzer, self).__init__(directory, fileNamesList)

        self.fileDict = {}
        self._topo_cache = {}
        self.topo_flatten_mode = self.FlattenMode.PolyFit
        self.topo_apply_smooth = True
        self._custom_topo = None
        self._custom_3darray = None

        BaseAnalyzer.topo_poly_order = 3
        BaseAnalyzer.gaussian_filter_sigma = 0.5

        if "topo_flatten_mode" in kwargs:
            self.topo_flatten_mode = kwargs["topo_flatten_mode"]
        if "topo_apply_smooth" in kwargs:
            self.topo_apply_smooth = kwargs["topo_apply_smooth"]
        if "topo_poly_order" in kwargs:
            BaseAnalyzer.topo_poly_order = kwargs["topo_poly_order"]
        if "gaussian_filter_sigma" in kwargs:
            BaseAnalyzer.gaussian_filter_sigma = kwargs["gaussian_filter_sigma"]
        if "custom_topo" in kwargs:
            self._custom_topo = kwargs["custom_topo"]
        if "custom_3darray" in kwargs:
            self._custom_3darray = kwargs["custom_3darray"]

    def OpenFile(self, searchAllDirectory=True):
        if not searchAllDirectory:
            if len(self.fileNamesList) == 0:
                print("Length of File Name List is 0")
                exit(0)
            for fileName in self.fileNamesList:
                self.__OpenFile(fileName)
        else:
            files = [file for file in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, file))]
            for fileName in files:
                extension = os.path.splitext(fileName)[1]
                if extension == '.3ds':
                    self.__OpenFile(fileName)

    def get_curve(self, point=(0, 0), fileName=None):
        if fileName is None and len(self.fileDict.values()) > 0:
            fileName = list(self.fileDict.keys())[0]
        elif fileName not in self.fileDict:
            raise ValueError("fileName error")

        # xy = self.get_xy_count(fileName)
        if self._custom_3darray is None:
            y = self.fileDict[fileName].signals["Frequency Shift (Hz)"][point[1], point[0], :]
        else:
            y = self._custom_3darray[point[1], point[0], :]
        return self.__get_xdata(self.fileDict[fileName], point, self.topo(fileName)), y

    def get_data_count(self, fileName):
        return self.fileDict[fileName].header["num_sweep_signal"]

    def get_xy_count(self, fileName):
        return self.fileDict[fileName].header["dim_px"]

    def get_height_range(self, fileName):
        minValue = 999
        maxValue = 0
        for i in range(0, self.get_xy_count(fileName)[0]):
            for j in range(0, self.get_xy_count(fileName)[1]):
                x, y = self.get_curve((i, j), fileName)
                if np.min(x) < minValue:
                    minValue = np.min(x)
                if np.max(x) > maxValue:
                    maxValue = np.max(x)
        return minValue - np.min(self.topo(fileName)), maxValue - np.min(self.topo(fileName))

    @staticmethod
    def __get_xdata(file, point, topo):
        # print(np.min(file.signals["topo"]), np.max(file.signals["topo"]))
        data_num = file.header["num_sweep_signal"]
        param = file.signals["params"][point[1], point[0], :]
        # print(param[0], param[1], param[4])
        x = np.linspace(param[0], param[1], data_num) + topo[point[1]][point[0]]

        x -= np.min(topo)
        return x * 10 ** 10

    def get_df_matrix(self, fileName):
        if self._custom_3darray is None:
            return self.fileDict[fileName].signals["Frequency Shift (Hz)"]
        else:
            return self._custom_3darray

    def topo(self, fileName):
        if self._custom_topo is not None:
            return self._custom_topo

        if fileName not in self._topo_cache:
            self._topo_cache[fileName] = self.fitting_map(self.fileDict[fileName].signals["topo"],
                                                          flatten=self.topo_flatten_mode)
            # return self.fileDict[fileName].signals["topo"]
        return self._topo_cache[fileName]

    def CalcFMap(self, fileName, param:fsh.formula.measurement_param):
        cvt = fsh.PakConvertor(fileName, fsh.default_project_path, False)

        xy = self.get_xy_count(fileName)
        header = fsh.PakConvertor.PakHeader(data_size=xy, data_count=self.get_data_count(fileName))
        header.comment = "3D force map, Calculate by GridAnalyzer"
        cvt.set_header(header)
        for i in range(0, xy[0]):
            for j in range(0, xy[1]):
                x, y = self.get_curve(point=(i, j), fileName=fileName)
                x = np.flipud(x)
                y = np.flipud(y)
                f = fsh.formula.CalcForceCurveSadar(y, param)
                cvt.add_data((i,j), force_curve(x=x, y=f))

                print("finished", (i, j), "/", xy)
        cvt.save()

    def __OpenFile(self, fileName):
        file_path = os.path.join(self.directory, fileName)
        file = nap.read.Grid(file_path)
        if file is None:
            print(fileName + "File Not Exist")
            exit(0)
        print(fileName)
        # print(file.signals["Frequency Shift (Hz)"].shape)
        # print(file.header)
        self.fileDict[fileName] = file


class SxmAnalyzer(BaseAnalyzer):

    class AnalyzeMode(Enum):
        Z = 1
        NanoSurf = 2
        Nanosurf_Amplitude = 3
        Nanosurf_Dissipation = 4
        Current = 5
        Phase = 6
        Amplitude = 7
        Frequency_Shift = 8
        Excitation = 9

    def __init__(self, directory, fileNamesList=None):
        super(SxmAnalyzer, self).__init__(directory, fileNamesList)

    def SaveTextureFiles(self, _analyzeMode=AnalyzeMode(1), searchAllDirectory=True):
        if not searchAllDirectory:
            if len(self.fileNamesList) == 0:
                print("Length of File Name List is 0")
                exit(0)
            for fileName in self.fileNamesList:
                self.__SaveFile(fileName, _analyzeMode.name, 'forward')
                self.__SaveFile(fileName, _analyzeMode.name, 'backward')
        else:
            files = [file for file in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, file))]
            for it in files:
                extension = os.path.splitext(it)[1]
                if extension == '.sxm':
                    self.__SaveFile(os.path.splitext(it)[0], _analyzeMode.name, 'forward')
                    self.__SaveFile(os.path.splitext(it)[0], _analyzeMode.name, 'backward')

    def __SaveFile(self, fileName, firstKey, secondKey):

        file_path = os.path.join(self.directory, fileName + ".sxm")

        file = nap.read.Scan(file_path)
        if file is None:
            print(fileName + "File Not Exist")
            exit(0)

        pixel_data = []
        data_size = Vector2(file.header['scan_pixels'][0], file.header['scan_pixels'][1])
        for i in range(0, file.header['scan_pixels'][0]):
            for j in range(0, file.header['scan_pixels'][1]):
                pixel_data.append(Vector3(i, j, file.signals[firstKey][secondKey][i][j]))

        self.create_texture(self.fitting_data(pixel_data, data_size), data_size,
                            dir=self.directory_out,
                            name=fileName + "_" + firstKey + "_" + secondKey)


class DulcineaAnalyzer(BaseAnalyzer):
    """""""""""""""""
    Note:
        top: Topography
        ch1: df
        ch2: Amplitude
        ch3: Dissipation
        ch4: Current

    """""""""""""""""
    def __init__(self, directory, fileNamesList=None):
        super(DulcineaAnalyzer, self).__init__(directory, fileNamesList)

    def SaveTextureFiles(self, searchAllDirectory=True):
        if not searchAllDirectory:
            if len(self.fileNamesList) == 0:
                print("Length of File Name List is 0")
                exit(0)
            for fileName in self.fileNamesList:
                extension = os.path.splitext(fileName)[1]
                if extension == '.top':
                    self.__SaveFile(fileName, True)
                elif extension.find("ch") != -1:
                    self.__SaveFile(fileName, False)
        else:
            files = [file for file in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, file))]
            for it in files:
                extension = os.path.splitext(it)[1]
                if extension == '.top':
                    self.__SaveFile(os.path.splitext(it)[0] + extension, True)
                elif extension == '.ch1':
                    self.__SaveFile(os.path.splitext(it)[0] + extension, False)

    def __SaveFile(self, fileName, doFitting=True, saveFile=True):

        def get_value(key: str):
            value = [x for x in header if x.find(key) != -1]
            if len(value) > 0:
                return value[0].split(":")[1]
            else:
                return None

        file_path = os.path.join(self.directory, fileName)
        print("Opening " + fileName + " ================================")

        with open(file_path, 'rb') as f:
            file = f.read().split(b'[Header end]')
            header = str(file[0]).replace(" ", "").split('\\r\\n')
            content = file[1]

            data_list = np.frombuffer(content, dtype='<h')
            # print(len(data_list)-1)
            """
                binaryデータの変換について
                    short型で読み込んで単位はm voltage(mV)です。
                    short型の32767は10V に対応します。

                    voltageから周波数シフトに変換するとき　72.24 Hz/V のPLLを使っています.

                    20180906以前のマッピングデータはshort型で出力しているので
                    ㎐ に変換する時　/ 3.2767  * 72.24 / 1000　の倍数関係を利用する

            """
            # V
            data_list = data_list[1:] / 3.2767 / 1000
            """
            data_list = []
            isJumpCode = False
            for i in range(2, len(content)):
                if not isJumpCode:
                    import struct
                    value = struct.unpack('>H', bytes([content[i],content[i+1]]))[0]
                    data_list.append(value)
                    isJumpCode = True
                else:
                    isJumpCode = False
            """
            data_size = Vector2(int(get_value("Numberofcolumns")), int(get_value("Numberofrows")))

            rowIndex = 0
            columnIndex = 0
            pixel_data = []

            for data in data_list:
                pixel_data.append(Vector3(data_size.y - columnIndex - 1, data_size.x - rowIndex - 1, data))

                rowIndex += 1
                if rowIndex == data_size.y:
                    rowIndex = 0
                    columnIndex += 1
                    if columnIndex >= data_size.y:
                        break
        if doFitting:
            pixel_data = self.fitting_data(pixel_data, data_size)

        if saveFile:
            self.create_texture(pixel_data, data_size, dir=self.directory_out, name=fileName)

        return pixel_data, data_size, header

    def GetDfMap(self, fileName, N=1, saveFile=False):
        result = self.__SaveFile(fileName, False)
        # print(result[2])

        def get_value(key: str):
            value = [x for x in result[2] if x.find(key) != -1]
            if len(value) > 0:
                return value[0].split(":")[1]
            else:
                return None

        if "\\xc5" in get_value("YAmplitude"):
            height_amp = float(get_value("YAmplitude").replace("\\xc5", "")) * 40.9 * 0.1
        elif "nm" in get_value("YAmplitude"):
            height_amp = float(get_value("YAmplitude").replace("nm", "")) * 40.9
        else:
            raise ValueError("Can not read YAmplitude unit in header")
        print("height", height_amp)
        data = np.zeros(shape=(result[1].x, result[1].y))
        for it in result[0]:
            # data[it.x][it.y] = (it.z - minDepth) / (maxDepth - minDepth) * -1
            data[it.x][it.y] = it.z

        mapper = DulcineaMapCreator()
        mapper.N = N
        data = mapper.CalcDfsMap(data)
        # plt.imshow(data)
        # plt.colorbar()
        # plt.show()
        index = np.zeros(shape=result[1].x)
        for i in range(0, result[1].x):
            index[i] = i / result[1].x * height_amp

        if saveFile:
            pd.DataFrame(data, index=index).to_csv(fsh.default_data_path + fileName + ".csv")
            temp = fsh.StandardConvertor(fileName=fileName + ".csv")

            pak = fsh.PakConvertor(fileName, fsh.default_project_path, autoLoad=False)
            for i in range(1, 1025):
                pak.add_data(i, fsh.structures.force_curve(temp.get_x_data(i), temp.get_y_data(i)))

            header = fsh.PakConvertor.PakHeader(data_size=result[1].x, data_count=result[1].y)
            pak.set_header(header)
            pak.save()
            del pak, temp
            os.remove(fsh.default_data_path + fileName + ".csv")

        return data, index

    def GetMap(self, fileName):
        if len(self.fileNamesList) == 0 or fileName not in self.fileNamesList:
            print("Length of FileNameList is 0 or error fileName in FileNameList")
            exit(0)
        extension = os.path.splitext(fileName)[1]
        if extension == '.top':
            pixel_data, size, _ = self.__SaveFile(fileName, True, False)
        elif extension.find("ch") != -1:
            pixel_data, size, _ = self.__SaveFile(fileName, False, False)
        else:
            raise ValueError()
        image = np.zeros((size.x, size.y))
        maxDepth = sys.float_info.min
        minDepth = sys.float_info.max
        for it in pixel_data:
            if it.z > maxDepth:
                maxDepth = it.z
            if it.z < minDepth:
                minDepth = it.z
        for v in pixel_data:
            color = (v.z - minDepth) / (maxDepth - minDepth) * 255
            image[v.x, v.y] = color
        return image

    def CalcFMap(self, fileName, saveFileName, f0, amp, k, N=1, method="sadar", useFilter=False, filterWindowSize=31):
        result = self.__SaveFile(fileName, False)

        def get_value(key: str):
            value = [x for x in result[2] if x.find(key) != -1]
            if len(value) > 0:
                return value[0].split(":")[1]
            else:
                return None
        height_amp = float(get_value("YAmplitude").replace("\\xc5", "").replace("nm", "")) * 40.9 * 0.1

        index = np.zeros(shape=result[1].x)
        for i in range(0, result[1].x):
            index[i] = i / result[1].x * height_amp

        param = measurement_param(height_amp, result[1].y)
        param.amp = amp
        param.f0 = f0
        param.k = k

        cvt = fsh.PakConvertor(saveFileName, autoLoad=False)
        header = fsh.PakConvertor.PakHeader(data_size=result[1].x, data_count=result[1].y)
        header.f0 = f0
        header.amp = amp
        header.k = k
        header.comment = "2D force map, Calculate by DulcineaAnalyzer"
        cvt.set_header(header)

        map, x = self.GetDfMap(fileName, N=N)
        for i in range(0, 1024):
                y = map[:, i]
                if useFilter:
                    y = formula.filter_1d.savitzky_golay_fliter(y, filterWindowSize)

                if method == "sadar":
                    f = formula.CalcForceCurveMatrix(y, param)
                elif method == "matrix":
                    f = formula.CalcForceCurveSadar(y, param)
                else:
                    raise ValueError("no known method")

                # plt.clf()
                # plt.axhline(y=0, c="black")
                # plt.plot(x, formula.CalcForceCurveMatrix(y, param), c="b")
                # plt.plot(x, formula.CalcForceCurveSadar(y, param), c="r")
                # plt.show()
                cvt.add_data((i), force_curve(x=x, y=f))
                print("finished", i, "/", 1024)
        cvt.save()

    def SmoothFMap(self, fileName, f_path, method="ave", method_param=20):
        result = self.__SaveFile(fileName, False)

        def get_value(key: str):
            value = [x for x in result[2] if x.find(key) != -1]
            if len(value) > 0:
                return value[0].split(":")[1]
            else:
                return None

        height_amp = float(get_value("YAmplitude").replace("\\xc5", "")) * 40.9 * 0.1
        index = np.zeros(shape=result[1].x)
        for i in range(0, result[1].x):
            index[i] = i / result[1].x * height_amp

        data = np.asarray(pd.read_csv(f_path).values)[:, 1:]
        if method == "ave":
            F = formula.filter_2d.SmoothMap(data, method_param)
        elif method == "fft":
            F = formula.filter_2d.FFTMap(data, method_param, index)
        elif method == "fft2":
            F = formula.filter_2d.FFT2Map(data, method_param)
        else:
            print("SmoothFMap method error")
            return
        pd.DataFrame(F, index=index).to_csv("Force_smooth-" + method + ".csv")
