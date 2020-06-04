from ForceCurveConventor.analyzer import DulcineaAnalyzer
from ForceCurveConventor.analyzer import GridAnalyzer
from ForceCurveConventor.analyzer import SxmAnalyzer
from ForceCurveConventor.visualizer import NanonisGridVisualizer


def test_sxm():
    mode = SxmAnalyzer.AnalyzeMode.Z
    analyzer = SxmAnalyzer("C:/Users/DIAO ZHUO/Desktop", mode)
    analyzer.SaveTextureFiles()


def test_dul():
    file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101213/"
    # file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101124/"

    analyzer = DulcineaAnalyzer(file_path)
    # analyzer.SaveTextureFiles(searchAllDirectory=True)
    # analyzer.SaveDfMap("MyFile_0020.f.ch1")
    analyzer.CalcFMap("Si_H_0041.f.ch1", "Force_H_0041.f.ch1", N=3, f0=159133, amp=225, k=31.8)
    # analyzer.CalcFMap("MyFile_0020.f.ch1", "Force_0020.f.ch1", N=3, f0=169913, amp=207, k=38.7)

def smooth_FMap():
    file_path = "A:/Document/data/20100820"

    analyzer = DulcineaAnalyzer(file_path)
    analyzer.SmoothFMap("Si_H_0021.f.ch1", "Si_H_0021.f.ch1.csv", method="fft2", method_param=20)
    # analyzer.SmoothFMap("MyFile_0020.f.ch1", "Force_0020.f.ch1-smooth.csv", method="fft", method_param=150)
    # analyzer.SmoothFMap("MyFile_0020.f.ch1", "MyFile_0020.f.ch1-ave.csv", method="fft", method_param=150)


def test_grid():
    file_path = "./"
    # fileName = "GridSpectroscopy004_C60_Good.3ds"
    fileName = "GridSpectroscopy002.3ds"
    analyzer = GridAnalyzer(file_path, fileNamesList=[fileName])
    analyzer.OpenFile(searchAllDirectory=False)

    NanonisGridVisualizer(fileName, analyzer)
    # print(analyzer.fileDict[fileName].header)
    # import matplotlib.pyplot as plt
    # plt.imshow(analyzer.topo(fileName))
    # plt.colorbar()
    # plt.show()


test_dul()
# smooth_FMap()
# test_grid()