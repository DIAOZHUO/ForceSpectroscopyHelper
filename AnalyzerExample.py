from ForceSpectroscopyHelper import *


def test_sxm():
    mode = SxmAnalyzer.AnalyzeMode.Z
    analyzer = SxmAnalyzer("C:/Users/DIAO ZHUO/Desktop", mode)
    analyzer.SaveTextureFiles()


def show_dul_topo():
    # file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101213/"
    file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101124/"
    # fileName = "Si_H_0040.f.top"
    fileName = "MyFile_0015.f.top"
    analyzer = DulcineaAnalyzer(file_path, [fileName])
    map = analyzer.GetMap(fileName)
    from scipy.ndimage import gaussian_filter
    plt.imshow(gaussian_filter(map, sigma=1), cmap='gray', interpolation="bilinear", origin="lower")
    plt.axis('off')
    plt.ioff()
    plt.savefig("2.png", Transparent=True, dpi=300)
    # plt.show()


def test_dul():
    # file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101213/"
    # file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101124/"
    file_path = "C:/Users/HatsuneMiku/Downloads/並川君データ/20071217/"

    analyzer = DulcineaAnalyzer(file_path)
    analyzer.SaveTextureFiles(searchAllDirectory=True)
    # analyzer.CalcFMap("Si_H_0041.f.ch1", "Force_H_0041.f.ch1", N=3, f0=159133, amp=225, k=31.8)
    # analyzer.CalcFMap("MyFile_0020.f.ch1", "Force_0020.f.ch1", N=3, f0=169913, amp=207, k=38.7)


def smooth_FMap():
    file_path = "A:/Document/data/20100820"

    analyzer = DulcineaAnalyzer(file_path)
    analyzer.SmoothFMap("Si_H_0021.f.ch1", "Si_H_0021.f.ch1.csv", method="fft2", method_param=20)
    # analyzer.SmoothFMap("MyFile_0020.f.ch1", "Force_0020.f.ch1-smooth.csv", method="fft", method_param=150)
    # analyzer.SmoothFMap("MyFile_0020.f.ch1", "MyFile_0020.f.ch1-ave.csv", method="fft", method_param=150)

def save_df_file():
    # file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20100820/"
    # analyzer = DulcineaAnalyzer(file_path)
    # analyzer.GetDfMap(fileName="MyFile_0015.f.ch1", saveFile=False)
    file_path = "E:/PythonProjects/ForceSpectroscopyHelper/Data/namikawa/map6/"
    analyzer = DulcineaAnalyzer(file_path)
    analyzer.GetDfMap(fileName="Si(111)_0007.b.ch1", saveFile=True)
    # analyzer.SaveTextureFiles()

def test_grid():
    file_path = "./Data"
    # fileName = "GridSpectroscopy004_C60_Good.3ds"
    fileName = "GridSpectroscopy002.3ds"
    analyzer = GridAnalyzer(file_path, fileNamesList=[fileName])
    analyzer.OpenFile(searchAllDirectory=False)

    NanonisGridVisualizer(fileName, analyzer)


def test_force_convert():
    file_path = "C:/Users/HatsuneMiku/OneDrive - 大阪大学/ForceCurve研究/data/20101213/"
    analyzer = DulcineaAnalyzer(file_path)
    analyzer.CalcFMap("Si_H_0041.f.ch1", "Force_H_0041.f.ch1", N=3, f0=159133, amp=225, k=31.8)


# show_dul_topo()
save_df_file()
# smooth_FMap()
# test_dul()
