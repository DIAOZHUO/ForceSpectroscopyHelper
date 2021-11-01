from . import formula
from . import structures
from . import filter_1d
from . import filter_2d
import os

def set_dir(root_directory_path, data_folder_path):
    f = open(os.path.dirname(__file__) + "/.setting", "w")
    f.writelines([root_directory_path.get(), " ", data_folder_path.get()])
    f.close()


def delete_dir_setting():
    os.remove(os.path.dirname(__file__) + "/.setting")


# default_project_path = str(Path(__file__).parent.parent).replace("\\", "/") + "/"
if not os.path.exists(os.path.dirname(__file__) + "/.setting"):
    import tkinter, tkinter.filedialog
    __root = tkinter.Tk()
    __root.title("ForceCurve Initialization Setting")
    __root.geometry("400x200")

    def __browse_button1():
        filename = tkinter.filedialog.askdirectory()
        __folder_path1.set(filename)
    def __browse_button2():
        filename = tkinter.filedialog.askdirectory()
        __folder_path2.set(filename)

    def __ok_button():
        print(__folder_path1.get(), __folder_path2.get())
        f = open(os.path.dirname(__file__) + "/.setting", "w")
        f.writelines([__folder_path1.get(), " ", __folder_path2.get()])
        f.close()
        __root.destroy()

    def __cancel_button():
        __root.destroy()

    __folder_path1 = tkinter.StringVar()
    tkinter.Label(text="welcome to use fsh!!!", foreground="#00ff00").grid(row=0, column=1)
    tkinter.Label(text="1. please select project root directory").grid(row=1, column=1)
    tkinter.Label(master=__root, textvariable=__folder_path1).grid(row=2, column=1)
    __button1 = tkinter.Button(text="Browse", command=__browse_button1)
    __button1.grid(row=2, column=0)

    __folder_path2 = tkinter.StringVar()
    tkinter.Label(text="2. please select anaysis data directory").grid(row=3, column=1)
    tkinter.Label(master=__root, textvariable=__folder_path2).grid(row=4, column=1)
    __button2 = tkinter.Button(text="Browse", command=__browse_button2)
    __button2.grid(row=4, column=0)

    __button3 = tkinter.Button(text="OK", command=__ok_button)
    __button3.grid(row=5, column=0)
    __button3 = tkinter.Button(text="Cancel", command=__cancel_button)
    __button3.grid(row=5, column=1)
    __root.mainloop()

if os.path.exists(os.path.dirname(__file__) + "/.setting"):
    __setting = open(os.path.dirname(__file__) + "/.setting").readline().split(" ")
    default_project_path = __setting[0] + "/"
    default_data_path = __setting[1] + "/"
else:
    default_project_path = os.path.dirname(__file__)
    print("Warning! 'default_project_path' not defined...")
    print("--", "temporary set default_project_path=", os.path.dirname(__file__))
    default_data_path = os.path.dirname(__file__)
    print("Warning! 'default_data_path' not defined...")
    print("--", "temporary set default_data_path=", os.path.dirname(__file__))


from .converter import *
from .analyzer import *
from .visualizer import *
from .DataSerializer import DataSerializer

