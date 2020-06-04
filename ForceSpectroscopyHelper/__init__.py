from . import formula
from . import structures
import os


# default_project_path = str(Path(__file__).parent.parent).replace("\\", "/") + "/"
if not os.path.exists(os.path.dirname(__file__) + "/.setting"):
    import tkinter, tkinter.filedialog
    root = tkinter.Tk()
    root.title("ForceCurve Initialization Setting")
    root.geometry("400x200")

    def browse_button1():
        filename = tkinter.filedialog.askdirectory()
        folder_path1.set(filename)
    def browse_button2():
        filename = tkinter.filedialog.askdirectory()
        folder_path2.set(filename)

    def ok_button():
        print(folder_path1.get(), folder_path2.get())
        f = open(os.path.dirname(__file__) + "/.setting", "w")
        f.writelines([folder_path1.get(), " ", folder_path2.get()])
        f.close()
        root.destroy()

    folder_path1 = tkinter.StringVar()
    l = tkinter.Label(text="welcome to use fsh!!!", foreground="#00ff00")
    l.grid(row=0, column=1)
    l1 = tkinter.Label(text="1. please select project root directory")
    l1.grid(row=1, column=1)
    lbl1 = tkinter.Label(master=root, textvariable=folder_path1)
    lbl1.grid(row=2, column=1)
    button1 = tkinter.Button(text="Browse", command=browse_button1)
    button1.grid(row=2, column=0)

    folder_path2 = tkinter.StringVar()
    l2 = tkinter.Label(text="2. please select anaysis data directory")
    l2.grid(row=3, column=1)
    lbl2 = tkinter.Label(master=root, textvariable=folder_path2)
    lbl2.grid(row=4, column=1)
    button2 = tkinter.Button(text="Browse", command=browse_button2)
    button2.grid(row=4, column=0)

    button3 = tkinter.Button(text="OK", command=ok_button)
    button3.grid(row=5, column=0)
    root.mainloop()


setting = open(os.path.dirname(__file__) + "/.setting").readline().split(" ")
default_project_path = setting[0]
default_data_path = setting[1]


from .converter import *
from .analyzer import *
from .visualizer import *