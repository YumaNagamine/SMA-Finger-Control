"""
General functions for all system.
"""

import os
import sys
import time
from typing import Any, Mapping, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

RUNTIME = time.time()
IMG_FOLDER = "./img/"
FIG_FOLDER = "./LOG/"
DATA_FOLDER = "./LOG/"
FIG_SIZE = (38.4, 21.6)  # (21.6,14.4)#(19.2,10.8)
FONTSIZE = 40


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _build_file_url(folder: str, file_name: str, f_type: str) -> str:
    if f_type not in file_name:
        file_url = os.path.join(folder, f"{file_name}.{f_type}")
    else:
        file_url = os.path.join(folder, file_name)
    if os.path.exists(file_url):
        file_url = os.path.join(folder, f"{file_name}2.{f_type}")
    return file_url


class Logger(object):
    def __init__(self, filename: Optional[str] = None):
        _ensure_dir(FIG_FOLDER)

        if filename is None:
            filename = os.path.join(
                FIG_FOLDER,
                time.strftime("_%m_%d_%H.%M.%S", time.localtime(RUNTIME)) + ".txt",
            )
        print(filename)
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

    def _eg_codes():
        path = os.path.abspath(os.path.dirname(__file__))
        type = sys.getfilesystemencoding()
        sys.stdout = Logger()

# Save expriment/sensor massive data 
def saveData(data, file_name, labels, f_type: str = "csv", **kwargs: Mapping[str, Any]):
    _ensure_dir(FIG_FOLDER)
    file_url = _build_file_url(FIG_FOLDER, file_name, f_type)
    header = str(labels).replace("'", "").replace("]", "").replace("[", "")
    if "csv" in f_type:
        np.savetxt(
            file_url,
            data,
            fmt="%.18e",
            delimiter=",",
            newline="\n",
            header=header,
            footer="",
            comments="# ",
            encoding=None,
        )

    return file_url


def saveFigure(
    data, file_name, labels, show_img=False, figure_mode="Double", **kwargs: Mapping[str, Any]
):
    _ensure_dir(FIG_FOLDER)
    data_np = np.array(data)
    data_size = data_np.shape
    print("Saving figure for sampled data:", data_size)

    figsize = FIG_SIZE
    if not data_size[1] > 3:
        figsize = (FIG_SIZE[1] * 2.35, FIG_SIZE[1])
    
    # if data_size[1]<2: ValueError

    plt.rcParams.update({"font.size": FONTSIZE})  # 改变所有字体大小

    plt.figure(figsize=figsize,dpi=100)
    plt.clf()

    if figure_mode == "Double":  # Muti-Dim(a,b,...,t)

        ax = plt.subplot(211)
        # plt.xticks(fontsize=FONTSIZE); plt.yticks(fontsize=FONTSIZE)
        # ax.legend(fontsize=FONTSIZE)
        
        for _i in range(3):
            i = _i + 1
            plt.plot(data_np[:, -1], data_np[:, i], label=labels[i])
        plt.legend(
            loc="upper center", ncol=3, frameon=False, shadow=False, framealpha=0
        )
        # plt.xlabel(labels[-1])

        ax = plt.subplot(212)
        # plt.xticks(fontsize=FONTSIZE); plt.yticks(fontsize=FONTSIZE)
        for _i in range(3):
            i = _i + 4
            plt.plot(data_np[:, -1], data_np[:, i], label=labels[i])
        # ax.legend(fontsize=FONTSIZE)
        plt.legend(
            loc="upper center", ncol=3, frameon=False, shadow=False, framealpha=0
        )
        plt.xlabel(labels[-1])

    elif figure_mode == "Single":  # One dim (x,t)
        # ax = plt.subplot(111)
        t_end = int(data_np[-1, -1])
        plt.rcParams.update(
            {"font.size": int(FONTSIZE * 1.4)}
        )  # 改变所有字体大小，改变其他性质类似
        x_locator = [
            MultipleLocator(t_end * 0.1),
            MultipleLocator(t_end * 0.025),
        ]  # Major / Minor

        ax = plt.gca()
        ax.xaxis.set_major_locator(x_locator[0])
        ax.xaxis.set_minor_locator(x_locator[1])

        # y_locator = [MultipleLocator(20),MultipleLocator(10)]
        # ax.yaxis.set_major_locator(y_locator[0]);ax.yaxis.set_minor_locator(y_locator[1])
        for i in range(data_size[1] - 1):

            plt.plot(data_np[:, -1], data_np[:, i], label=labels[i])
            # plt.scatter(data_np[:,-1],data_np[:,i],label =labels[i],s=10,alpha=0.4)

        plt.legend()
        plt.grid(which="major")
        plt.grid(which="minor", alpha=0.4)
        plt.xlabel(labels[-1])
        # print("'t_end'",x_locator)


    f_type = "pdf"
    file_url = _build_file_url(FIG_FOLDER, file_name, f_type)

    print("Figure saving as: ", FIG_FOLDER + file_name)
    # plt.margins(0,0)
    plt.savefig(file_url, bbox_inches="tight", pad_inches=0.1)
    if show_img:
        plt.show()
    
    # saveData(data,file_name,labels)

    return file_url
