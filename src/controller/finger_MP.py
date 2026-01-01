#!/usr/bin/env python3
"""Multiprocess controller for SMA finger."""

import multiprocessing as mp
import os
import sys
import time

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from pyftdi import i2c
from pca9685.pca9685 import Pca9685_01 as PWMGENERATOR
from utils.generalfunctions import Logger

DO_PLOT = False
VOT = 9
LOAD = 20
EXIT_CHECK_FREQ = 0.5

ACT_TYPE = 2
ACT_TYPES = {0: "SMA", 1: "TSMA", 2: "CTSMA", 3: "SSA"}

if "RUNTIME" not in globals():
    RUNTIME = time.time()


def print_info(dutys, intervals):
    print("Experiment data:\n")
    print("\t act_type:\t", ACT_TYPES[ACT_TYPE])
    print("\t DUTYS:    \t", dutys, "%")
    print("\t INTERVALS:\t", intervals, "Sec")
    print("\t VOT: \t", VOT, "Volts")
    print("\t LOAD: \t", LOAD, "Grams")
    print("\t DO_PLOT:\t", DO_PLOT)
    print("\t EXIT_CHECK_FREQ:\t", EXIT_CHECK_FREQ)
    print("\n\n")


def experiment_bio_01(actuator_device):
    flexsion_ch = [0x0, 0x2]
    adduction_ch = [0x6]

    dutys_unit = [1, 0]
    intervals_unit = [2, 6]
    dutys, intervals = [], []
    num_cycles = 1

    for _ in range(num_cycles):
        dutys.extend(dutys_unit)
        intervals.extend(intervals_unit)

    print_info(dutys, intervals)

    to_activated = []
    to_activated.extend(flexsion_ch)
    to_activated.extend(adduction_ch)
    to_activated.extend([0x08])

    actuator_device.test_wires(to_activated, dutys, intervals, is_show=False)


def ctrlProcess(i2c_actuator_controller_url=None, _angle_sensor_id="SNS000", _process_share_dict=None):
    process_start = time.time()
    print("\nCtrlProcess Starts:")
    print("", process_start - RUNTIME, "s after runtime:",
          time.strftime("%Y:%m:%d %H:%M:%S", time.localtime(RUNTIME)))

    wire_freq = 1526

    try:
        i2c_device = i2c.I2cController()
        i2c_device.configure(
            i2c_actuator_controller_url,
            frequency=1e6,
            rdoptim=True,
            clockstretching=True,
        )
        print("IIC device configured")
        actuator_device = PWMGENERATOR(i2c_device, debug=False)
    except UsbToolsError as err:
        print("Caught error:", err)
        return False

    actuator_device.setPWMFreq(wire_freq)
    print("Connection established: ", actuator_device)
    return actuator_device


def _print_banner():
    print("Multi process version of SMA finger")
    print("Current time", time.strftime("%Y:%m:%d %H:%M:%S", time.localtime()))
    print("RUNTIME", time.strftime("%m.%dth,%HH:%MM:%SS .%F", time.localtime(RUNTIME)))


def _resolve_urls():
    url_0 = os.environ.get("FTDI_DEVICE", "ftdi://ftdi:232h:0:FF/0")
    url_1 = os.environ.get("FTDI_DEVICE", "ftdi://ftdi:232h:0:FE/0")
    return url_0, url_1


if __name__ == "__main__":
    sys.stdout = Logger()
    sys.stderr = sys.stdout

    _print_banner()
    print("SMA Finger MultiProcess: \nTwo thread with Python threading library")

    url_0, url_1 = _resolve_urls()

    url_control = url_0
    url_sensor = url_1

    if url_control == [] or url_sensor == []:
        print("Failed on finding USB FT232H device addr:", url_control, url_sensor)
        sys.exit(1)
    print("Found USB FT232H device @:", url_control, url_sensor)

    with mp.Manager() as process_manager:
        process_share_dict = process_manager.dict()
        angle_sensor_id = "ADC001"

        process_ctrl = mp.Process(
            target=ctrlProcess,
            args=(url_control, angle_sensor_id, process_share_dict),
        )
        process_ctrl.start()
        process_ctrl.join()
