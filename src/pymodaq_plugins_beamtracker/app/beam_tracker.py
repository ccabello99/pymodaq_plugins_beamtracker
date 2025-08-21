from pymodaq.utils import gui_utils as gutils
import argparse
from qtpy import QtWidgets
from pymodaq_gui.utils.utils import mkQApp
import pymodaq_plugins_beamtracker.app.beam_tracker_1 as bt1
import pymodaq_plugins_beamtracker.app.beam_tracker_2 as bt2


def main():
    parser = argparse.ArgumentParser(description="Beam Tracker application")
    parser.add_argument(
        "--mode", "-m",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of cameras to use (1 or 2). Default = 1"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_template",
        help="Name of config file to use"
    )        
    args = parser.parse_args()

    app = mkQApp("BeamTracker")

    mainwindow = QtWidgets.QMainWindow()
    dockarea = gutils.DockArea()
    mainwindow.setCentralWidget(dockarea)

    if args.mode == 1:
        prog = bt1.BeamTracker1(dockarea, args.config)
    elif args.mode == 2:
        prog = bt2.BeamTracker2(dockarea, args.config)
    mainwindow.show()
    app.exec()


if __name__ == "__main__":
    main()