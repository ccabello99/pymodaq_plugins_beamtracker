from qtpy import QtWidgets, QtCore, QtGui
import numpy as np
import os
import xml.etree.ElementTree as ET
import tomllib
from pathlib import Path
import platform

from pymodaq.utils import gui_utils as gutils
from pymodaq.utils.config import Config
from pymodaq.utils.logger import set_logger, get_module_name

from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq.utils.gui_utils.widgets.lcd import LCD
from pymodaq.control_modules.daq_viewer import DAQ_Viewer
from pymodaq.utils.data import DataToExport, DataFromPlugins, Axis
from pymodaq_gui.managers.roi_manager import EllipseROI
from pyqtgraph.parametertree import Parameter, ParameterTree
import random

from pymodaq_plugins_beamtracker.utils import Config as PluginConfig
import laserbeamsize as lbs

logger = set_logger(get_module_name(__file__))

main_config = Config()
plugin_config = PluginConfig()


class BeamSizeWorker(QtCore.QObject):
    result_ready = QtCore.Signal(object, int)
    error = QtCore.Signal(str, int)

    @QtCore.Slot(object, int)
    def process(self, img: object, frame_id: int):
        try:
            arr = np.asarray(img)
            values = lbs.beam_size(arr, iso_noise=False)
            self.result_ready.emit(values, frame_id)
        except Exception as e:
            self.error.emit(f"Beam size computation failed: {e}", frame_id)


class BeamTracker1(gutils.CustomApp):
    params = [
        {'title': 'Gaussian overlay', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Show ellipse', 'name': 'show_ellipse', 'type': 'led_push', 'value': False, 'default': False},
            {'title': 'Show lineout', 'name': 'show_lineout', 'type': 'led_push', 'value': False, 'default': False},
            {'title': 'Pixel calibration', 'name': 'pixel_calibration', 
             'type': 'float', 'value': 1.0, 'default': 1.0, 'tip': 'Distance per pixel'},
            {'title': 'Units', 'name': 'units', 'type': 'list', 'limits': ['pixels', 'um', 'mm', 'cm'], 'default': 'pixels'},
            {'title': 'Save current ROI', 'name': 'save_ellipse', 'type': 'bool_push', 'value': False, 'default': False},            
        ]},
        {'title': 'Load saved ROIs', 'name': 'load_saved_rois', 'type': 'group', 'children': [
            {'title': 'Saving Base Path:', 'name': 'load_rois_file', 'type': 'browsepath', 'value': '', 'filetype': True},
            {'title': 'Load ROIs', 'name': 'load_rois', 'type': 'bool_push', 'value': False, 'default': False}
        ]},
        {'title': 'Config Base Path:', 'name': 'config_base_path', 'type': 'browsepath', 'value': '', 'filetype': False},
    ]

    def __init__(self, parent: gutils.DockArea, config_name: str):
        super().__init__(parent)

        self.pixel_calibration = 1.0
        self.units = 'um'

        self.viewer = {
            'viewer': None,
            'roi': None,
            'lcd': None,
            'show_ellipse': False,
            'show_lineout': False,
            'worker': None,
            'thread': None,
            'input': None,
            'latest_frame': None,
            'worker_busy': False,
            '_saving_roi': False,
            '_loading_roi': False,
            'frame_id': 0,
        }
        self.config_name = config_name
        self.config = {}
        self.qsettings = QtCore.QSettings("PyMoDAQ", "BeamTracker")
        if platform.system() == "Windows":
            self.configs_dir = self.qsettings.value('beam_tracker_configs/basepath', 
                                   Path(os.environ.get("USERPROFILE", Path.home())) / "Documents")
        else:
            self.configs_dir = self.qsettings.value('beam_tracker_configs/basepath', 
                                   os.path.join(os.path.expanduser('~'), 'Documents'))
        self.settings.param('config_base_path').setValue(self.configs_dir)

        self.setup_config()
        self.setup_ui()
        self._setup_worker_threads()
        self.show_viewer(True)

    def setup_docks(self):
        self.docks['viewer2D'] = gutils.Dock('Beam Tracker')
        self.dockarea.addDock(self.docks['viewer2D'])
        self.target_viewer = Viewer2D(QtWidgets.QWidget())
        self.docks['viewer2D'].addWidget(self.target_viewer.parent)

        self.docks['lcds'] = gutils.Dock('Beam properties')
        self.dockarea.addDock(self.docks['lcds'], 'right', self.docks['viewer2D'])
        self.lcd = LCD(QtWidgets.QWidget(), Nvals=5, labels=['x0', 'y0', f'd_major ({self.units})',
                                                            f'd_minor ({self.units})', 'phi (deg)'])
        self.docks['lcds'].addWidget(self.lcd.parent)

        self.docks['settings'] = gutils.Dock('Settings')
        self.dockarea.addDock(self.docks['settings'], 'bottom')
        self.docks['settings'].addWidget(self.settings_tree)
       
        cam_window = QtWidgets.QMainWindow() 
        dockarea = gutils.DockArea()
        cam_window.setCentralWidget(dockarea) 
        self.camera_viewer = DAQ_Viewer(dockarea, title='Beam Tracker', daq_type='DAQ2D') 
        self.camera_viewer.detector = self.config['detector']
        self.camera_viewer.settings.child('detector_settings', 'camera_list').setValue(self.config['camera_name'])
        self.camera_viewer.init_signal.connect(lambda _: self._on_viewer_initialized(camera_viewer=self.camera_viewer, target_viewer=self.target_viewer))
        self.camera_viewer.init_hardware(do_init=self.config['do_init'])

    def setup_actions(self): 
        self.add_action('snap', 'Snap Data', 'Snapshot2_32', tip='Click to get one data shot') 
        self.add_action('grab', 'Grab', 'camera', checkable=True, tip="Grab from camera") 
        self.add_action('stop', 'Stop', 'stop', tip="Stop grabbing from camera") 
        self.add_action('quit', 'Quit', 'close2', "Quit program") 
        self.add_action('show', 'Show/hide', 'read2', "Show Hide DAQViewers", checkable=True)

    def connect_things(self):
        self.connect_action('snap', self.camera_viewer.snap)
        self.connect_action('grab', self.camera_viewer.grab)
        self.connect_action('stop', self.camera_viewer.stop)
        self.connect_action('show', lambda do_show: self.show_viewer(do_show))
        self.connect_action('quit', self.quit_app)

        self.camera_viewer.grab_done_signal.connect(
            lambda dte: self._on_new_frame(dte)
        )

    def set_camera_presets(self, camera_viewer, target_viewer):
        cam_props = self.config.get("camera_properties", {})
        for key, val in cam_props.items():
            if isinstance(val, dict):
                for prop, value in val.items():
                    param = camera_viewer.settings.child('detector_settings', str(key), str(prop))
                    if param is not None:
                        param.setValue(value)
                        param.sigValueChanged.emit(param, param.value())
        if self.config['references']['use_references']:
            xml_path = f"{self.configs_dir}/{self.config['references']['reference']}.xml"
            self.load_roi_from_xml(target_viewer, xml_path)

    def setup_config(self):
        config_template_path = Path(__file__).parent.joinpath(f'{self.configs_dir}/{self.config_name}.toml')        
        with open(config_template_path, "rb") as f:
            self.config = tomllib.load(f)


    def _setup_worker_threads(self):
            th = QtCore.QThread(self)
            w = BeamSizeWorker()
            w.moveToThread(th)
            th.start()

            class WorkerInput(QtCore.QObject):
                trigger = QtCore.Signal(object, int)

            inp = WorkerInput()
            inp.trigger.connect(w.process, QtCore.Qt.QueuedConnection)

            w.result_ready.connect(lambda values, fid: self._on_worker_result(values, fid))
            w.error.connect(lambda err_msg, fid: self._on_worker_error(err_msg, fid))

            self.viewer = {
                'viewer': self.target_viewer,
                'roi': None,
                'lcd': self.lcd,
                'show_ellipse': False,
                'show_lineout': False,
                'worker': w,
                'thread': th,
                'input': inp,
                'latest_frame': None,
                'worker_busy': False,
                '_saving_roi': False,
                '_loading_roi': False,
                'frame_id': 0,
            }

    def _on_new_frame(self, dte):
        self.show_data(dte)
        self._kick_analysis()            

    def _kick_analysis(self):
        viewer_info = self.viewer
        viewer = viewer_info['viewer']
        viewer_info['frame_id'] += 1
        fid = viewer_info['frame_id']
        if fid%5 != 0 or viewer_info['latest_frame'] is None or viewer_info['worker_busy']:
            return

        frame = viewer_info['latest_frame']
        try:
            frame_to_process = np.asarray(frame, dtype=frame.dtype).copy(order='C')
        except Exception:
            frame_to_process = frame.copy()

        nrows, ncols = frame_to_process.shape[:2] 
        if viewer.view.ROIselect.isVisible(): 
            x0, y0 = viewer.view.ROIselect.pos() 
            width, height = viewer.view.ROIselect.size() 
            y0 = int(y0) 
            x0 = int(x0) 
            y1 = min(y0 + int(height), nrows) 
            x1 = min(x0 + int(width), ncols) 
        else: 
            x0, y0 = 0, 0 
            y1, x1 = nrows, ncols 
        frame_to_process = frame_to_process[y0:y1, x0:x1]

        viewer_info['worker_busy'] = True
        viewer_info['input'].trigger.emit(frame_to_process, fid)

    @QtCore.Slot(object, int)
    def _on_worker_result(self, values, frame_id):
        info = self.viewer
        try:
            x, y, dmajor, dminor, theta = values
            dmajor_cal = dmajor * self.pixel_calibration
            dminor_cal = dminor * self.pixel_calibration

            info['lcd'].setvalues([np.array([t]) for t in (x, y, dmajor_cal, dminor_cal, theta)])
            self._update_fit_roi(x, y, dmajor, dminor, theta)
        except Exception:
            logger.exception(f"_on_worker_result failed:")
        finally:
            info['worker_busy'] = False


    @QtCore.Slot(str, int)
    def _on_worker_error(self, msg, frame_id):
        logger.error(msg)
        info = self.viewer
        if frame_id != info['frame_id']:
            return
        info['worker_busy'] = False

    def _update_fit_roi(self, x, y, dmajor, dminor, theta):
        viewer_info = self.viewer
        viewer = viewer_info['viewer']

        if (not viewer_info['show_ellipse'] or viewer_info['latest_frame'] is None
            or dmajor == 0 or dminor == 0):
            try:
                viewer.view.plotitem.removeItem(self.viewer['roi'])
            except Exception:
                pass
            return

        if viewer.view.ROIselect.isVisible():
            roi_x0, roi_y0 = viewer.view.ROIselect.pos()
            roi_w, roi_h = viewer.view.ROIselect.size()
        else:
            roi_x0, roi_y0 = 0, 0
            roi_h, roi_w = viewer_info['latest_frame'].shape

        global_x = roi_x0 + x
        global_y = roi_y0 + y
        global_center = QtCore.QPointF(global_x, global_y)

        xmin, xmax, ymin, ymax = self.get_bounding_rect(dmajor/2, dminor/2, theta, center=(global_x, global_y))
        width, height = np.abs(xmax-xmin), np.abs(ymax-ymin)
        size_view = [width, height]

        viewer.view.plotitem.removeItem(self.viewer['roi'])
        roi = EllipseROI(index=0, pos = [global_x - width/2, global_y - width/2], 
                         size = size_view)
        roi.set_center((global_x, global_y))
        roi.setAngle(-theta*180/np.pi, center=(0.5, 0.5))
        roi.setPen(QtGui.QPen(QtGui.QColor(255,0,0),0.05))
        roi.setZValue(10) 
        roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.viewer['roi'] = roi
        viewer.view.plotitem.addItem(roi)
        viewer.view.set_crosshair_position(global_center.x(), global_center.y())
        roi.setVisible(True)

        if viewer_info['show_lineout']:
            gaussian_fit = (x, y, width/2, height/2, theta)
            self.get_crosshair_data_within_roi((roi_x0, roi_y0, roi_w, roi_h), gaussian_fit)

    def get_crosshair_data_within_roi(self, roi_bounds, gaussian_fit):
        viewer_info = self.viewer
        viewer = viewer_info['viewer']
        frame = viewer_info['latest_frame']
        roi_x0, roi_y0, roi_w, roi_h = roi_bounds
        gaussian_x, gaussian_y, gaussian_dx, gaussian_dy, gaussian_theta = gaussian_fit

        centroid = viewer.view.get_crosshair_position()
        cx, cy = int(centroid[0]), int(centroid[1])

        x0, x1 = int(max(0, min(frame.shape[1], roi_x0))), int(max(0, min(frame.shape[1], roi_x0 + roi_w)))
        y0, y1 = int(max(0, min(frame.shape[0], roi_y0))), int(max(0, min(frame.shape[0], roi_y0 + roi_h)))

        x_axis = np.linspace(1, int(roi_w), int(roi_w))
        y_axis = np.linspace(1, int(roi_h), int(roi_h))

        # Horizontal lineout
        if x0 <= cx < x1:
            hor_raw = np.squeeze(frame[cy, x0:x1])
            gauss_hor = gaussian_lineout_x(x_axis, gaussian_x, gaussian_y, gaussian_dx, gaussian_dy, gaussian_theta, cy - roi_y0)
            min_len = min(len(hor_raw), len(gauss_hor))
            hor_raw, gauss_hor = hor_raw[:min_len], gauss_hor[:min_len]
            if np.max(gauss_hor) != 0:
                gauss_hor = gauss_hor * (np.max(hor_raw)/np.max(gauss_hor))
            x_axis_trim = np.linspace(1, len(hor_raw), len(hor_raw))
            dwa_hor = DataFromPlugins(name='hor', data=[hor_raw, gauss_hor], dim='Data1D', labels=['crosshair_hor','gaussian_hor'], axes=[Axis(label='Pixels', data=x_axis_trim)])
            viewer.view.lineout_viewers['hor'].view.display_data(dwa_hor, displayer='crosshair')

        # Vertical lineout
        if y0 <= cy < y1:
            ver_raw = np.squeeze(frame[y0:y1, cx])
            gauss_ver = gaussian_lineout_y(y_axis, gaussian_x, gaussian_y, gaussian_dx, gaussian_dy, gaussian_theta, cx - roi_x0)
            min_len = min(len(ver_raw), len(gauss_ver))
            ver_raw, gauss_ver = ver_raw[:min_len], gauss_ver[:min_len]
            if np.max(gauss_ver) != 0:
                gauss_ver = gauss_ver * (np.max(ver_raw)/np.max(gauss_ver))
            y_axis_trim = np.linspace(1, len(ver_raw), len(ver_raw))
            dwa_ver = DataFromPlugins(name='ver', data=[ver_raw, gauss_ver], dim='Data1D', labels=['crosshair_ver','gaussian_ver'], axes=[Axis(label='Pixels', data=y_axis_trim)])
            viewer.view.lineout_viewers['ver'].view.display_data(dwa_ver, displayer='crosshair')

    def show_data(self, dte: DataToExport):
        info = self.viewer
        try:
            dsrc = dte.get_data_from_source('raw')
            data2d_list = dsrc.get_data_from_dim('Data2D')
            data2d = data2d_list[0]

            arr = data2d[0] if isinstance(data2d, (list, tuple)) else data2d
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            info['viewer'].show_data(data2d)
            info['latest_frame'] = arr
        except Exception as e:
            logger.error(f"show_data failed: {e}")     

    def value_changed(self, param):
        viewer_info = self.viewer
        viewer = viewer_info['viewer']
        fit_roi = viewer_info['roi']
        lcd = viewer_info['lcd']

        if param.name() == 'show_ellipse':
            viewer_info['show_ellipse'] = param.value()
            viewer.view.show_hide_crosshair(show=param.value())

        elif param.name() == 'show_lineout':
            viewer_info['show_lineout'] = param.value()
            if param.value():                
                viewer.view.prepare_image_widget_for_lineouts()
                viewer.view.lineout_viewers['hor'].view.add_data_displayer('crosshair_hor', 'red')
                viewer.view.lineout_viewers['ver'].view.add_data_displayer('crosshair_ver', 'red')
            else:
                viewer.view.prepare_image_widget_for_lineouts(1)
                viewer.view.lineout_viewers['hor'].view.remove_data_displayer('crosshair_hor')
                viewer.view.lineout_viewers['ver'].view.remove_data_displayer('crosshair_ver')

        elif param.name() == 'pixel_calibration':
            self.pixel_calibration = param.value()

        elif param.name() == 'units':
            self.units = param.value()
            new_labels = ['x0', 'y0', f'd_major ({self.units})', f'd_minor ({self.units})', 'phi (deg)']
            qlabels = lcd.parent.findChildren(QtWidgets.QLabel)
            for lbl, new_text in zip(qlabels, new_labels):
                lbl.setText(new_text)

        elif param.name() == 'save_ellipse':
            if viewer_info['_saving_roi'] or fit_roi is None:
                return
            viewer_info['_saving_roi'] = True
            try:
                viewer.view.roi_manager.add_roi_programmatically('EllipseROI')
                roi_dict = viewer.view.roi_manager.ROIs
                latest_key, latest_roi = list(roi_dict.items())[-1]
                pos = fit_roi.pos()
                size = fit_roi.size()
                angle = fit_roi.angle()
                latest_roi.setPos(pos)
                latest_roi.setSize(size)
                latest_roi.setAngle(angle)
                r, g, b = [random.randint(0, 255) for _ in range(3)]
                latest_roi.setPen(QtGui.QPen(QtGui.QColor(r, g, b), 0.1))
            finally:
                viewer_info['_saving_roi'] = False
                param.blockSignals(True)
                param.setValue(False)
                param.blockSignals(False)

        elif param.name() == 'load_rois':
            xml_path = self.settings.child('load_saved_rois', 'load_rois_file').value()
            if not xml_path or not os.path.isfile(xml_path) or viewer_info['_loading_roi']:
                return
            viewer_info['_loading_roi'] = True
            try:
                self.load_roi_from_xml(viewer, xml_path)
            finally:
                viewer_info['_loading_roi'] = False
                param.blockSignals(True)
                param.setValue(False)
                param.blockSignals(False)

        elif param.name() == 'config_base_path':
            self.qsettings.setValue('focal_spot_configs/basepath', param.value())

    def load_roi_from_xml(self, viewer, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for roi_group in root.findall("./*"):  
            roi_type = roi_group.find('roi_type').text

            pos_x = float(roi_group.find('./position/x').text)
            pos_y = float(roi_group.find('./position/y').text)
            width = float(roi_group.find('./size/width').text)
            height = float(roi_group.find('./size/height').text)
            angle = float(roi_group.find('./angle').text)

            color_node = roi_group.find('Color')
            color = None
            if color_node is not None:
                color = eval(color_node.text)

            viewer.view.roi_manager.add_roi_programmatically(roi_type)
            roi_dict = viewer.view.roi_manager.ROIs
            roi_key, roi = list(roi_dict.items())[-1]
            roi.setPos([pos_x, pos_y])
            roi.setSize([width, height])
            roi.setAngle(angle)
            roi.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            if color:
                roi.setPen(color)                    

    def get_bounding_rect(self, a, b, theta, center=(0.0, 0.0)):
        cx, cy = center

        x_max = np.sqrt((a * np.cos(theta))**2 + (b * np.sin(theta))**2)
        y_max = np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)

        x_min, x_max = cx - x_max, cx + x_max
        y_min, y_max = cy - y_max, cy + y_max

        return x_min, x_max, y_min, y_max            

    def show_viewer(self, do_show: bool):
        self.camera_viewer.parent.parent().setVisible(do_show)

    def _on_viewer_initialized(self, camera_viewer, target_viewer):
        self.set_camera_presets(camera_viewer, target_viewer)

    def quit_app(self):
        try:
            self.camera_viewer.quit_fun()
            self.camera_viewer.dockarea.parent().close()
        except Exception:
            pass
        self.mainwindow.close()


def gaussian_lineout_x(x_axis, x0, y0, dx, dy, theta, y_cross, A=1.0):
    cos_t = np.cos(theta * np.pi / 180)
    sin_t = np.sin(theta * np.pi / 180)
    X = x_axis - x0
    Y = y_cross - y0
    exponent = ((X * cos_t + Y * sin_t)**2) / (dx**2) + ((-X * sin_t + Y * cos_t)**2) / (dy**2)
    return A * np.exp(-exponent)

def gaussian_lineout_y(y_axis, x0, y0, dx, dy, theta, x_cross, A=1.0):
    cos_t = np.cos(theta * np.pi / 180)
    sin_t = np.sin(theta * np.pi / 180)
    X = x_cross - x0
    Y = y_axis - y0
    exponent = ((X * cos_t + Y * sin_t)**2) / (dx**2) + ((-X * sin_t + Y * cos_t)**2) / (dy**2)
    return A * np.exp(-exponent)


def main():
    from pymodaq_gui.utils.utils import mkQApp
    app = mkQApp('BeamTracker')

    mainwindow = QtWidgets.QMainWindow()
    dockarea = gutils.DockArea()
    mainwindow.setCentralWidget(dockarea)

    default_config_name = 'config_template'
    prog = BeamTracker1(dockarea, default_config_name)

    mainwindow.show()
    app.exec()


if __name__ == '__main__':
    main()
