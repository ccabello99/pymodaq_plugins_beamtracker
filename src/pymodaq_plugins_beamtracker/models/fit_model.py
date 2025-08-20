import numpy as np
from scipy.optimize import curve_fit

from pymodaq_plugins_beamtracker.extensions.utils.model import BeamTrackerModel, np  # np will be used in method eval of the formula

from pymodaq_utils.math_utils import gauss1D, my_moment

from pymodaq_data.data import DataToExport, DataWithAxes
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_beamtracker.extensions.utils.parser import (
    extract_data_names, split_formulae, replace_names_in_formula)

from pymodaq_utils.logger import set_logger, get_module_name
logger = set_logger(get_module_name(__name__))

def gaussian_fit(x, amp, x0, dx, offset):
    return amp * gauss1D(x, x0, dx) + offset


class BeamTrackerModelFit(BeamTrackerModel):
    params = [
        {'title': 'Get Data:', 'name': 'get_data', 'type': 'bool_push', 'value': False,
         'label': 'Get Data'},
        {'title': 'Edit Formula:', 'name': 'edit_formula', 'type': 'text', 'value': ''},
        {'title': 'Data0D:', 'name': 'data0D', 'type': 'itemselect',
         'value': dict(all_items=[], selected=[])},
        {'title': 'Data1D:', 'name': 'data1D', 'type': 'itemselect',
         'value': dict(all_items=[], selected=[])},
        {'title': 'Data2D:', 'name': 'data2D', 'type': 'itemselect',
         'value': dict(all_items=[], selected=[])},
        {'title': 'DataND:', 'name': 'dataND', 'type': 'itemselect',
         'value': dict(all_items=[], selected=[])},
    ]

    def ini_model(self):
        self.show_data_list()

    def update_settings(self, param: Parameter):
        if param.name() == 'get_data':
            self.show_data_list()

    def show_data_list(self):
        dte = self.modules_manager.get_det_data_list()

        data_list0D = dte.get_full_names('data0D')
        data_list1D = dte.get_full_names('data1D')
        data_list2D = dte.get_full_names('data2D')
        data_listND = dte.get_full_names('dataND')

        self.settings.child('data0D').setValue(dict(all_items=data_list0D, selected=[]))
        self.settings.child('data1D').setValue(dict(all_items=data_list1D, selected=[]))
        self.settings.child('data2D').setValue(dict(all_items=data_list2D, selected=[]))
        self.settings.child('dataND').setValue(dict(all_items=data_listND, selected=[]))

    def process_dte(self, dte: DataToExport):
        dte_processed = DataToExport('Computed')
        dwa = dte.get_data_from_full_name(dte.get_full_names()[0]).deepcopy()

        if not (dte.get_full_names()[0] in self.settings.child('data1D').value()['selected_items'] or 
                dte.get_full_names()[0] in self.settings.child('data2D').value()['selected_items']):
            logger.warning("Select the data from the data1D or data2D list !")
        elif dte.get_full_names()[0] in self.settings.child('data1D').value()['selected_items']:
            dwa.append(dwa.fit(gaussian_fit, self.get_guess(dwa)))
        elif dte.get_full_names()[0] in self.settings.child('data2D').value()['selected_items']:
            data = dwa.data
            x = dwa.get_axis_from_index(1)
            y = dwa.get_axis_from_index(1)
            x, y = np.meshgrid(x, y)
            # initial guess of parameters (can be whatever)
            initial_guess = (1200, 120, 80, 20, 20, 0, 50)
            # find the optimal Gaussian parameters
            popt, pcov = curve_fit(self.twoD_Gaussian, (x, y), data, p0=initial_guess)
            # create new data with these parameters
            data_fitted = DataWithAxes(name='fitted', data=self.twoD_Gaussian((x, y), *popt), axes=[y, x]) 
            dwa.append(data_fitted)

        dte_processed.append(dwa)

        return dte_processed

    @staticmethod
    def get_guess(dwa):
        offset = np.min(dwa).value()
        moments = my_moment(dwa.axes[0].get_data(), dwa.data[0])
        amp = (np.max(dwa) - np.min(dwa)).value()
        x0 = float(moments[0])
        dx = np.abs(float(moments[1]))

        return amp, x0, dx, offset

    @staticmethod
    def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()