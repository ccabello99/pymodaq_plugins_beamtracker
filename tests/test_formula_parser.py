import pytest
import re
import numpy as np

from pymodaq_plugins_beamtracker.extensions.utils.parser import (
    split_formulae, extract_data_names, replace_names_in_formula)
from pymodaq_data.data import DataToExport, DataRaw

data_name_1 = 'Integrated_ROI_01'
origin1 = 'Spectrum - ROI_01'
data_name_2 = 'Integrated_ROI_00'
origin2 = 'Spectrum - ROI_00'
data_name_3 = 'Harmonics'
origin3 = 'Spectrum'

dte = DataToExport('dte', data=[
    DataRaw(data_name_1, data=[np.array([-1.,])], origin=origin1),
    DataRaw(data_name_2, data=[np.array([4.,])], origin=origin2),
    DataRaw(data_name_3, data=[np.arange(10)], origin=origin3),
])

NFORMULA = 3

FORMULA = (f'{{{origin3}/{data_name_3}}}/'
           f'np.abs({{{origin1}/{data_name_1}}}/{{{origin2}/{data_name_2}}})'
           )

FORMULAE = ''
for ind in range(NFORMULA):
    FORMULAE += f'{FORMULA}'
    if ind != NFORMULA-1:
        FORMULAE += '\n'


def test_split_formulae():
    lines = split_formulae(FORMULAE)
    assert len(lines) == NFORMULA
    assert lines[0] == FORMULA


def test_extract_data_names():
    data_name_list = extract_data_names(FORMULA)
    assert data_name_list == [f'{origin3}/{data_name_3}',
                              f'{origin1}/{data_name_1}',
                              f'{origin2}/{data_name_2}']


def test_replace_name_in_formula():
    formula_to_eval, names = replace_names_in_formula(formula=FORMULA)

    assert eval(formula_to_eval) == dte[2] / np.abs(dte[0] / dte[1])

