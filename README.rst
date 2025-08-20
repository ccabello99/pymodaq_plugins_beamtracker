pymodaq_plugins_beamtracker
#########################

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_beamtracker.svg
   :target: https://pypi.org/project/pymodaq_plugins_beamtracker/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/ccabello99/pymodaq_plugins_beamtracker/workflows/Upload%20Python%20Package/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_plugins_beamtracker

.. image:: https://github.com/PyMoDAQ/pymodaq_plugins_beamtracker/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_plugins_beamtracker/actions/workflows/Test.yml


Plugin exposing a new app which can be run with `beam_tracker` in the terminal: the BeamTracker performs 2D Gaussian fitting
and tracking to displayed data. Drawing the fit and displaying its lineouts is an option in the parameter tree. Once a fit is
displayed in the viewer, one can choose to save it via the parameter tree, which saves it to the ROIManager. It can then be
loaded from the parameter tree as well after specifying the .xml file. The loading of saved roi files is an option in the 
parameter tree because performing the loading here will prevent the display of the lineout viewers and the ROIManager.

There is a BeamTracker class for 1 camera usage and a BeamTracker class for 2 camera usage.


Authors
=======

* Christian Cabello  (christian.cabello@ip-paris.fr)



Instruments
===========

Below is the list of instruments included in this plugin


Viewer0D
++++++++

Application
==========

* BeamTracker: application which can be run via terminal as `beam_tracker`: The BeamTracker performs 2D Gaussian fitting
and tracking to displayed data, allowing the option to draw the fit in the viewer and display its lineouts.


Installation instructions
=========================

* pymodaq >= 5.0.1
* pymodaq_data >= 0.0.1