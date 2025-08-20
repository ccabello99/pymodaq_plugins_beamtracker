pymodaq_plugins_beamtracker
#########################

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_beamtracker.svg
   :target: https://pypi.org/project/pymodaq_plugins_beamtracker/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/ccabello99/pymodaq_plugins_beamtracker/workflows/Upload%20Python%20Package/badge.svg
    :target: https://github.com/ccabello99/pymodaq_plugins_beamtracker

.. image:: https://github.com/ccabello99/pymodaq_plugins_beamtracker/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/ccabello99/pymodaq_plugins_beamtracker/actions/workflows/Test.yml


This plugin provides a new application that can be launched from the terminal using::

   beam_tracker

The **BeamTracker** performs 2D Gaussian fitting and tracking on displayed camera data.  
Features include:

- Optional drawing of the fit ellipse and display of its horizontal/vertical lineouts (via the parameter tree).
- Saving the current fit to the **ROI Manager** directly from the parameter tree.
- Loading saved ROIs from an ``.xml`` file (also via the parameter tree).  
  Note: loading ROIs through the parameter tree bypasses the lineout viewers and ROI Manager to avoid conflicts.

Two variants of the BeamTracker are available:

- **Single-camera mode**
- **Dual-camera mode**

You can start the appropriate version either by running the class directly or by calling::

   beam_tracker --mode 1   # single camera (default)
   beam_tracker --mode 2   # dual camera

If you are running dual-camera mode, it is suggested to keep exposure time above 100 ms for both cameras, or adjust the 
**Wait time (ms)** parameter in the **Main Settings** parameter tree of the cameras. This will keep app performance smooth.


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

* BeamTracker: Performs 2D Gaussian fitting and tracking on displayed camera data. 
   provides a new application that can be launched from the terminal using::

   beam_tracker


Installation instructions
=========================

* pymodaq >= 5.0.1
* pymodaq_data >= 0.0.1