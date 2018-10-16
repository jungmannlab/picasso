localize
========
Camera Config
-------------

Picasso can remember default cameras and will use saved camera
parameters. In order to use camera configs, create a file named
``config.yaml`` in the picasso folder. To start with a template, modify
``config_template.yaml`` that can be found in the folder per default.
Picasso will compare the entries with Micro-Manager-Metadata and match
the sensitivity values. If no matching entries can be found (e.g., if
the file was not created with Micro-Manager) the config file will still
be usable to create a dropdown menu to select the different categories.
The camera config can also be used to define a default camera that will
always be used. Indentions are used for definitions.

Example 1: Default Camera
~~~~~~~~~~~~~~~~~~~~~~~~~

::

   Cameras:
     Camera1:
       Baseline: 100
       Sensitivity: 0.5
       Quantum Efficiency: 1.0

If there is only one camera entry, picasso will create a dropdown menu
that has always selected this camera. ### Example 2: Default Camera with
different settings Consider a camera that is used for different
wavelengths and has different quantum efficiencies:

::

   Cameras:
     Camera1:
       Baseline: 100
       Sensitivity: 1
       Quantum Efficiency:
         525: 0.5
         595: 0.6
         700: 0.7

This will create a dropdown menu labeled “Emission Wavelength” were the
user can select the corresponding entry that will change the sensitivity
parameter.

Picasso will search for the following entries in the config.yaml: ####
Gain If the string ``Gain Property Name`` can be found in the config,
picasso will search for a value for this key in the Micro-Manager
metadata and match if found.

Sensitivity
^^^^^^^^^^^

If the string ``Sensitivity Categories`` can be found in the config,
picasso will create a dropdown menu for each entry and if the property
can be found in the Micro-Manager Metadata, it will be automatically
set.

::

   Cameras:
     Camera1:
       Baseline: 100
       Quantum Efficiency:
         525: 0.5
       Sensitivity Categories:
         - PixelReadoutRate
         - Sensitivity/DynamicRange
       Sensitivity:
         540 MHz - fastest readout:
           12-bit (high well capacity): 7.18
           12-bit (low noise): 0.29
           16-bit (low noise & high well capacity): 0.46
         200 MHz - lowest noise:
           12-bit (high well capacity): 7.0
           12-bit (low noise): 0.26
           16-bit (low noise & high well capacity): 0.45

Here, two Sensitivity Categories are given ``PixelReadoutRate`` and
``Sensitivity/DynamicRange``. In the upper dropdown menu, one now will
be able to choose from ``540 MHz - fastest readout`` and
``200 MHz - lowest noise``. Within 540 MHz it will be
``12-bit (high well capacity): 7.18``, ``12-bit (low noise): 0.29`` and
``16-bit (low noise & high well capacity): 0.46``. Accordingly for the
200 MHz entry. The dropdown menus can be further nested, e.g. when
considering Gain modes:

::

       Sensitivity:
         Electron Multiplying:
           17.000 MHz:
             Gain 1: 15.9
             Gain 2: 9.34
             Gain 3: 5.32

Quantum Efficiency
^^^^^^^^^^^^^^^^^^

If the string ``Quantum Efficiency`` can be found in the config, picasso
will search for a value for the key named ``Channel Device`` in the
Micro-Manager metadata and match if found.

::

   Cameras:
     Camera_1:
       Baseline: 100
       Quantum Efficiency:
         525: 0.5
         595: 0.6
         700: 0.7
       Channel Device:
         Name: TIFilterBlock1-Label
         Emission Wavelengths:
           1-R640: 700
           2-G561: 595
           3-B489: 525
       Sensitivity: 0.47

Picasso will search for the entry ``TIFilterBlock1-Label`` in the
Micro-Manager Metadata. If this would be ``1-G561``, the
Emission-Wavelength of ``595`` will be used to determine the Quantum
Efficiency (here 0.6).

Several Cameras
^^^^^^^^^^^^^^^

::

   Cameras:
     Camera1:
     Camera2:
     Camera3:

Once there are several cameras present, Picasso will select the camera
who’s name matches the Micro-Manager Metadata. If no camera is found,
the first one is automatically selected.

3D-Calibration
--------------

Theory
~~~~~~

3D Calibration is performed by an adapted version of the 2008 Science
paper by Huang
``Three-dimensional Super-resolution Imaging by Stochastic Optical Reconstruction Microscopy``.

Calibrating z
~~~~~~~~~~~~~

After entering the step size, picasso will calculate the mean and the
variance for sigma_x and sigma_y for each z position. Localizations that
are not within one standard deviation are discarded. A six-degree
polynomial is fitted to the mean values of x and y.

-  mean_sx = cx[6]z0 + cx[5]z1 .. + cx[0]z6
-  mean_sy = cy[6]z0 + cy[5]z1 .. + cy[0]z6

The calibration coefficients are stored in the yaml file and contain the
parameters of cx and cy. The first entry being c[0], the last being
c[6].

Fitting z
~~~~~~~~~

For each localization, sigma_x and sigma_y is determined. Similar to the
Science paper, the following equation is used to minimize the Distance
D: \* D = (sx0.5 - wx0.5)2 + (sy0.5 - wy0.5)2 with w being c[6]z0 +
c[5]z1 .. + c[0]z6