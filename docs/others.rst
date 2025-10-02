=====
Other
=====

Sound notifications
-------------------
Starting in version 0.8.5, Picasso supports sound notifications. In Render and SPINNA, these can be selected in the ``File`` menu in the menu bar. The available files are read from the ``picasso/gui/notification_sounds`` folder. ``.mp3`` and ``.wav`` files are supported. Default sound notification is saved automatically when manually changed.

Custom notifications
~~~~~~~~~~~~~~~~~~~~
To add custom notification sounds, copy the sound files (``.mp3`` or ``.wav``) to the  ``picasso/gui/notification_sounds`` folder. Depending on how you installed Picasso, this folder can be found in different locations:

GitHub
------
If you cloned the GitHub repository, you can add sound notifications by following these steps:

- Find the directory where you cloned the GitHub repository with Picasso.
- Go to ``picasso/gui/notification_sounds``.
- Copy the sound files to this folder.

PyPI
----
If you installed Picasso using ``pip install picassosr``, you can add sound notifications by following these steps:

- Activate your conda environment where ``picassosr`` is installed by typing ``conda activate YOUR_ENVIRONMENT``.
- To find the location of the package, type ``pip show picassosr`` and look for the line starting with ``Location:``.
- Navigate to this location and go to ``picasso/gui/notification_sounds``.
- Copy the sound files to this folder.


One click installer
-------------------
If you installed Picasso using the one click installer from `the Picasso release page <https://github.com/jungmannlab/picasso/releases/>`__ , you can add sound notifications by following these steps:

- Find the location where you installed Picasso. By default, it is ``C:/Picasso``. *Before version 0.8.3, the default location was* ``C:/Program Files/Picasso``.
- Go to the following subfolder: ``picasso/gui/notification_sounds``.
- Copy the sound files to this folder.