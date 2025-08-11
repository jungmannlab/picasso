FAQ
===

Localize
--------

What doest MLE / LQ stand for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- MLE, integrated Gaussian (based on `Smith et al., 2010 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2862147/>`_.)
- LQ, Gaussian (least squares)


Picasso freezes during the localization step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This was a bug that could occur in the old picasso versions and should be fixed with version 0.2.4.

If it is always at the same localization:

- Try fitting with the LQ method; it tends to be more  robust
- Try increasing the gradient
- Make sure to set the correct camera parameters (Baseline etc.)
