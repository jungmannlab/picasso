FAQ
===

Localize
--------

What doest MLE / LQ stand for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- MLE, integrated Gaussian (based on `Smith et al., 2014 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2862147/>`_.)
- LQ, Gaussian (least squares)


Picasso freezes during the localization step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If it is always at the same localization:
- Try fitting with the LQ method; it tends to be more  robust
- Try increasing the gradient
- Make sure to set the correct camera parameters (Baseline etc.)