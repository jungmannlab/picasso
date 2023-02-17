"""
Some rudimentary tests.
"""

from picasso import __main__ as main


def test_localize():
    """
    Test localization with mle and drift correction
    """
    import argparse

    import os

    cwd = os.getcwd()
    print(cwd)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.files = "./tests/data/testdata.raw"
    args.fit_method = "mle"
    args.box_side_length = 7
    args.gradient = 5000
    args.baseline = 0
    args.sensitivity = 1
    args.gain = 1
    args.qe = 1
    args.roi = None
    args.drift = 100

    for fit_method in ["mle"]:
        args.fit_method = fit_method
        main._localize(args)
