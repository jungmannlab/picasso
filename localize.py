#!/usr/bin/env python
'''
Identify and localize fluorescent single molecules in an image sequence.
'''

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', help='load graphical user interface', action='store_true')
    args = parser.parse_args()
    if args.gui:
        import gui
        gui.main()
    else:
        print('Command line interface not implemented yet.')    # TODO
