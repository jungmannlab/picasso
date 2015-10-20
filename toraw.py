#!/usr/bin/env python
'''
Convert image sequences to binary raw and accompanied YAML information files.
'''

import argparse
import glob


def main(paths):
    print('Converting files:')
    print(paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='one or multiple files specified by a unix style pathname pattern')
    args = parser.parse_args()
    paths = glob.glob(args.files)
    main(paths)
