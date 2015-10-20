#!/usr/bin/env python
'''
Identify and localize fluorescent single molecules in an image sequence.
'''

import argparse


def main(files, parameters):




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='one or multiple raw files specified by a unix style path pattern')
    parser.add_argument('parameters', help='a yaml parameter file')
    args = parser.parse_args()
    main(args.files, args.parameters)
