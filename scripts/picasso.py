#!/usr/bin/env python
"""
    scripts/picasso
    ~~~~~~~~~~~~~~~

    Picasso command line interface

    :author: Joerg Schnitzbauer, 2015
"""
import glob
import os.path
import yaml
import sys
sys.path.pop(0)
sys.path.insert(0, '..')
from picasso import io, localize


def _localize(files, parameters_file, verbose=True):
    paths = glob.glob(files)
    n_files = len(paths)
    if n_files:
        with open(parameters_file, 'r') as parameters_file:
            parameters = yaml.load(parameters_file)
        for i, path in enumerate(paths):
            if verbose:
                print('Localizing in file {}/{}...'.format(i + 1, n_files), end='\r')
            movie, info = io.load_raw(path)
            locs = localize.localize(movie, info, parameters)
            base, ext = os.path.splitext(path)
            io.save_locs(base + '_locs.txt', locs, info, parameters)
    else:
        if verbose:
            print('No files matching {}'.format(files))
    return n_files


if __name__ == '__main__':
    import argparse

    # Main parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # toraw parser
    toraw_parser = subparsers.add_parser('toraw', help='convert image sequences to binary raw files and accompanied YAML information files')
    toraw_parser.add_argument('files', help='one or multiple files specified by a unix style pathname pattern')

    # localize parser
    localize_parser = subparsers.add_parser('localize', help='identify and localize fluorescent single molecules in an image sequence')
    localize_parser.add_argument('files', help='one or multiple raw files specified by a unix style path pattern')
    localize_parser.add_argument('parameters', help='a yaml parameter file')

    # Parse
    args = parser.parse_args()
    if args.command:
        if args.command == 'toraw':
            io.to_raw(args.files, verbose=True)
        elif args.command == 'localize':
            _localize(args.files, args.parameters, verbose=True)
    else:
        parser.print_help()
