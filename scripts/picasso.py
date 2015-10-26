#!/usr/bin/env python
"""
    scripts/picasso
    ~~~~~~~~~~~~~~~

    Picasso command line interface

    :author: Joerg Schnitzbauer, 2015
"""


def import_nolocal(module):
    """
    Imports a module, but ignores the current directory.
    This is needed, when we want to import the `picasso` package.
    """
    import sys
    import importlib
    temp = sys.path.pop(0)
    module = importlib.import_module(module)
    sys.path.insert(0, temp)
    return module


# localize = import_nolocal('picasso.localize')   # Added this, because multiprocessing thinks that this file is the picasso package


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
            io = import_nolocal('picasso.io')
            io.to_raw(args.files, verbose=True)
        elif args.command == 'localize':
            localize = import_nolocal('picasso.localize')
            localize.localize(args.files, args.parameters)
    else:
        parser.print_help()
