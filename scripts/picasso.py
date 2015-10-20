#!/usr/bin/env python
"""
    scripts/picasso
    ~~~~~~~~~~~~~~~

    Picasso command line interface

    :author: Joerg Schnitzbauer
"""


def import_nolocal(module):
    """
    Imports a module, but ignores the current file.
    This is needed, when we want to import `picasso` package.
    """
    import sys
    import importlib
    temp = sys.path.pop(0)
    module = importlib.import_module(module)
    sys.path.insert(0, temp)
    return module


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

    # GUI parser
    gui_parser = subparsers.add_parser('gui', help='load graphical user interface for the following command')
    gui_parser.add_argument('tool', choices=['toraw', 'localize'])

    # Parse
    args = parser.parse_args()
    if args.command:
        if args.command == 'gui':
            module = import_nolocal('picasso.gui.' + args.tool)
            module.main()
        elif args.command == 'toraw':
            io = import_nolocal('picasso.io')
            io.to_raw(args.files)
        elif args.command == 'localize':
            localize = import_nolocal('picasso.localize')
            localize.localize(args.files, args.parameters)
    else:
        parser.print_help()
