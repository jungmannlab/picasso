#!/usr/bin/env python
'''
A wrapper script which can be used to call the subscripts. It might be useful for command line sanity or getting reminded which scripts
are available in picasso.
'''


import argparse

if __name__ == '__main__':
    # Main parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', help='load graphical user interface for the given command', action='store_true')
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
        if args.gui:
            print('Gui was requested.')
        else:
            # Run command line interface
            if args.command == 'toraw':
                from picasso import io
                io.to_raw.main(args.files)
            elif args.command == 'localize':
                from picasso import localize
                localize.localize(args.files, args.parameters)
    else:
        parser.print_help()
