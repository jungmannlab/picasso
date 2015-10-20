#!/usr/bin/env python
'''
Convert image sequences to binary raw files and accompanied YAML information files.
'''

import argparse
import glob
import os.path
import yaml
from lib import io


def main(files):
    paths = glob.glob(files)
    for path in paths:
        path_base, path_extension = os.path.splitext(path)
        path_extension = path_extension.lower()
        if path_extension == '.tif' or path_extension == 'tiff':
            movie, info = io.load_tif(path)
        else:
            print('Image format {} not supported.'.format(path_extension))
            return
        raw_file_name = path_base + '.raw'
        movie.tofile(raw_file_name)
        info['original file'] = info.pop('file')
        info['raw file'] = raw_file_name
        with open(path_base + '.yaml', 'w') as info_file:
            yaml.dump(info, info_file, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='one or multiple files specified by a unix style pathname pattern')
    args = parser.parse_args()
    main(args.files)
