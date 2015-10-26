"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, 2015
"""


import glob
import os.path
from numpy import memmap, fromfile, dtype, reshape
import yaml
import tifffile


class FileFormatNotSupported(Exception):
    pass


def load_raw(path, memory_map=True):
    info = load_raw_info(path)
    if memory_map:
        movie = memmap(path, info['data type'], 'r', shape=info['shape'])
    else:
        movie = fromfile(path, info['data type'])
        movie = reshape(movie, info['shape'])
    return movie, info


def load_raw_info(path):
    path_base, path_extension = os.path.splitext(path)
    with open(path_base + '.yaml', 'r') as info_file:
        info = yaml.load(info_file)
    info['shape'] = tuple(info['shape'])
    return info


def load_tif(path):
    info = {}
    with tifffile.TiffFile(path) as tif:
        movie = tif.asarray()
        info['file'] = tif.filename
        if 'datetime' in tif.pages[0].tags:
            info['timestamp'] = tif.pages[0].tags.datetime.value.decode()
        info['data type'] = str(dtype(tif.pages[0].dtype))
        info['byte order'] = tif.byteorder
        info['bits per sample'] = tif.pages[0].tags.bits_per_sample.value
    if movie.ndim == 3:
        info['frames'], info['width'], info['height'] = movie.shape
    elif movie.ndim == 2:
        info['frames'] = 1
        info['width'], info['height'] = movie.shape
    elif movie.ndim == 1:
        info['frames'] = info['height'] = 1
        info['width'] = len(movie)
    info['shape'] = [info['frames'], info['width'], info['height']]
    return movie, info


def to_raw_single(path):
    path_base, path_extension = os.path.splitext(path)
    path_extension = path_extension.lower()
    if path_extension == '.tif' or path_extension == 'tiff':
        movie, info = load_tif(path)
    else:
        raise FileFormatNotSupported("File format must be '.tif' or '.tiff'.")
    raw_file_name = path_base + '.raw'
    movie.tofile(raw_file_name)
    info['original file'] = info.pop('file')
    info['raw file'] = os.path.basename(raw_file_name)
    with open(path_base + '.yaml', 'w') as info_file:
        yaml.dump(info, info_file, default_flow_style=False)


def to_raw(files):
    paths = glob.glob(files)
    for path in paths:
        to_raw_single(path)
