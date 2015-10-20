''' General purpose library for handling input and output of files. '''


import glob
import os.path
import yaml
import tifffile


def load_tif(path):
    info = {}
    with tifffile.TiffFile(path) as tif:
        movie = tif.asarray()
        info['file'] = tif.filename
        info['timestamp'] = tif.pages[0].tags['datetime'].value.decode()
    if movie.ndim == 2:
        info['frames'] = 1
        info['x size'], info['y size'] = movie.shape
    elif movie.ndim == 3:
        info['frames'], info['x size'], info['y size'] = movie.shape
    else:
        info['shape'] = movie.shape
    return movie, info


def to_raw(files):
    paths = glob.glob(files)
    for path in paths:
        path_base, path_extension = os.path.splitext(path)
        path_extension = path_extension.lower()
        if path_extension == '.tif' or path_extension == 'tiff':
            movie, info = load_tif(path)
        else:
            print('Image format {} not supported.'.format(path_extension))
            return
        raw_file_name = path_base + '.raw'
        movie.tofile(raw_file_name)
        info['original file'] = info.pop('file')
        info['raw file'] = raw_file_name
        with open(path_base + '.yaml', 'w') as info_file:
            yaml.dump(info, info_file, default_flow_style=False)
