''' General purpose library for handling input and output of files. '''
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
