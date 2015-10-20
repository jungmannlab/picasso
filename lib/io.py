''' General purpose library for handling input and output of files. '''
import tifffile


def load_tif(path):
    info = {}
    with tifffile.TiffFile(path) as tif:
        movie = tif.asarray()
        info['file'] = tif.filename
        info['timestamp'] = tif.pages[0].tags['datetime'].value.decode()
    info['frames'], info['x size'], info['y size'] = movie.shape
    return movie, info
