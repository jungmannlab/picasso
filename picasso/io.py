"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, 2015
"""


import os.path as _ospath
import numpy as _np
import yaml as _yaml
import glob as _glob
import h5py as _h5py
import re as _re
import struct as _struct
import json as _json


def to_little_endian(movie, info):
    if info['Byte Order'] == '>':
        movie = movie.byteswap()
        info['Byte Order'] = '<'
    return movie, info


def load_raw(path, memory_map=True):
    info = load_info(path)
    info = info[0]
    dtype = _np.dtype(info['Data Type'])
    shape = (info['Frames'], info['Height'], info['Width'])
    if memory_map:
        movie = _np.memmap(path, dtype, 'r', shape=shape)
    else:
        movie = _np.fromfile(path, dtype)
        movie = _np.reshape(movie, shape)
    movie, info = to_little_endian(movie, info)
    return movie, info


def load_info(path):
    path_base, path_extension = _ospath.splitext(path)
    with open(path_base + '.yaml', 'r') as info_file:
        info = list(_yaml.load_all(info_file))
    return info


def save_info(path, info):
    with open(path, 'w') as file:
        _yaml.dump_all(info, file, default_flow_style=False)


class TiffFile:

    TIFF_TYPES = {1: 'B', 2: 'c', 3: 'H', 4: 'L', 5: 'RATIONAL'}
    TYPE_SIZES = {'c': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'L': 4, 'RATIONAL': 8}

    def __init__(self, path):
        self._file_handle = open(path, 'rb')
        self.byte_order = {b'II': '<', b'MM': '>'}[self._file_handle.read(2)]
        self.path = path
        self._file_handle.seek(4)
        self._first_ifd_offset = self._read('L')
        self._create_info()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file_handle.close()

    def __iter__(self):
        offset = self._first_ifd_offset
        while offset != 0:
            self._file_handle.seek(offset)
            n_entries = self._read('H')
            for i in range(n_entries):
                self._file_handle.seek(offset + 2 + i * 12)
                tag = self._read('H')
                if tag == 273:
                    type = self.TIFF_TYPES[self._read('H')]
                    count = self._read('L')
                    image_offset = self._read(type, count)
                    break
            else:
                raise Exception('IFD does not have tag 273 (data offset)')
            self._file_handle.seek(offset + 2 + n_entries * 12)
            offset = self._read('L')
            self._file_handle.seek(image_offset)
            buffer = self._file_handle.read(self.image_byte_count)
            flat_image = _np.frombuffer(buffer, dtype=self.dtype)
            yield _np.reshape(flat_image, (self.image_height, self.image_width))

    def _read(self, type, count=1):
        if type == 'c':
            return self._file_handle.read(count)
        elif type == 'RATIONAL':
            return self._read_numbers('L') / self._read_numbers('L')
        else:
            return self._read_numbers(type, count)

    def _read_numbers(self, type, count=1):
        size = self.TYPE_SIZES[type]
        fmt = self.byte_order + count * type
        return _struct.unpack(fmt, self._file_handle.read(count * size))[0]

    def _create_info(self):
        self._file_handle.seek(self._first_ifd_offset)
        n_entries = self._read('H')
        self.info = {'Byte Order': self.byte_order, 'File': self.path}
        for i in range(n_entries):
            self._file_handle.seek(self._first_ifd_offset + 2 + i * 12)
            tag = self._read('H')
            type = self.TIFF_TYPES[self._read('H')]
            count = self._read('L')
            if count * self.TYPE_SIZES[type] > 4:
                self._file_handle.seek(self._read('L'))
            if tag == 256:
                self.image_width = self.info['Width'] = self._read(type, count)
            elif tag == 257:
                self.image_height = self.info['Height'] = self._read(type, count)
            elif tag == 258:
                bits_per_sample = self._read(type, count)
                self.dtype = _np.dtype(self.byte_order + 'u' + str(int(bits_per_sample/8)))
                self.info['Data Type'] = self.dtype.name
            elif tag == 279:
                self.image_byte_count = self._read(type, count)
            elif tag == 51123:
                readout = self._read(type, count)
                mm_info = _json.loads(readout.decode())
                self.info['Camera'] = {'Manufacturer': mm_info['Camera']}
                if self.info['Camera']['Manufacturer'] == 'Andor':
                    _, type, model, serial_number, _ = (_.strip() for _ in mm_info['Andor-Camera'].split('|'))
                    self.info['Camera']['Type'] = type
                    self.info['Camera']['Model'] = model
                    self.info['Camera']['Serial Number'] = int(serial_number)
                    em = (mm_info['Andor-EMSwitch'] == 'On') or (mm_info['Andor-Output_Amplifier'] == 'Electron Multiplying')
                    self.info['Electron Multiplying'] = em
                    self.info['EM Real Gain'] = int(mm_info['Andor-Gain'])
                    self.info['Pre-Amp Gain'] = int(mm_info['Andor-Pre-Amp-Gain'].split()[1])
                    self.info['Readout Mode'] = mm_info['Andor-ReadoutMode']
                self.info['Excitation Wavelength'] = int(mm_info['TIFilterBlock1-Label'][-3:])
        offset = self._first_ifd_offset
        n_frames = 0
        while offset != 0:
            n_frames += 1
            self._file_handle.seek(offset)
            n_entries = self._read('H')
            self._file_handle.seek(offset + 2 + n_entries * 12)
            offset = self._read('L')
        self.info['Frames'] = n_frames

    def close(self):
        self._file_handle.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = (byte_order != self.byte_order)
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


def to_raw_combined(paths):
    path_base, path_extension = _ospath.splitext(paths[0])
    path_extension = path_extension.lower()
    raw_file_name = path_base + '.raw'
    with open(raw_file_name, 'wb') as file_handle:
        with TiffFile(paths[0]) as tif:
            tif.tofile(file_handle, '<')
            info = tif.info
        for path in paths[1:]:
            with TiffFile(path) as tif:
                info['Frames'] += tif.info['Frames']
                tif.tofile(file_handle, '<')
        info['Generated by'] = 'Picasso ToRaw'
        info['Byte Order'] = '<'
        info['Original File'] = _ospath.basename(info.pop('File'))
        info['Raw File'] = _ospath.basename(raw_file_name)
        save_info(path_base + '.yaml', [info])


def get_movie_groups(paths):
    groups = []
    paths = sorted(paths)
    while len(paths) > 0:
        path = paths[0]
        if path.endswith('.ome.tif'):
            path_base = path[0:-8]
            pattern = r'{}'.format(path_base + '_([0-9]).ome.tif')
            matches = [_re.match(pattern, _) for _ in paths]
            group = [path] + [_.group() for _ in matches if _]
            groups.append(group)
            for path_done in group:
                paths.remove(path_done)
    return groups


def to_raw(path, verbose=True):
    paths = _glob.glob(path)
    paths = [path.replace('\\', '/') for path in paths]
    groups = get_movie_groups(paths)
    n_groups = len(groups)
    if n_groups:
        for i, group in enumerate(groups):
            if verbose:
                print('Converting movie {}/{}...'.format(i + 1, n_groups), end='\r')
            to_raw_combined(group)
        if verbose:
            print()
    else:
        if verbose:
            print('No files matching {}'.format(path))


def save_locs(path, locs, info):
    with _h5py.File(path, 'w') as locs_file:
        locs_file.create_dataset('locs', data=locs)
    base, ext = _ospath.splitext(path)
    info_path = base + '.yaml'
    save_info(info_path, info)


def load_locs(path):
    with _h5py.File(path, 'r') as locs_file:
        locs = locs_file['locs'][...]
    locs = _np.rec.array(locs, dtype=locs.dtype)    # Convert to rec array with fields as attributes
    info = load_info(path)
    return locs, info
