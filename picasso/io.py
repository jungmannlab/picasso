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
from pprint import pprint as _pprint


def to_little_endian(movie, info):
    if info[0]['Byte Order'] == '>':
        movie = movie.byteswap()
        info[0]['Byte Order'] = '<'
    return movie, info


def load_raw(path, memory_map=True):
    info = load_info(path)
    dtype = _np.dtype(info[0]['Data Type'])
    shape = (info[0]['Frames'], info[0]['Height'], info[0]['Width'])
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

    def __init__(self, path, verbose=False):
        self._verbose = verbose
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
        current_frame = 0
        while offset != 0:
            current_frame += 1
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
                raise Exception('IFD does not have tag 273 (data offset) (File: {}, Frame: {})'.format(self.path,
                                                                                                       current_frame))
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
        self.info = {'Byte Order': self.byte_order, 'File': self.path}
        self._file_handle.seek(28)
        comments_offset = self._read('L')
        self._file_handle.seek(36)
        summary_length = self._read('L')
        self.info['Summary'] = _json.loads(self._read('c', summary_length).decode())
        self._file_handle.seek(self._first_ifd_offset)
        n_entries = self._read('H')
        for i in range(n_entries):
            self._file_handle.seek(self._first_ifd_offset + 2 + i * 12)
            tag = self._read('H')
            type = self.TIFF_TYPES[self._read('H')]
            count = self._read('L')
            if self._verbose:
                print('Tag {} (type: {}, count: {})'.format(tag, type, count))
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
                readout = self._read(type, count).strip(b'\0')      # Strip null bytes which MM 1.4.22 adds
                mm_info = _json.loads(readout.decode())
                if self._verbose:
                    _pprint(mm_info)
                # Read out info specifically for Picasso
                camera = mm_info['Camera']
                self.info['Camera'] = camera
                if camera + '-Output_Amplifier' in mm_info:
                    em = (mm_info[camera + '-Output_Amplifier'] == 'Electron Multiplying')
                    self.info['Electron Multiplying'] = em
                if camera + '-Gain' in mm_info:
                    self.info['EM Real Gain'] = int(mm_info[camera + '-Gain'])
                if camera + '-Pre-Amp-Gain' in mm_info:
                    try:
                        self.info['Pre-Amp Gain'] = int(mm_info[camera + '-Pre-Amp-Gain'].split()[1])
                    except IndexError:      # In case gain is specified in the format "5x"
                        self.info['Pre-Amp Gain'] = str(mm_info[camera + '-Pre-Amp-Gain']) + ' (CONVERT TO INDEX!)'
                    self.info['Readout Mode'] = mm_info[camera + '-ReadoutMode']
                if camera + '-PixelReadoutRate' in mm_info:
                    self.info['Readout Rate'] = mm_info[camera + '-PixelReadoutRate'].split('-')[0].strip()
                if camera + '-Sensitivity/DynamicRange' in mm_info:
                    self.info['Gain Setting'] = mm_info[camera + '-Sensitivity/DynamicRange']
                if 'TIFilterBlock1-Label' in mm_info:
                    try:
                        self.info['Excitation Wavelength'] = int(mm_info['TIFilterBlock1-Label'][-3:])
                    except ValueError:      # Last three digits are not a number
                        self.info['Excitation Wavelength'] = None
                # Dump the rest
                self.info['Micro-Manager Metadata'] = mm_info
        offset = self._first_ifd_offset
        n_frames = 0
        while offset != 0:
            n_frames += 1
            self._file_handle.seek(offset)
            n_entries = self._read('H')
            self._file_handle.seek(offset + 2 + n_entries * 12)
            offset = self._read('L')
        self.info['Frames'] = n_frames
        if comments_offset:
            self._file_handle.seek(comments_offset + 4)
            comments_length = self._read('L')
            if comments_length:
                comments_bytes = self._read('c', comments_length)
                try:
                    comments_json = comments_bytes.decode()
                except UnicodeDecodeError:
                    print('Did not find UTF-8 decoded comment bytes!')
                else:
                    self.info['Comments'] = _json.loads(comments_json)

    def close(self):
        self._file_handle.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = (byte_order != self.byte_order)
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


def to_raw_combined(basename, paths):
    raw_file_name = basename + '.ome.raw'
    with open(raw_file_name, 'wb') as file_handle:
        with TiffFile(paths[0]) as tif:
            tif.tofile(file_handle, '<')
            info = tif.info
        for path in paths[1:]:
            with TiffFile(path) as tif:
                info['Frames'] += tif.info['Frames']
                if 'Comments' in tif.info:
                    info['Comments'] = tif.info['Comments']
                tif.tofile(file_handle, '<')
        info['Generated by'] = 'Picasso ToRaw'
        info['Byte Order'] = '<'
        info['Original File'] = _ospath.basename(info.pop('File'))
        info['Raw File'] = _ospath.basename(raw_file_name)
        save_info(basename + '.ome.yaml', [info])


def get_movie_groups(paths):
    groups = {}
    if len(paths) > 0:
        pattern = _re.compile('(.*?)(_(\d*))?.ome.tif')    # This matches the basename + an optional appendix of the file number
        matches = [_re.match(pattern, path) for path in paths]
        match_infos = [{'path': _.group(), 'base': _.group(1), 'index': _.group(3)} for _ in matches]
        for match_info in match_infos:
            if match_info['index'] is None:
                match_info['index'] = 0
            else:
                match_info['index'] = int(match_info['index'])
        basenames = set([_['base'] for _ in match_infos])
        for basename in basenames:
            match_infos_group = [_ for _ in match_infos if _['base'] == basename]
            group = [_['path'] for _ in match_infos_group]
            indices = [_['index'] for _ in match_infos_group]
            group = [path for (index, path) in sorted(zip(indices, group))]
            groups[basename] = group
    return groups


def to_raw(path, verbose=True):
    paths = _glob.glob(path)
    groups = get_movie_groups(paths)
    n_groups = len(groups)
    if n_groups:
        for i, (basename, group) in enumerate(groups.items()):
            if verbose:
                print('Converting movie {}/{}...'.format(i + 1, n_groups), end='\r')
            to_raw_combined(basename, group)
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
