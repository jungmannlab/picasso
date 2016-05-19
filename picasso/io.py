"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry
"""


import os.path as _ospath
import numpy as _np
import yaml as _yaml
import glob as _glob
import h5py as _h5py
import re as _re
import struct as _struct
import json as _json
import os as _os
import threading as _threading
from . import lib as _lib


def _user_settings_filename():
    home = _ospath.expanduser('~')
    return _ospath.join(home, '.picasso', 'settings.yaml')


def to_little_endian(movie, info):
    if info[0]['Byte Order'] != '<':
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


def load_tif(path):
    movie = TiffMultiMap(path, memmap_frames=False)
    info = movie.info()
    return movie, [info]


def load_movie(path):
    base, ext = _ospath.splitext(path)
    ext = ext.lower()
    if ext == '.raw':
        return load_raw(path)
    elif ext == '.tif':
        return load_tif(path)


def load_info(path):
    path_base, path_extension = _ospath.splitext(path)
    with open(path_base + '.yaml', 'r') as info_file:
        info = list(_yaml.load_all(info_file))
    return info


def load_user_settings():
    settings_filename = _user_settings_filename()
    try:
        settings_file = open(settings_filename, 'r')
    except FileNotFoundError:
        return _lib.AutoDict()
    settings = _yaml.load(settings_file)
    settings_file.close()
    return _lib.AutoDict(settings)


def save_info(path, info):
    with open(path, 'w') as file:
        _yaml.dump_all(info, file, default_flow_style=False)


def _to_dict_walk(node):
    ''' Converts mapping objects (subclassed from dict) to actual dict objects, including nested ones '''
    node = dict(node)
    for key, val in node.items():
        if isinstance(val, dict):
            node[key] = _to_dict_walk(val)
    return node


def save_user_settings(settings):
    settings = _to_dict_walk(settings)
    settings_filename = _user_settings_filename()
    _os.makedirs(_ospath.dirname(settings_filename), exist_ok=True)
    with open(settings_filename, 'w') as settings_file:
        _yaml.dump(dict(settings), settings_file, default_flow_style=False)


class TiffMap:

    TIFF_TYPES = {1: 'B', 2: 'c', 3: 'H', 4: 'L', 5: 'RATIONAL'}
    TYPE_SIZES = {'c': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'L': 4, 'RATIONAL': 8}

    def __init__(self, path, memmap_frames=False, verbose=False):
        if verbose:
            print('Reading info from {}'.format(path))
        self.path = _ospath.abspath(path)
        self.file = open(self.path, 'rb')
        self.byte_order = {b'II': '<', b'MM': '>'}[self.file.read(2)]
        self.file.seek(4)
        self.first_ifd_offset = self.read('L')
        self.file.seek(12)
        index_map_offset = self.read('I')

        # Read info from first IFD
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read('H')
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read('H')
            type = self.TIFF_TYPES[self.read('H')]
            count = self.read('L')
            if tag == 256:
                self.width = self.read(type, count)
            elif tag == 257:
                self.height = self.read(type, count)
            elif tag == 258:
                bits_per_sample = self.read(type, count)
                self.dtype = _np.dtype(self.byte_order + 'u' + str(int(bits_per_sample/8)))
        self.frame_shape = (self.height, self.width)
        self.frame_size = self.height*self.width

        # Collect image offsets
        self.file.seek(index_map_offset + 4)
        n_index_entries = self.read('I')
        ifd_offsets = []
        for i in range(n_index_entries):
            self.file.seek(index_map_offset + 20*i + 24)
            ifd_offsets.append(self.read('I'))
        self.image_offsets = [_ + 162 for _ in ifd_offsets]  # 2+12*13+4
        self.image_offsets[0] += 48     # there are some extra tags in the first one
        self.n_frames = len(self.image_offsets)

        if memmap_frames:
            self.get_frame = self.memmap_frame
        else:
            self.get_frame = self.read_frame

        self.lock = _threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        with self.lock:     # Otherwise we get messed up when reading frames from multiple threads
            if isinstance(it, tuple):
                if isinstance(it, int) or _np.issubdtype(it[0], _np.integer):
                    return self[it[0]][it[1:]]
                elif isinstance(it[0], slice):
                    indices = range(*it[0].indices(self.n_frames))
                    stack = _np.array([self.get_frame(_) for _ in indices])
                    if len(indices) == 0:
                        return stack
                    else:
                        if len(it) == 2:
                            return stack[:, it[1]]
                        elif len(it) == 3:
                            return stack[:, it[1], it[2]]
                        else:
                            raise IndexError
                elif it[0] == Ellipsis:
                    stack = self[it[0]]
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            elif isinstance(it, slice):
                indices = range(*it.indices(self.n_frames))
                return _np.array([self.get_frame(_) for _ in indices])
            elif it == Ellipsis:
                return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
            elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self):
        info = {'Byte Order': self.byte_order, 'File': self.path, 'Height': self.height,
                'Width': self.width, 'Data Type': self.dtype.name, 'Frames': self.n_frames}
        self.file.seek(28)
        comments_offset = self.read('L')
        self.file.seek(36)
        summary_length = self.read('L')
        info['Summary'] = _json.loads(self.read('c', summary_length).decode())
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read('H')
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset + 2 + i * 12)
            tag = self.read('H')
            type = self.TIFF_TYPES[self.read('H')]
            count = self.read('L')
            if count * self.TYPE_SIZES[type] > 4:
                self.file.seek(self.read('L'))
            if tag == 51123:
                readout = self.read(type, count).strip(b'\0')      # Strip null bytes which MM 1.4.22 adds
                mm_info = _json.loads(readout.decode())
                camera = mm_info['Camera']
                info['Camera'] = camera
                if camera + '-Output_Amplifier' in mm_info:
                    em = (mm_info[camera + '-Output_Amplifier'] == 'Electron Multiplying')
                    info['Electron Multiplying'] = em
                if camera + '-Gain' in mm_info:
                    info['EM Real Gain'] = int(mm_info[camera + '-Gain'])
                if camera + '-Pre-Amp-Gain' in mm_info:
                    info['Pre-Amp Gain'] = mm_info[camera + '-Pre-Amp-Gain']
                if camera + '-ReadoutMode' in mm_info:
                    info['Readout Mode'] = mm_info[camera + '-ReadoutMode']
                if camera + '-PixelReadoutRate' in mm_info:
                    info['Readout Rate'] = mm_info[camera + '-PixelReadoutRate'].split('-')[0].strip()
                if camera + '-Sensitivity/DynamicRange' in mm_info:
                    info['Gain Setting'] = mm_info[camera + '-Sensitivity/DynamicRange']
                if 'TIFilterBlock1-Label' in mm_info:
                    try:
                        info['Excitation Wavelength'] = int(mm_info['TIFilterBlock1-Label'][-3:])
                    except ValueError:      # Last three digits are not a number
                        info['Excitation Wavelength'] = None
                # Dump the rest
                info['Micro-Manager Metadata'] = mm_info
        if comments_offset:
            self.file.seek(comments_offset + 4)
            comments_length = self.read('L')
            if comments_length:
                comments_bytes = self.read('c', comments_length)
                try:
                    comments_json = comments_bytes.decode()
                except UnicodeDecodeError:
                    print('Did not find UTF-8 decoded comment bytes!')
                else:
                    info['Comments'] = _json.loads(comments_json)
        return info

    def memmap_frame(self, index):
        return _np.memmap(self.path, dtype=self.dtype, mode='r', offset=self.image_offsets[index], shape=self.frame_shape)

    def read_frame(self, index, array=None):
        self.file.seek(self.image_offsets[index])
        return _np.reshape(_np.fromfile(self.file, dtype=self.dtype, count=self.frame_size), self.frame_shape)

    def read(self, type, count=1):
        if type == 'c':
            return self.file.read(count)
        elif type == 'RATIONAL':
            return self.read_numbers('L') / self.read_numbers('L')
        else:
            return self.read_numbers(type, count)

    def read_numbers(self, type, count=1):
        size = self.TYPE_SIZES[type]
        fmt = self.byte_order + count * type
        return _struct.unpack(fmt, self.file.read(count * size))[0]

    def close(self):
        self.file.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = (byte_order != self.byte_order)
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class TiffMultiMap:

    def __init__(self, path, memmap_frames=False, verbose=False):
        self.path = _ospath.abspath(path)
        self.dir = _ospath.dirname(self.path)
        base, ext = _ospath.splitext(_ospath.splitext(self.path)[0])    # split two extensions as in .ome.tif
        base = base.replace('\\', '/')
        pattern = _re.compile(base + '_(\d*).ome.tif')    # This matches the basename + an appendix of the file number
        entries = [_ for _ in _os.scandir(self.dir) if _.is_file()]
        matches = [_re.match(pattern, _.path.replace('\\', '/')) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        paths_indices = [(int(_.group(1)), _.group(0)) for _ in matches]
        self.paths = [self.path] + [path for index, path in sorted(paths_indices)]
        self.maps = [TiffMap(path, verbose=verbose) for path in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = _np.insert(_np.cumsum(self.n_frames_per_map), 0, 0)
        self.dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = _np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or _np.issubdtype(it[0], _np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return _np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map in self.maps:
            map.close()

    def get_frame(self, index):
        # TODO deal with negative numbers
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i+1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self):
        info = self.maps[0].info()
        info['Frames'] = self.n_frames
        return info

    def tofile(self, file_handle, byte_order=None):
        for map in self.maps:
            map.tofile(file_handle, byte_order)


def to_raw_combined(basename, paths):
    raw_file_name = basename + '.ome.raw'
    with open(raw_file_name, 'wb') as file_handle:
        with TiffMap(paths[0]) as tif:
            tif.tofile(file_handle, '<')
            info = tif.info()
        for path in paths[1:]:
            with TiffMap(path) as tif:
                info_ = tif.info()
                info['Frames'] += info_['Frames']
                if 'Comments' in info_:
                    info['Comments'] = info_['Comments']
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
