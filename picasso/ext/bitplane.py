"""
    ext/bitplane
    ~~~~~~~~~~~~~~~~~~~~
    Utility functions to handle bitplane data
    :author: Maximilian T Strauss, 2021-2022
    :copyright: Copyright (c) 2021-2022 Maximilian T Strauss
"""

import os.path as _ospath
import numpy as np
import h5py
import datetime

try:
    from PyImarisWriter.ImarisWriterCtypes import *
    from PyImarisWriter import PyImarisWriter as PW

    IMSWRITER = True
except ModuleNotFoundError:
    IMSWRITER = False

if IMSWRITER:

    class MovieMapper:
        """
        MovieMapper class to map ims files.
        """

        def __init__(self, file, RL, lookup_dict, channel, frames, x, y, dtype):
            self.file = file
            self.RL = RL
            self.lookup_dict = lookup_dict
            self.channel = channel
            self.x = x
            self.y = y
            self.frames = frames
            self.dtype = dtype

        def __getitem__(self, item):
            return self.file["DataSet"][self.RL][self.lookup_dict[item]][self.channel][
                "Data"
            ][0][: self.y, : self.x]

        def __len__(self):
            return len(self.frames)

        def __iter__(self):
            for item in range(len(self.frames)):
                yield self.file["DataSet"][self.RL][self.lookup_dict[item]][
                    self.channel
                ]["Data"][0][: self.y, : self.x]

    class MovieMapperStack:
        """
        MovieMapperStack class to map ims files. This class is for z-stacks.
        """

        def __init__(self, file, RL, channel, frames, dtype, n_frames):
            self.file = file
            self.RL = RL
            self.channel = channel
            self.frames = frames
            self.dtype = dtype
            self.n_frames = n_frames

        def __getitem__(self, item):
            return self.file["DataSet"][self.RL][self.frames[0]][self.channel]["Data"][
                item, :, :
            ]

        def __len__(self):
            return self.n_frames

        def __iter__(self):
            for item in range(self.n_frames):
                yield self.file["DataSet"][self.RL][self.frames[0]][self.channel][
                    "Data"
                ][item, :, :]

    class IMSFile:
        """
        Class for handling IMS files.

        This class is assuming the following:
            * ResolutionLevel 0 is the default.
            * The image size doesn't change from one channel to another.

        TODO:
            * check if channel exists..

        """

        RL = "ResolutionLevel 0"

        def __init__(self, path, verbose=False):
            if verbose:
                print("Reading info from {}".format(path))
            self.path = _ospath.abspath(path)
            self.file = h5py.File(path, "r")

            self.frames = list(self.file["DataSet"][self.RL].keys())
            self.n_frames = len(self.frames)
            self.channels = list(self.file["DataSet"][self.RL][self.frames[0]].keys())

            self.set_channel(self.channels[0])

            self.frames_read = False
            self.lookup_dict = None

        def set_channel(self, channel):
            self.channel = channel
            self.img_size = np.array(
                self.file["DataSet"][self.RL][self.frames[0]][self.channel]["Data"]
            ).shape
            self.dtype = np.array(
                self.file["DataSet"][self.RL][self.frames[0]][self.channel]["Data"]
            ).dtype

            try:
                z = int(
                    "".join(
                        [
                            _.decode()
                            for _ in self.file["DataSetInfo"]["Image"].attrs["Z"]
                        ]
                    )
                )
            except KeyError:
                z = self.img_size[0]
            self.z = z

            try:
                self.x = int(
                    "".join(
                        [
                            _.decode()
                            for _ in self.file["DataSetInfo"]["Image"].attrs["X"]
                        ]
                    )
                )
                self.y = int(
                    "".join(
                        [
                            _.decode()
                            for _ in self.file["DataSetInfo"]["Image"].attrs["Y"]
                        ]
                    )
                )
            except KeyError:
                self.x = self.img_size[1]
                self.y = self.img_size[2]

            # The pixelsize is being estimated on the image dimensions

            ext_max0 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMax0"]
                    ]
                )
            )
            ext_min0 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMin0"]
                    ]
                )
            )

            ext_max1 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMax1"]
                    ]
                )
            )
            ext_min1 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMin1"]
                    ]
                )
            )

            ext_max2 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMax2"]
                    ]
                )
            )
            ext_min2 = float(
                "".join(
                    [
                        _.decode()
                        for _ in self.file["DataSetInfo"]["Image"].attrs["ExtMin2"]
                    ]
                )
            )

            delta_x = ext_max0 - ext_min0
            delta_y = ext_max1 - ext_min1

            px_x = delta_x / self.x * 1000
            px_y = delta_y / self.y * 1000

            px_nm = (px_x + px_y) / 2

            print(
                f"Image dimensions X: {delta_x} um Y: {delta_y} um. Pixelsize x {px_x}, y {px_y}, p {px_nm}"
            )

            self.pixelsize = px_nm

            self.ext_max0 = ext_max0
            self.ext_max1 = ext_max1
            self.ext_max2 = ext_max2

            self.ext_min0 = ext_min0
            self.ext_min1 = ext_min1
            self.ext_min2 = ext_min2

        def read_frames(self):
            if not self.frames_read:
                self.frames_int = np.array(
                    [int(_.split("TimePoint ")[1]) for _ in self.frames]
                )
                self.lookup_dict = dict(zip(self.frames_int, self.frames))

                self.frames_read = True

        def read_frame(self, frame):
            if not self.lookup_dict:
                self.read_frames()
            return self.file["DataSet"][self.RL][self.lookup_dict[frame]][self.channel][
                "Data"
            ][0]

        def read_z_stack(self):
            print("Reading stack")
            self.n_frames = self.z
            self.movie = MovieMapperStack(
                self.file, self.RL, self.channel, self.frames, self.dtype, self.n_frames
            )

        def read_stack(self):
            print("Reading movie")
            if not self.lookup_dict:
                self.read_frames()

            self.movie = MovieMapper(
                self.file,
                self.RL,
                self.lookup_dict,
                self.channel,
                self.frames,
                self.x,
                self.y,
                self.dtype,
            )

        def read_movie(self):

            if self.z > 1:
                self.read_z_stack()
            else:
                self.read_stack()

    class ImsCallback(PW.CallbackClass):
        def __init__(self):
            self.mUserDataProgress = 0

        def RecordProgress(self, progress, total_bytes_written):
            progress100 = int(progress * 100)
            if progress100 - self.mUserDataProgress >= 5:
                self.mUserDataProgress = progress100
                # print(
                #     "User Progress {}, Bytes written: {}".format(
                #         self.mUserDataProgress, total_bytes_written
                #     )
                # )

    def numpy_to_imaris(
        array, filename, colors, oversampling, viewport, info, z_min, z_max, pixelsize
    ):

        if len(array.shape) == 3:
            array = np.expand_dims(array, 1)

        dtype = array.dtype
        dtype_imaris = str(dtype)

        x = array.shape[3]
        y = array.shape[2]
        z = array.shape[1]
        c = array.shape[0]

        x_min = viewport[0][1]
        x_max = viewport[1][1]
        y_min = viewport[0][0]
        y_max = viewport[1][0]

        np_data = array.copy()

        image_size = PW.ImageSize(x=x, y=y, z=z, c=c, t=1)
        dimension_sequence = PW.DimensionSequence("x", "y", "z", "c", "t")
        block_size = image_size
        sample_size = PW.ImageSize(x=1, y=1, z=1, c=1, t=1)
        output_filename = filename

        options = PW.Options()
        options.mNumberOfThreads = 1
        options.mCompressionAlgorithmType = PW.eCompressionAlgorithmGzipLevel2
        options.mEnableLogProgress = True

        application_name = "PyImarisWriter"
        application_version = "1.0.0"

        callback_class = ImsCallback()
        converter = PW.ImageConverter(
            dtype_imaris,
            image_size,
            sample_size,
            dimension_sequence,
            block_size,
            output_filename,
            options,
            application_name,
            application_version,
            callback_class,
        )

        num_blocks = image_size / block_size

        block_index = PW.ImageSize()
        for c in range(num_blocks.c):
            block_index.c = c
            for t in range(num_blocks.t):
                block_index.t = t
                for z in range(num_blocks.z):
                    block_index.z = z
                    for y in range(num_blocks.y):
                        block_index.y = y
                        for x in range(num_blocks.x):
                            block_index.x = x
                            if converter.NeedCopyBlock(block_index):
                                converter.CopyBlock(np_data, block_index)

        adjust_color_range = True

        x_0 = (x_min) * pixelsize / 1000
        y_0 = (y_min) * pixelsize / 1000

        x_1 = (x_max) * pixelsize / 1000
        y_1 = (y_max) * pixelsize / 1000

        # TODO: Later use GlobalExtMin to add
        # Todo: Check for z
        try:
            x_0 += info[0]["ExtMin0"]
            y_0 += info[0]["ExtMin1"]

            x_1 += info[0]["ExtMin0"]
            y_1 += info[0]["ExtMin1"]

        except KeyError as e:
            print(f"Exception: {e}")

        try:
            z_base = (info[0]["ExtMin2"] + info[0]["ExtMax2"]) / 2
        except KeyError as e:
            print(f"Exception: {e}")
            z_base = 0

        if z_min == z_max == 0:
            z_0 = z_base - (image_size.z / 2) * pixelsize / 1000 / oversampling
            z_1 = z_base + (image_size.z / 2) * pixelsize / 1000 / oversampling
        else:
            z_0 = z_base + z_min / 1000
            z_1 = z_base + z_max / 1000

        # The image extends are not for multiple z-stacks.
        image_extents = PW.ImageExtents(x_0, y_0, z_0, x_1, y_1, z_1)

        print(f"Image dimensions {x_0, y_0, z_0, x_1, y_1, z_1}\n")

        parameters = PW.Parameters()
        parameters.set_value("Image", "Info", "PicassoExport")
        time_infos = [datetime.datetime.today()]

        # Channels
        color_infos = [PW.ColorInfo() for _ in range(image_size.c)]

        for idx, color in enumerate(colors):
            parameters.set_channel_name(idx, color)
            color_infos[idx].set_base_color(color)

        converter.Finish(
            image_extents, parameters, time_infos, color_infos, adjust_color_range
        )
        converter.Destroy()
        print("Wrote {} to {}".format("Minimal Example", output_filename))
