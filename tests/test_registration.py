"""
Tests for picasso.registration: building the standalone channel-registration
calibration from beads and from the experimental blinking signal.
"""

import numpy as np
import pytest

from picasso import io, registration, wavelet
from picasso import transforms as tform

BOX = 7


def _matrix(entry) -> np.ndarray:
    """The 2x3 affine of a stored channel transform."""
    return tform.from_dict(entry).matrix[:2]


def _rotation(theta: float, tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), tx],
            [np.sin(theta), np.cos(theta), ty],
        ]
    )


def _apply(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, np.ones(len(xy))]) @ matrix.T


def _bead_image(xy, shape=(128, 128), amp=4000.0, sigma=1.3, seed=0):
    """A single-frame movie of Gaussian beads at ``xy``, with Poisson noise."""
    rng = np.random.RandomState(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.zeros(shape, dtype=np.float32)
    for x, y in xy:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return rng.poisson(img + 100).astype(np.uint16)[None]


def _bead_grid(start=20, stop=111, step=15):
    return np.array(
        [
            [x, y]
            for x in range(start, stop, step)
            for y in range(start, stop, step)
        ],
        dtype=float,
    )


def _blinking_movies(transforms, n_frames=80, seed=0, shape=(64, 64)):
    """Frame-synchronized movies sharing signal: every emitter in the reference
    frame also appears in each channel at that channel's transform."""
    rng = np.random.RandomState(seed)
    height, width = shape
    movies = [
        np.zeros((n_frames, height, width), dtype=np.float32)
        for _ in range(len(transforms) + 1)
    ]

    def render(frame, x, y, amp, sigma=1.2):
        xi, yi = int(round(x)), int(round(y))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                yy, xx = yi + dy, xi + dx
                if 0 <= yy < height and 0 <= xx < width:
                    frame[yy, xx] += amp * np.exp(
                        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
                    )

    for f in range(n_frames):
        for _ in range(rng.randint(5, 8)):
            x = rng.uniform(16, 48)
            y = rng.uniform(16, 48)
            amp = rng.uniform(2500, 4000)
            render(movies[0][f], x, y, amp)
            for c, matrix in enumerate(transforms, start=1):
                cx, cy = _apply(np.array([[x, y]]), matrix)[0]
                render(movies[c][f], cx, cy, amp)
    return [
        rng.poisson(np.maximum(m, 0) + 100).astype(np.uint16) for m in movies
    ]


class TestBeadRegistration:
    """Registration from fiducial bead images."""

    def test_recovers_each_channel_transform(self):
        grid = _bead_grid()
        truth = [_rotation(0.015, 5.0, -4.0), _rotation(-0.02, -6.0, 3.5)]
        movies = [_bead_image(grid, seed=1)] + [
            _bead_image(_apply(grid, m), seed=10 + i)
            for i, m in enumerate(truth)
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        assert calibration["model"] == registration.REGISTRATION_MODEL
        assert calibration["n_channels"] == 3
        assert calibration["source"] == "beads"
        for c, matrix in enumerate(truth, start=1):
            np.testing.assert_allclose(
                _matrix(calibration["channel_transforms"][c]),
                matrix,
                atol=0.15,
            )

    def test_reference_channel_is_the_identity(self):
        grid = _bead_grid()
        truth = _rotation(0.01, 3.0, -2.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        np.testing.assert_allclose(
            tform.from_dict(calibration["channel_transforms"][0]).matrix,
            np.eye(3),
        )

    def test_transforms_map_reference_into_the_channel(self):
        """The stored direction is reference -> channel, the direction
        ``localize.get_spots_multichannel`` maps detections in."""
        grid = _bead_grid()
        truth = _rotation(0.0, 6.0, -5.0)  # pure, unambiguous translation
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        mapped = tform.from_dict(calibration["channel_transforms"][1]).apply(
            grid
        )
        np.testing.assert_allclose(mapped, _apply(grid, truth), atol=0.15)

    def test_translation_model_fits_a_pure_shift_from_two_beads(self):
        """The 2-DOF model needs one pair, so a bead count that is below the
        affine minimum still registers - and it must not absorb anything
        beyond the shift."""
        sparse = np.array([[30.0, 30.0], [80.0, 80.0]])
        truth = _rotation(0.0, 6.0, -5.0)
        movies = [
            _bead_image(sparse, seed=1),
            _bead_image(_apply(sparse, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0, model="translation"
        )

        transform = tform.from_dict(calibration["channel_transforms"][1])
        assert transform.model == "translation"
        np.testing.assert_allclose(transform.shift, [6.0, -5.0], atol=0.15)
        np.testing.assert_allclose(
            transform.jacobian(sparse[:1])[0], np.eye(2)
        )

    def test_rejects_a_single_channel(self):
        with pytest.raises(ValueError, match="at least 2 channels"):
            registration.calibrate_channel_registration_from_beads(
                [_bead_image(_bead_grid())], box=BOX, minimum_ng=2000.0
            )

    def test_reports_too_few_bead_pairs(self):
        # two beads is below the affine minimum of three
        sparse = np.array([[30.0, 30.0], [80.0, 80.0]])
        movies = [
            _bead_image(sparse, seed=1),
            _bead_image(sparse + 4.0, seed=2),
        ]
        with pytest.raises(ValueError, match="matched bead pair"):
            registration.calibrate_channel_registration_from_beads(
                movies, box=BOX, minimum_ng=2000.0
            )


class TestSignalRegistration:
    """Registration from the experimental blinking signal."""

    def test_bootstraps_without_a_seed(self):
        """The new capability: build a registration from scratch, with no
        prior transform to pair at."""
        truth = _rotation(0.02, 4.0, -3.0)
        movies = _blinking_movies([truth])

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0
        )

        assert calibration["source"] == "signal"
        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.35
        )
        assert calibration["n_pairs"][0] >= 40
        assert calibration["rms"][0] < 1.0

    def test_a_seed_refines_a_stale_registration(self):
        truth = _rotation(0.02, 4.0, -3.0)
        movies = _blinking_movies([truth])
        stale = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.0]])
        seeds = [
            tform.identity(),
            tform.AffineTransform(matrix=np.vstack([stale, [0, 0, 1]])),
        ]

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0, seed_transforms=seeds
        )

        fitted = _matrix(calibration["channel_transforms"][1])
        np.testing.assert_allclose(fitted, truth, atol=0.35)
        # and it moved towards the truth rather than sitting on the seed
        assert np.abs(fitted - truth).max() < 0.5 * np.abs(stale - truth).max()

    def test_registers_several_channels(self):
        truth = [_rotation(0.02, 4.0, -3.0), _rotation(-0.015, -5.0, 2.0)]
        movies = _blinking_movies(truth)

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0
        )

        assert calibration["n_channels"] == 3
        for c, matrix in enumerate(truth, start=1):
            np.testing.assert_allclose(
                _matrix(calibration["channel_transforms"][c]),
                matrix,
                atol=0.35,
            )

    def test_frame_bounds_limit_the_sample(self):
        truth = _rotation(0.0, 3.0, -2.0)
        movies = _blinking_movies([truth])

        calibration = registration.calibrate_channel_registration_from_signal(
            movies,
            box=BOX,
            minimum_ng=800.0,
            frame_bounds=[[0, 39]],
            max_frames=20,
        )

        assert calibration["n_sampled_frames"] <= 20
        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.35
        )

    def test_the_bootstrap_survives_a_tied_vote(self):
        """A dense cloud saturates the RANSAC inlier count - hundreds of wrong
        candidates map *every* point within the tolerance and tie at the
        maximum - so the winner may not be left to the sampling order: which
        spots are detected varies with the machine (the identification's thread
        count, a spot sitting on the net-gradient threshold), and a tie broken
        by order then registers a different transform on a different machine.
        Thinning the point sets stands in for that here; the detections are
        whole pixels, as identified ones are."""
        rng = np.random.RandomState(0)
        truth = np.array([[1.0, 0.0, 5.4], [0.0, 1.0, -3.7]])
        ref_by_frame, chan_by_frame = {}, {}
        for f in range(25):
            xy = rng.uniform(14, 34, size=(rng.randint(5, 8), 2))
            ref_by_frame[f] = np.rint(xy)
            chan_by_frame[f] = np.rint(_apply(xy, truth))

        for trial in range(4):
            drop = np.random.RandomState(trial)

            def thinned(by_frame):
                out = {}
                for f, xy in by_frame.items():
                    kept = xy[drop.rand(len(xy)) > 0.06]
                    if len(kept):
                        out[f] = kept
                return out

            info = registration.register_from_point_sets(
                thinned(ref_by_frame), thinned(chan_by_frame), "affine", BOX
            )

            np.testing.assert_allclose(
                info["transform"].matrix[:2], truth, atol=0.8
            )

    def test_rejects_an_empty_frame_range(self):
        movies = _blinking_movies([_rotation(0.0, 2.0, 0.0)], n_frames=10)
        with pytest.raises(ValueError, match="No frames"):
            registration.calibrate_channel_registration_from_signal(
                movies, box=BOX, minimum_ng=800.0, frame_bounds=[[50, 60]]
            )

    def test_reports_the_channel_that_could_not_be_registered(self):
        """A channel carrying unrelated signal names itself in the error."""
        movies = _blinking_movies([_rotation(0.0, 3.0, -2.0)], n_frames=40)
        rng = np.random.RandomState(7)
        movies[1] = rng.poisson(np.full(movies[1].shape, 100.0)).astype(
            np.uint16
        )
        with pytest.raises(ValueError, match="Channel 1"):
            registration.calibrate_channel_registration_from_signal(
                movies, box=BOX, minimum_ng=800.0
            )


class TestCalibrationFile:
    def test_round_trips_through_the_generic_calibration_io(self, tmp_path):
        """The registration file has no large arrays, so it rides the existing
        YAML calibration path with no new I/O code."""
        grid = _bead_grid()
        truth = _rotation(0.01, 4.0, -3.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]
        path = str(tmp_path / "channel_registration.yaml")

        saved = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0, path=path
        )
        loaded = io.load_any_calibration(path)

        assert loaded["model"] == registration.REGISTRATION_MODEL
        assert loaded["n_channels"] == saved["n_channels"]
        np.testing.assert_allclose(
            _matrix(loaded["channel_transforms"][1]),
            _matrix(saved["channel_transforms"][1]),
        )

    def test_transforms_are_consumable_by_the_multichannel_helpers(self):
        """The stored transforms are the same wire format the existing
        multichannel geometry helpers already take."""
        import pandas as pd

        from picasso import localize

        grid = _bead_grid()
        truth = _rotation(0.0, 6.0, -5.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]
        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        ids = pd.DataFrame(
            {
                "frame": [0, 0],
                "x": [40, 60],
                "y": [40, 60],
                "net_gradient": [1.0, 1.0],
            }
        )
        residuals = localize.channel_roi_residuals(
            ids, calibration["channel_transforms"]
        )
        assert residuals.shape == (2, 2, 2)
        # the reference channel's box sits on the detection itself
        np.testing.assert_array_equal(residuals[:, 0, :], 0.0)


class TestSplitFovRegistration:
    """Channels imaged side by side on one sensor, registered from that single
    movie's own regions."""

    H, W = 64, 128
    REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]
    # the fine misregistration on top of the 64 px region offset
    FX, FY = 1.4, -0.85

    def _render(self, frame, x, y, amp, sigma=1.25):
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        frame += amp * np.exp(-0.5 * ((i - x) ** 2 + (j - y) ** 2) / sigma**2)

    def _blinking_movie(self, n_frames=70, seed=4):
        rng = np.random.RandomState(seed)
        movie = np.zeros((n_frames, self.H, self.W))
        for f in range(n_frames):
            for _ in range(rng.randint(4, 7)):
                x = rng.uniform(12, 52)
                y = rng.uniform(12, 52)
                amp = rng.uniform(2500, 4000)
                self._render(movie[f], x, y, amp)
                self._render(movie[f], x + 64 + self.FX, y + self.FY, amp)
        return rng.poisson(np.maximum(movie, 0) + 100).astype(np.uint16)

    def _bead_movie(self, seed=1):
        rng = np.random.RandomState(seed)
        img = np.zeros((self.H, self.W))
        for x in range(10, 56, 11):
            for y in range(10, 56, 11):
                self._render(img, x, y, 4000.0, 1.3)
                self._render(img, x + 64 + self.FX, y + self.FY, 4000.0, 1.3)
        return rng.poisson(img + 100).astype(np.uint16)[None]

    def _local_shift(self, calibration):
        """The region-local translation the registration recovered."""
        matrix = _matrix(calibration["channel_registration"][1])
        return matrix[0, 2], matrix[1, 2]

    def test_beads_recover_the_fine_offset(self):
        calibration = registration.calibrate_channel_registration_from_beads(
            [self._bead_movie()],
            box=BOX,
            minimum_ng=2000.0,
            regions=self.REGIONS,
        )

        assert calibration["split_fov"] is True
        assert calibration["n_channels"] == 2
        dx, dy = self._local_shift(calibration)
        assert dx == pytest.approx(self.FX, abs=0.15)
        assert dy == pytest.approx(self.FY, abs=0.15)

    def test_signal_recovers_the_fine_offset(self):
        calibration = registration.calibrate_channel_registration_from_signal(
            [self._blinking_movie()],
            box=BOX,
            minimum_ng=800.0,
            regions=self.REGIONS,
        )

        assert calibration["split_fov"] is True
        dx, dy = self._local_shift(calibration)
        # `identify` reports a centroid, so a sub-pixel bias remains - as for
        # the separate-movie signal path
        assert dx == pytest.approx(self.FX, abs=0.35)
        assert dy == pytest.approx(self.FY, abs=0.35)

    def test_the_registration_stores_the_region_layout(self):
        calibration = registration.calibrate_channel_registration_from_beads(
            [self._bead_movie()],
            box=BOX,
            minimum_ng=2000.0,
            regions=self.REGIONS,
        )

        assert calibration["regions"] == self.REGIONS
        assert calibration["reference"] == 0
        # the absolute transform carries the region offset, the region-local
        # one does not - that is what lets the ROIs be re-drawn
        absolute = _matrix(calibration["channel_transforms"][1])
        assert absolute[0, 2] == pytest.approx(64 + self.FX, abs=0.2)
        assert self._local_shift(calibration)[0] == pytest.approx(
            self.FX, abs=0.15
        )

    def test_the_channels_can_be_replaced_at_redrawn_rois(self):
        """The stored registration is ROI-agnostic, so moving the ROIs moves
        the channels with them."""
        from picasso import localize

        calibration = registration.calibrate_channel_registration_from_beads(
            [self._bead_movie()],
            box=BOX,
            minimum_ng=2000.0,
            regions=self.REGIONS,
        )
        moved = [[[0, 0], [64, 64]], [[0, 70], [64, 134]]]

        _, _, transforms = localize.split_fov_fit_geometry(calibration, moved)

        placed = transforms[1].matrix[:2]
        assert placed[0, 2] == pytest.approx(70 + self.FX, abs=0.2)

    def test_rejects_more_than_one_movie(self):
        movie = self._bead_movie()
        with pytest.raises(ValueError, match="single bead movie"):
            registration.calibrate_channel_registration_from_beads(
                [movie, movie],
                box=BOX,
                minimum_ng=2000.0,
                regions=self.REGIONS,
            )

    def test_signal_rejects_more_than_one_movie(self):
        movie = self._blinking_movie(n_frames=10)
        with pytest.raises(ValueError, match="single movie"):
            registration.calibrate_channel_registration_from_signal(
                [movie, movie],
                box=BOX,
                minimum_ng=800.0,
                regions=self.REGIONS,
            )


class TestSplitFovFlips:
    """A splitter commonly folds one channel about an axis. The drawn ROIs fix
    where a channel sits but not how it is oriented, so every mirror is tried
    and the one that pairs the most beads wins."""

    H, W = 64, 128
    REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]
    FX, FY = 1.4, -0.85

    def _render(self, img, x, y, amp=4000.0, sigma=1.3):
        yy, xx = np.mgrid[0 : self.H, 0 : self.W]
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

    def _random_beads(self, rng, n=34):
        """Randomly placed beads. A regular lattice will not do: its mirror is
        indistinguishable from a translation, so the orientation would be
        genuinely ambiguous rather than merely hard."""
        pts = []
        while len(pts) < n:
            p = rng.uniform(8, 55, 2)
            if all(np.hypot(*(p - q)) > 6 for q in pts):
                pts.append(p)
        return np.array(pts)

    def _movie(self, map_fn, seed=3, n_fov=1):
        rng = np.random.RandomState(seed)
        frames = []
        for _ in range(n_fov):
            img = np.zeros((self.H, self.W))
            for x, y in self._random_beads(rng):
                self._render(img, x, y)
                self._render(img, *map_fn(x, y))
            frames.append(rng.poisson(img + 100))
        return np.stack(frames).astype(np.uint16)

    # (sx, sy) mirror signs -> the mapping the channel actually applies
    ORIENTATIONS = {
        "none": (lambda s, x, y: (x + 64 + s.FX, y + s.FY), 1.0),
        "flip-x": (lambda s, x, y: (64 + (63 - x) + s.FX, y + s.FY), -1.0),
        "flip-y": (lambda s, x, y: (x + 64 + s.FX, (63 - y) + s.FY), -1.0),
        "flip-xy": (
            lambda s, x, y: (64 + (63 - x) + s.FX, (63 - y) + s.FY),
            1.0,
        ),
    }

    @pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
    def test_every_mirror_orientation_is_recovered(self, orientation):
        map_fn, expected_det = self.ORIENTATIONS[orientation]
        movie = self._movie(lambda x, y: map_fn(self, x, y))

        calibration = registration.calibrate_channel_registration_from_beads(
            [movie], box=BOX, minimum_ng=2000.0, regions=self.REGIONS
        )

        matrix = _matrix(calibration["channel_registration"][1])
        # a single mirror flips the sign of the determinant; two restore it
        assert np.linalg.det(matrix[:, :2]) == pytest.approx(
            expected_det, abs=0.05
        )
        assert calibration["rms"][0] < 0.15
        # and the fine offset is recovered in whichever axis is not mirrored
        if orientation in ("none", "flip-y"):
            assert matrix[0, 2] == pytest.approx(self.FX, abs=0.2)
        if orientation in ("none", "flip-x"):
            assert matrix[1, 2] == pytest.approx(self.FY, abs=0.2)

    def test_a_mirrored_channel_maps_beads_onto_beads(self):
        """The end-to-end property: the recovered transform must actually send
        each reference bead onto its own partner."""
        map_fn, _ = self.ORIENTATIONS["flip-x"]
        movie = self._movie(lambda x, y: map_fn(self, x, y))

        calibration = registration.calibrate_channel_registration_from_beads(
            [movie], box=BOX, minimum_ng=2000.0, regions=self.REGIONS
        )

        transform = tform.from_dict(calibration["channel_transforms"][1])
        rng = np.random.RandomState(3)
        beads = self._random_beads(rng)
        expected = np.array([map_fn(self, x, y) for x, y in beads])
        np.testing.assert_allclose(transform.apply(beads), expected, atol=0.3)

    def test_signal_registration_also_searches_orientations(self):
        """The same seeding is used when registering on blinking signal."""
        rng = np.random.RandomState(11)
        n_frames = 60
        movie = np.zeros((n_frames, self.H, self.W))
        for f in range(n_frames):
            for _ in range(rng.randint(4, 7)):
                x, y = rng.uniform(12, 52), rng.uniform(12, 52)
                amp = rng.uniform(2500, 4000)
                self._render(movie[f], x, y, amp, 1.2)
                # channel 1 is mirrored in x
                self._render(
                    movie[f], 64 + (63 - x) + self.FX, y + self.FY, amp, 1.2
                )
        movie = rng.poisson(np.maximum(movie, 0) + 100).astype(np.uint16)

        calibration = registration.calibrate_channel_registration_from_signal(
            [movie], box=BOX, minimum_ng=800.0, regions=self.REGIONS
        )

        matrix = _matrix(calibration["channel_registration"][1])
        assert np.linalg.det(matrix[:, :2]) == pytest.approx(-1.0, abs=0.1)


class TestMultiFovBeads:
    """Several fields of view in one bead movie. Different fields land on the
    same sensor coordinates, so a bead may only ever be paired with one in the
    *same* frame."""

    H, W = 64, 128
    REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]
    FX, FY = 1.4, -0.85

    def _movie(self, n_fov, seed=5):
        rng = np.random.RandomState(seed)
        yy, xx = np.mgrid[0 : self.H, 0 : self.W]
        frames = []
        for _ in range(n_fov):
            img = np.zeros((self.H, self.W))
            # a fresh, randomly placed field each frame
            pts = []
            while len(pts) < 18:
                p = rng.uniform(8, 55, 2)
                if all(np.hypot(*(p - q)) > 7 for q in pts):
                    pts.append(p)
            for x, y in pts:
                for cx, cy in (
                    (x, y),
                    (x + 64 + self.FX, y + self.FY),
                ):
                    img += 4000 * np.exp(
                        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.3**2)
                    )
            frames.append(rng.poisson(img + 100))
        return np.stack(frames).astype(np.uint16)

    def _register(self, movie, multi_fov):
        return registration.calibrate_channel_registration_from_beads(
            [movie],
            box=BOX,
            minimum_ng=2000.0,
            regions=self.REGIONS,
            multi_fov=multi_fov,
        )

    def test_every_field_contributes_correspondences(self):
        one = self._register(self._movie(1), multi_fov=True)
        three = self._register(self._movie(3), multi_fov=True)

        # three fields give about three times the pairs, all constraining one
        # global transform
        assert three["n_pairs"][0] > 2.5 * one["n_pairs"][0]
        matrix = _matrix(three["channel_registration"][1])
        assert matrix[0, 2] == pytest.approx(self.FX, abs=0.15)
        assert matrix[1, 2] == pytest.approx(self.FY, abs=0.15)

    def test_averaging_distinct_fields_is_worse(self):
        """Without ``multi_fov`` the frames are averaged, which smears beads
        of different fields together - the reason the flag exists."""
        movie = self._movie(3)

        pooled = self._register(movie, multi_fov=False)
        per_field = self._register(movie, multi_fov=True)

        assert per_field["n_pairs"][0] > pooled["n_pairs"][0]

    def test_a_single_frame_movie_is_unaffected(self):
        movie = self._movie(1)
        assert self._register(movie, multi_fov=True)["n_pairs"] == (
            self._register(movie, multi_fov=False)["n_pairs"]
        )


class TestWaveletDetection:
    """Both builders can detect by wavelet segmentation (Izeddin et al.,
    2012) instead of by the net gradient, as selected for the movie."""

    WAVELET = wavelet.WaveletParameters()

    def test_beads(self):
        grid = _bead_grid()
        truth = _rotation(0.015, 5.0, -4.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=None, wavelet=self.WAVELET
        )

        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.15
        )
        assert calibration["identification_method"] == "wavelet"
        assert calibration["wavelet"] == self.WAVELET.to_dict()
        assert "minimum_ng" not in calibration

    def test_signal(self):
        truth = _rotation(0.02, 4.0, -3.0)
        movies = _blinking_movies([truth])

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=None, wavelet=self.WAVELET
        )

        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.35
        )
        assert calibration["identification_method"] == "wavelet"

    def test_net_gradient_is_recorded_as_before(self):
        grid = _bead_grid()
        movies = [_bead_image(grid, seed=1), _bead_image(grid, seed=2)]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        assert calibration["identification_method"] == "net gradient"
        assert calibration["minimum_ng"] == 2000.0
        assert "wavelet" not in calibration

    def test_the_calibration_file_is_plain_yaml(self, tmp_path):
        grid = _bead_grid()
        movies = [_bead_image(grid, seed=1), _bead_image(grid, seed=2)]
        path = str(tmp_path / "registration.yaml")

        registration.calibrate_channel_registration_from_beads(
            movies,
            box=BOX,
            minimum_ng=None,
            path=path,
            wavelet=self.WAVELET,
        )

        with open(path) as f:
            text = f.read()
        assert "!!python" not in text
        assert io.load_any_calibration(path)["wavelet"] == (
            self.WAVELET.to_dict()
        )

    def test_detections_by_frame(self):
        truth = _rotation(0.0, 0.0, 0.0)
        movie = _blinking_movies([truth], n_frames=6)[0]

        by_frame = registration.detections_by_frame(
            movie, None, BOX, np.arange(6), wavelet=self.WAVELET
        )

        assert sorted(by_frame) == list(range(6))
        for xy in by_frame.values():
            assert xy.shape[1] == 2 and len(xy) >= 4
