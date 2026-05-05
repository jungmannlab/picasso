"""Test ``picasso.simulate`` — astigmatic PSF, photon physics, structure
generation, and movie I/O round-trips.

Stochastic tests seed ``np.random.seed`` at the top of the test so every
run is deterministic on CI.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import io, simulate

from tests.conftest import CALIB_3D


# ---------------------------------------------------------------------------
# calculate_zpsf — polynomial astigmatic PSF
# ---------------------------------------------------------------------------


class TestCalculateZpsf:
    def test_known_values(self):
        # Replicates the in-module reference vector at simulate.py:65-83 so
        # any change to the polynomial formula is caught.
        cx = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
        cy = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
        z = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
        wx, wy = simulate.calculate_zpsf(z, cx, cy)
        expected = np.array(
            [
                4.90350522e01,
                7.13644987e02,
                5.52316597e03,
                2.61621620e04,
                9.06621337e04,
                2.54548124e05,
                6.14947219e05,
            ]
        )
        assert np.allclose(wx, expected, rtol=1e-4)
        assert np.allclose(wy, expected, rtol=1e-4)

    def test_scalar_input(self):
        cx = np.array(CALIB_3D["X Coefficients"], dtype=float)
        cy = np.array(CALIB_3D["Y Coefficients"], dtype=float)
        wx, wy = simulate.calculate_zpsf(0.0, cx, cy)
        # At z=0 only the constant term contributes
        assert wx == pytest.approx(cx[6])
        assert wy == pytest.approx(cy[6])

    def test_real_calibration_positive_widths(self):
        cx = np.array(CALIB_3D["X Coefficients"], dtype=float)
        cy = np.array(CALIB_3D["Y Coefficients"], dtype=float)
        z = np.linspace(-300.0, 300.0, 13)
        wx, wy = simulate.calculate_zpsf(z, cx, cy)
        assert np.all(wx > 0)
        assert np.all(wy > 0)


# ---------------------------------------------------------------------------
# noise + dtype helpers
# ---------------------------------------------------------------------------


class TestCheckType:
    def test_clamps_above_uint16_and_dtype(self):
        movie = np.array([[[10, 70000], [65535, 100000]]], dtype=np.int64)
        out = simulate.check_type(movie)
        assert out.dtype == np.dtype("<u2")
        assert out.max() == (2**16) - 1

    def test_below_max_passes_through(self):
        movie = np.array([[[1, 2], [3, 4]]], dtype=np.int64)
        out = simulate.check_type(movie)
        assert np.all(out == [[[1, 2], [3, 4]]])


class TestNoise:
    def test_noisy_clips_negative(self):
        np.random.seed(0)
        # zero image, large negative bias → entire result must clip to 0
        img = np.zeros((16, 16), dtype=float)
        out = simulate.noisy(img, mu=-1000.0, sigma=0.001)
        assert np.all(out == 0.0)

    def test_noisy_p_mean_approx(self):
        np.random.seed(1)
        img = np.zeros((128, 128), dtype=float)
        mu = 5.0
        out = simulate.noisy_p(img, mu=mu)
        # Poisson noise added on top of zero image — mean ~ mu, std ~ sqrt(mu)
        assert abs(out.mean() - mu) < 0.1


# ---------------------------------------------------------------------------
# Photon kinetics
# ---------------------------------------------------------------------------


class TestSamplePhotonRate:
    def test_zero_std_is_exact(self):
        out = simulate._sample_photon_rate(53.0, 0.0, 1.5)
        assert out == np.round(53.0 * 1.5)

    def test_clamped_at_zero(self):
        np.random.seed(2)
        # very negative mean → clamp to 0
        out = simulate._sample_photon_rate(-1e6, 1.0, 1.0)
        assert out == 0.0


class TestPaintgen:
    def test_returns_triple(self):
        np.random.seed(3)
        photonsinframe, timetrace, kinetics = simulate.paintgen(
            meandark=500,
            meanbright=50,
            frames=200,
            time=1.0,
            photonrate=10.0,
            photonratestd=0.0,
            photonbudget=1e6,
        )
        assert photonsinframe.shape == (200,)
        assert timetrace.ndim == 1
        assert len(kinetics) == 4
        # On-events count is non-negative
        assert kinetics[0] >= 0

    def test_zero_photonrate_gives_zero_frames(self):
        np.random.seed(4)
        photonsinframe, _, _ = simulate.paintgen(
            meandark=100,
            meanbright=20,
            frames=200,
            time=1.0,
            photonrate=0.0,
            photonratestd=0.0,
            photonbudget=1e6,
        )
        assert np.all(photonsinframe == 0.0)

    def test_simulated_kinetics_match_inputs(self):
        # With a long enough movie the simulated mean dark/bright times
        # should approach the requested means.
        np.random.seed(5)
        meandark = 500
        meanbright = 50
        _, _, kinetics = simulate.paintgen(
            meandark=meandark,
            meanbright=meanbright,
            frames=20000,
            time=1.0,
            photonrate=20.0,
            photonratestd=0.0,
            photonbudget=1e7,
        )
        # spotkinetics = [onevents, n_bright_frames, sim_dark, sim_bright]
        assert kinetics[2] == pytest.approx(meandark, rel=0.2)
        assert kinetics[3] == pytest.approx(meanbright, rel=0.3)


class TestDistphotons:
    def test_matches_paintgen_shapes(self):
        np.random.seed(6)
        structures = np.zeros((5, 1))
        photonsinframe, timetrace, kinetics = simulate.distphotons(
            structures=structures,
            itime=1.0,
            frames=100,
            taud=200.0,
            taub=20.0,
            photonrate=5.0,
            photonratestd=0.0,
            photonbudget=1e5,
        )
        assert photonsinframe.shape == (100,)
        assert len(kinetics) == 4


class TestDistphotonsxy:
    def test_total_rows_match_photon_count(self):
        np.random.seed(7)
        # one binding site with 50 photons in frame 0
        structures = np.array(
            [
                [10.0],  # x
                [10.0],  # y
                [1.0],  # exchange
                [0.0],  # group
                [0.0],  # z
            ]
        )
        photondist = np.array([[50]])  # 1 frame x 1 site
        photonpos = simulate.distphotonsxy(
            runner=0,
            photondist=photondist,
            structures=structures,
            psf=1.0,
            mode3Dstate=False,
            cx=[],
            cy=[],
        )
        assert photonpos.shape == (50, 2)

    def test_positions_concentrated_at_binding_site(self):
        np.random.seed(8)
        structures = np.array([[20.0], [20.0], [1.0], [0.0], [0.0]])
        photondist = np.array([[5000]])
        psf = 0.8
        photonpos = simulate.distphotonsxy(
            runner=0,
            photondist=photondist,
            structures=structures,
            psf=psf,
            mode3Dstate=False,
            cx=[],
            cy=[],
        )
        # Mean position close to binding site, std close to psf
        assert abs(photonpos[:, 0].mean() - 20.0) < 0.1
        assert abs(photonpos[:, 1].mean() - 20.0) < 0.1
        assert photonpos[:, 0].std() == pytest.approx(psf, rel=0.2)


class TestConvertMovie:
    def test_zero_photons_zero_frame(self):
        structures = np.array([[5.0], [5.0], [1.0], [0.0], [0.0]])
        photondist = np.array([[0]])
        frame = simulate.convertMovie(
            runner=0,
            photondist=photondist,
            structures=structures,
            imagesize=16,
            frames=1,
            psf=1.0,
            photonrate=0.0,
            background=0.0,
            noise=0.0,
            mode3Dstate=False,
            cx=[],
            cy=[],
        )
        assert frame.shape == (16, 16)
        assert np.all(frame == 0.0)

    def test_concentrated_photons_create_bright_pixel(self):
        np.random.seed(9)
        structures = np.array([[8.0], [8.0], [1.0], [0.0], [0.0]])
        photondist = np.array([[2000]])
        frame = simulate.convertMovie(
            runner=0,
            photondist=photondist,
            structures=structures,
            imagesize=16,
            frames=1,
            psf=0.5,
            photonrate=0.0,
            background=0.0,
            noise=0.0,
            mode3Dstate=False,
            cx=[],
            cy=[],
        )
        assert frame.sum() == 2000  # all photons land in the histogram
        # peak near (8, 8) — accounting for np.flipud applied at end
        peak_y, peak_x = np.unravel_index(frame.argmax(), frame.shape)
        flipped_y = (frame.shape[0] - 1) - peak_y
        assert abs(flipped_y - 8) <= 1
        assert abs(peak_x - 8) <= 1


# ---------------------------------------------------------------------------
# Structure generation
# ---------------------------------------------------------------------------


class TestDefineStructure:
    def test_centers_when_mean_true(self):
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([5.0, 10.0, 15.0])
        ex = np.array([1.0, 1.0, 1.0])
        z = np.array([0.0, 0.0, 0.0])
        s = simulate.defineStructure(x, y, ex, z, pixelsize=1.0, mean=True)
        # rows: x, y, ex, z
        assert s.shape == (4, 3)
        assert s[0].mean() == pytest.approx(0.0, abs=1e-9)
        assert s[1].mean() == pytest.approx(0.0, abs=1e-9)

    def test_no_center_when_mean_false(self):
        x = np.array([10.0, 20.0])
        y = np.array([5.0, 15.0])
        ex = np.array([1.0, 1.0])
        z = np.array([0.0, 0.0])
        s = simulate.defineStructure(x, y, ex, z, pixelsize=1.0, mean=False)
        np.testing.assert_allclose(s[0], x)
        np.testing.assert_allclose(s[1], y)

    def test_pixelsize_conversion(self):
        x = np.array([130.0, 260.0])
        y = np.array([0.0, 130.0])
        ex = np.array([1.0, 1.0])
        z = np.array([0.0, 0.0])
        s = simulate.defineStructure(x, y, ex, z, pixelsize=130.0, mean=False)
        np.testing.assert_allclose(s[0], [1.0, 2.0])
        np.testing.assert_allclose(s[1], [0.0, 1.0])


class TestGeneratePositions:
    def test_grid_arrangement(self):
        pos = simulate.generatePositions(
            number=4, imagesize=100, frame=10, arrangement=0
        )
        assert pos.shape == (4, 2)
        # All within [frame, imagesize-frame]
        assert np.all(pos >= 10)
        assert np.all(pos <= 90)

    def test_random_arrangement_count_and_range(self):
        np.random.seed(10)
        n = 25
        pos = simulate.generatePositions(
            number=n, imagesize=100, frame=10, arrangement=1
        )
        assert pos.shape == (n, 2)
        assert np.all(pos >= 10)
        assert np.all(pos <= 90)


class TestRotateStructure:
    def test_preserves_pairwise_distances(self):
        np.random.seed(11)
        s = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        s_rot = simulate.rotateStructure(s)

        def _pair_dists(arr):
            xy = arr[:2].T
            d = []
            for i in range(len(xy) - 1):
                for j in range(i + 1, len(xy)):
                    d.append(np.linalg.norm(xy[i] - xy[j]))
            return np.array(sorted(d))

        np.testing.assert_allclose(
            _pair_dists(s), _pair_dists(s_rot), atol=1e-9
        )

    def test_preserves_exchange_and_z(self):
        np.random.seed(12)
        s = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],  # exchange
                [10.0, 20.0, 30.0],  # z
            ]
        )
        s_rot = simulate.rotateStructure(s)
        np.testing.assert_array_equal(s_rot[2], s[2])
        np.testing.assert_array_equal(s_rot[3], s[3])


class TestIncorporateStructure:
    def test_full_incorporation_returns_full(self):
        np.random.seed(13)
        s = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        out = simulate.incorporateStructure(s, incorporation=1.0)
        assert out.shape == s.shape

    def test_zero_incorporation_returns_empty(self):
        np.random.seed(14)
        s = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        out = simulate.incorporateStructure(s, incorporation=0.0)
        assert out.shape[1] == 0


class TestRandomExchange:
    def test_preserves_xy_and_z(self):
        np.random.seed(15)
        pos = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],  # x
                [10.0, 11.0, 12.0, 13.0],  # y
                [1.0, 2.0, 3.0, 4.0],  # exchange
                [100.0, 200.0, 300.0, 400.0],  # z
            ]
        )
        out = simulate.randomExchange(pos.copy())
        np.testing.assert_array_equal(out[0], pos[0])
        np.testing.assert_array_equal(out[1], pos[1])
        np.testing.assert_array_equal(out[3], pos[3])
        # exchange row contains the same multiset
        np.testing.assert_array_equal(np.sort(out[2]), np.sort(pos[2]))


class TestPrepareStructures:
    def test_concatenates_per_position(self):
        np.random.seed(16)
        # 4 binding sites per origami
        structure = np.array(
            [
                [0.0, 1.0, 0.0, 1.0],  # x
                [0.0, 0.0, 1.0, 1.0],  # y
                [1.0, 1.0, 1.0, 1.0],  # exchange
                [0.0, 0.0, 0.0, 0.0],  # z
            ]
        )
        gridpos = np.array([[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]])
        newpos = simulate.prepareStructures(
            structure=structure,
            gridpos=gridpos,
            orientation=0,
            number=3,
            incorporation=1.0,
            exchange=0,
        )
        # output has 5 rows (x, y, exchange, group, z)
        assert newpos.shape == (5, 12)
        # group ID (row 3) takes integer values 0..N-1
        assert set(np.unique(newpos[3].astype(int))) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Movie I/O round-trip
# ---------------------------------------------------------------------------


class TestMovieRoundtrip:
    def test_save_and_load(self, tmp_path):
        # tiny synthetic movie: 5 frames x 8 x 8 px
        rng = np.random.default_rng(17)
        movie = rng.integers(0, 65535, size=(5, 8, 8), dtype=np.uint16)
        info = {
            "Byte Order": "<",
            "Data Type": "uint16",
            "Frames": 5,
            "Height": 8,
            "Width": 8,
            "Generated by": "test_simulate roundtrip",
        }
        raw_path = tmp_path / "movie.raw"
        simulate.saveMovie(str(raw_path), movie, info)
        # saveMovie writes both .raw and .yaml (via io.save_raw → save_info)
        yaml_path = tmp_path / "movie.yaml"
        assert raw_path.exists()
        assert yaml_path.exists()

        loaded, loaded_info = io.load_raw(str(raw_path))
        np.testing.assert_array_equal(np.asarray(loaded), movie)
        assert loaded_info[0]["Frames"] == 5
        assert loaded_info[0]["Width"] == 8

    def test_save_info_writes_yaml(self, tmp_path):
        info = {
            "Frames": 10,
            "Width": 32,
            "Height": 32,
            "Pixelsize": 130,
        }
        path = tmp_path / "info.yaml"
        simulate.saveInfo(str(path), info)
        assert path.exists()
        loaded = io.load_info(str(path))
        # save_info wraps into [info]; load_info also returns a list
        assert loaded[0]["Frames"] == 10
        assert loaded[0]["Pixelsize"] == 130
