"""Tests for picasso.transforms (the selectable 2D transform models).

Everything downstream - the channel registration in picasso.spline, the
lateral corrections in picasso.lib, and the per-spot Jacobians the spline fit
kernels consume - trusts this module blindly, so it is tested on its own
terms: exact recovery of known transforms, the Jacobian against finite
differences, the exactness of the region-composition primitive, and
round trips through both serialization formats Picasso uses.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from picasso import transforms

# ---------------------------------------------------------------------------
# Fixtures: synthetic correspondences on a realistically large chip
# ---------------------------------------------------------------------------

# 2048 px, so the polynomial conditioning is genuinely exercised: without the
# input normalization a degree-3 fit here loses ten significant digits.
CHIP = 2048.0

TRANSLATION_TRUTH = np.array(
    [[1.0, 0.0, 3.5], [0.0, 1.0, -2.25], [0.0, 0.0, 1.0]]
)
AFFINE_TRUTH = np.array(
    [[1.004, -0.011, 3.5], [0.008, 0.997, -2.25], [0.0, 0.0, 1.0]]
)
PROJECTIVE_TRUTH = np.array(
    [[1.002, -0.010, 4.0], [0.009, 0.998, -3.0], [1.0e-6, -2.0e-6, 1.0]]
)


def _src(n=60, seed=0):
    return np.random.RandomState(seed).uniform(10.0, CHIP - 10, size=(n, 2))


def _apply_matrix(matrix, xy):
    """Reference implementation: homogeneous map with perspective divide."""
    num = xy @ matrix[:2, :2].T + matrix[:2, 2]
    w = xy @ matrix[2, :2] + matrix[2, 2]
    return num / w[:, None]


def _poly_truth(xy, degree=3):
    """A smooth warp of a few pixels, the scale of real field distortion.

    ``degree`` selects the highest monomial present, so each polynomial model
    can be tested on data it is able to represent exactly.
    """
    x, y = xy[:, 0] / 1000.0, xy[:, 1] / 1000.0
    out = np.column_stack(
        [
            xy[:, 0] + 3.0 + 0.5 * x * y + 0.2 * x**2,
            xy[:, 1] - 2.0 + 0.3 * y**2 - 0.15 * x * y,
        ]
    )
    if degree >= 3:
        out += np.column_stack([-0.1 * y**3, -0.4 * x**2 * y])
    return out


def _model(model, n=60, seed=0):
    """A fitted transform of each model, on data it can represent exactly."""
    src = _src(n, seed)
    if model == "translation":
        dst = _apply_matrix(TRANSLATION_TRUTH, src)
    elif model == "affine":
        dst = _apply_matrix(AFFINE_TRUTH, src)
    elif model == "projective":
        dst = _apply_matrix(PROJECTIVE_TRUTH, src)
    else:
        dst = _poly_truth(src, transforms._polynomial_degree(model))
    return src, dst, transforms.estimate(src, dst, model)


ALL_MODELS = list(transforms.MODELS)


@pytest.fixture(params=ALL_MODELS)
def model(request):
    src, dst, transform = _model(request.param)
    return request.param, src, dst, transform


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


class TestEstimate:
    """Each model recovers the transform it was generated from."""

    def test_affine_is_bit_identical_to_the_previous_implementation(self):
        """The affine path must not move a single bit: it is the default, and
        every existing calibration was fitted with it.

        ``_legacy_*`` are verbatim copies of the ``(2, 3)`` implementations
        this module replaced (``picasso.localize.estimate_affine_transform``
        and ``apply_affine_transform``). Keeping them here pins the claim even
        though the originals are gone.
        """

        def _legacy_estimate(src_xy, dst_xy):
            src = np.asarray(src_xy, dtype=np.float64)
            dst = np.asarray(dst_xy, dtype=np.float64)
            a = np.hstack([src, np.ones((len(src), 1))])
            solution, *_ = np.linalg.lstsq(a, dst, rcond=None)
            return solution.T.astype(np.float64)

        def _legacy_apply(xy, transform):
            xy = np.asarray(xy, dtype=np.float64)
            transform = np.asarray(transform, dtype=np.float64)
            return xy @ transform[:, :2].T + transform[:, 2]

        src = _src()
        dst = _apply_matrix(AFFINE_TRUTH, src)
        old = _legacy_estimate(src, dst)  # (2, 3)
        new = transforms.estimate(src, dst, "affine")
        assert np.array_equal(old, new.matrix[:2, :])
        assert np.array_equal(_legacy_apply(src, old), new.apply(src))

    def test_affine_recovers_known_transform(self):
        src, dst, t = _model("affine")
        assert np.allclose(t.matrix, AFFINE_TRUTH, atol=1e-10)
        assert np.allclose(t.apply(src), dst, atol=1e-10)

    def test_projective_dlt_recovers_known_homography(self):
        src, dst, t = _model("projective")
        assert np.allclose(t.matrix, PROJECTIVE_TRUTH, atol=1e-10)
        assert np.allclose(t.apply(src), dst, atol=1e-9)

    def test_projective_beats_affine_on_perspective_data(self):
        """The point of the feature: a homography registers homography data
        better than an affine can."""
        src, dst, projective = _model("projective")
        affine = transforms.estimate(src, dst, "affine")
        rms = lambda t: np.sqrt(  # noqa: E731
            np.mean(np.sum((t.apply(src) - dst) ** 2, axis=1))
        )
        assert rms(projective) < 0.05 * rms(affine)

    def test_polynomial_degree_3_recovers_cubic_warp_at_chip_scale(self):
        """Fails without the input normalization: raw coordinates cubed on a
        2048 chip condition the design matrix at ~1e10."""
        src, dst, t = _model("polynomial3")
        assert np.allclose(t.apply(src), dst, atol=1e-8)

    def test_polynomial_degree_2_cannot_represent_a_cubic(self):
        src = _src()
        cubic = _poly_truth(src, degree=3)
        deg2 = transforms.estimate(src, cubic, "polynomial2")
        deg3 = transforms.estimate(src, cubic, "polynomial3")
        err2 = np.abs(deg2.apply(src) - cubic).max()
        err3 = np.abs(deg3.apply(src) - cubic).max()
        assert err3 < 1e-8 < err2

    def test_estimate_is_exact_on_data_each_model_can_represent(self, model):
        _, src, dst, t = model
        assert np.allclose(t.apply(src), dst, atol=1e-7)

    def test_domain_is_the_source_bounding_box(self, model):
        _, src, _, t = model
        assert np.allclose(t.domain[0], src.min(axis=0))
        assert np.allclose(t.domain[1], src.max(axis=0))


class TestEstimateValidation:
    """Bad models, thin data and degenerate correspondences."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("affine", 3),
            ("projective", 4),
            ("polynomial2", 6),
            ("polynomial3", 10),
        ],
    )
    def test_min_points(self, model, expected):
        assert transforms.min_points(model) == expected

    @pytest.mark.parametrize(
        "name", ["lwm", "polynomial", "polynomial4", "poly2", ""]
    )
    def test_unknown_model_raises(self, name):
        """One flat name space: a bare "polynomial" or an unoffered degree is
        just an unknown model, so no state like ("affine", 3) exists."""
        with pytest.raises(ValueError, match="Unknown transform model"):
            transforms.estimate(_src(), _src(), name)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_too_few_correspondences_raises(self, model):
        n = transforms.min_points(model) - 1
        src = _src(n)
        with pytest.raises(ValueError, match="at least"):
            transforms.estimate(src, src, model)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_barely_enough_correspondences_warns(self, model):
        n = transforms.min_points(model)
        src = _src(n, seed=5)
        # A translation has no simpler model to fall back on, so it warns
        # about the noise it inherits rather than about interpolating it.
        expected = (
            "inherits their localization noise"
            if model == "translation"
            else "interpolate the noise"
        )
        with pytest.warns(UserWarning, match=expected):
            transforms.estimate(src, src, model)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            transforms.estimate(_src(10), _src(11), "affine")

    def test_collinear_points_raise_for_a_polynomial(self):
        x = np.linspace(0.0, 1000.0, 30)
        src = np.column_stack([x, 2.0 * x + 5.0])  # exactly collinear
        with pytest.raises(ValueError, match="degenerate"):
            transforms.estimate(src, src, "polynomial2")


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


class TestJacobian:
    """The local linear part - what carries a non-affine model into the
    spline fit kernels."""

    def test_matches_central_finite_differences(self, model):
        _, _, _, t = model
        probe = np.random.RandomState(7).uniform(100, CHIP - 100, size=(12, 2))
        h = 1e-3
        analytic = t.jacobian(probe)
        numeric = np.empty_like(analytic)
        for j in range(2):
            step = np.zeros(2)
            step[j] = h
            numeric[:, :, j] = (
                t.apply(probe + step) - t.apply(probe - step)
            ) / (2 * h)
        assert np.allclose(analytic, numeric, atol=1e-7)

    def test_affine_jacobian_is_constant_and_is_the_linear_part(self):
        _, _, t = _model("affine")
        probe = np.random.RandomState(1).uniform(0, CHIP, size=(9, 2))
        J = t.jacobian(probe)
        assert np.allclose(J, AFFINE_TRUTH[:2, :2])
        assert np.array_equal(J[0], J[-1])

    def test_non_affine_jacobian_varies_across_the_field(self):
        """The whole reason the kernels need a per-spot Jacobian."""
        for name in ["projective", "polynomial3"]:
            _, _, t = _model(name)
            corners = np.array([[10.0, 10.0], [CHIP - 10, CHIP - 10]])
            J = t.jacobian(corners)
            assert not np.allclose(J[0], J[1], atol=1e-9)

    def test_returns_a_writable_copy(self, model):
        """Callers stack Jacobians into a (n, C, 4) buffer, so a broadcast
        read-only view would be a trap."""
        _, _, _, t = model
        J = t.jacobian(np.zeros((3, 2)))
        J[0, 0, 0] = 12.0  # must not raise
        assert J.shape == (3, 2, 2)


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


class TestInverse:
    def test_affine_and_projective_invert_exactly(self):
        probe = np.random.RandomState(2).uniform(50, CHIP - 50, size=(20, 2))
        for kind in ("affine", "projective"):
            _, _, t = _model(kind)
            assert np.allclose(
                t.inverse().apply(t.apply(probe)), probe, atol=1e-9
            )
            assert t.inverse().model == kind

    def test_polynomial_roundtrip_is_approximate_but_measured(self):
        """The reverse branch is an independent fit, so the round trip is
        approximate - and the transform says by how much."""
        src, _, t = _model("polynomial3")
        back = t.inverse().apply(t.apply(src))
        rms = np.sqrt(np.mean(np.sum((back - src) ** 2, axis=1)))
        assert 0.0 < t.roundtrip_rms_px < 0.05
        # the stored figure is measured on a grid over `domain`, so it
        # predicts the error on the correspondences themselves
        assert rms == pytest.approx(t.roundtrip_rms_px, abs=0.05)

    def test_polynomial_inverse_swaps_the_branches(self):
        _, _, t = _model("polynomial2")
        inv = t.inverse()
        assert np.array_equal(inv.forward, t.reverse)
        assert np.array_equal(inv.reverse, t.forward)


# ---------------------------------------------------------------------------
# compose_translations - the split-FOV region primitive
# ---------------------------------------------------------------------------


class TestComposeTranslations:
    """Pins the exactness claim that lets split-FOV regions be re-placed at
    fit time for every model, with no refit."""

    PRE = np.array([13.5, -7.25])
    POST = np.array([-4.0, 11.5])

    def test_is_exact(self, model):
        _, _, _, t = model
        probe = np.random.RandomState(4).uniform(50, CHIP - 50, size=(15, 2))
        composed = t.compose_translations(pre=self.PRE, post=self.POST)
        assert np.allclose(
            composed.apply(probe),
            t.apply(probe + self.PRE) + self.POST,
            atol=1e-9,
        )

    def test_polynomial_composition_substitutes_coefficients(self):
        """Coefficient substitution, not a refit: `pre` moves `center` and
        `post` lands on the constant monomial, which `_monomial_powers`
        guarantees is the first one.

        (The result agrees with an explicit shift only to round-off, not
        bitwise, because `post` is summed inside the dot product here and
        added afterwards there.)
        """
        _, _, t = _model("polynomial3")
        composed = t.compose_translations(pre=self.PRE, post=self.POST)
        assert np.array_equal(composed.center, t.center - self.PRE)
        assert composed.scale == t.scale
        assert np.array_equal(
            composed.forward[:, 0], t.forward[:, 0] + self.POST
        )
        # every non-constant coefficient is untouched
        assert np.array_equal(composed.forward[:, 1:], t.forward[:, 1:])
        # and the reverse branch is substituted with the opposite signs
        assert np.array_equal(
            composed.reverse_center, t.reverse_center + self.POST
        )
        assert np.array_equal(
            composed.reverse[:, 0], t.reverse[:, 0] - self.PRE
        )

    def test_round_trip_restores_the_original(self, model):
        """decompose/compose in picasso.localize are this primitive applied
        with opposite signs, so they must be mutual inverses."""
        _, _, _, t = model
        probe = np.random.RandomState(4).uniform(50, CHIP - 50, size=(15, 2))
        there = t.compose_translations(pre=self.PRE, post=self.POST)
        back = there.compose_translations(pre=-self.PRE, post=-self.POST)
        assert np.allclose(back.apply(probe), t.apply(probe), atol=1e-9)

    def test_identity_composition_is_a_no_op(self, model):
        _, _, _, t = model
        probe = np.random.RandomState(4).uniform(50, CHIP - 50, size=(15, 2))
        same = t.compose_translations()
        assert np.allclose(same.apply(probe), t.apply(probe), atol=1e-12)

    def test_domain_follows_the_pre_shift(self, model):
        _, _, _, t = model
        composed = t.compose_translations(pre=self.PRE, post=self.POST)
        assert np.allclose(composed.domain, t.domain - self.PRE)

    def test_polynomial_roundtrip_rms_is_placement_invariant(self):
        """Moving a region does not change how well the two branches invert
        each other."""
        _, _, t = _model("polynomial3")
        composed = t.compose_translations(pre=self.PRE, post=self.POST)
        assert composed.roundtrip_rms_px == t.roundtrip_rms_px


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_recovers_rotation_and_scale(self):
        angle = np.radians(1.5)
        scale_x, scale_y = 1.01, 0.98
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        matrix = np.eye(3)
        matrix[:2, :2] = rot @ np.diag([scale_x, scale_y])
        matrix[:2, 2] = [4.0, -6.0]
        dec = transforms.AffineTransform(matrix=matrix).decompose(
            pixelsize=130
        )
        assert dec["rotation_deg"] == pytest.approx(1.5, abs=1e-6)
        assert dec["scale_x"] == pytest.approx(scale_x, abs=1e-9)
        assert dec["scale_y"] == pytest.approx(scale_y, abs=1e-9)
        assert dec["shear_deg"] == pytest.approx(0.0, abs=1e-6)
        assert dec["tx_px"] == pytest.approx(4.0)
        assert dec["ty_px"] == pytest.approx(-6.0)
        assert dec["tx_nm"] == pytest.approx(4.0 * 130)
        assert dec["ty_nm"] == pytest.approx(-6.0 * 130)
        assert dec["mirror"] is False
        assert dec["flip_axis"] is None

    def test_omits_nm_without_pixelsize(self):
        dec = transforms.identity().decompose()
        assert "tx_nm" not in dec and "ty_nm" not in dec

    @pytest.mark.parametrize(
        "axis, linear",
        [
            ("x", np.diag([-1.0, 1.0])),
            ("y", np.diag([1.0, -1.0])),
        ],
    )
    def test_pure_mirror_reads_as_zero_rotation(self, axis, linear):
        """A reflected channel must not report ~180 degrees; the refinement
        rebuilds its seed from `mirror` and `flip_axis`."""
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        dec = transforms.AffineTransform(matrix=matrix).decompose(
            pixelsize=1.0
        )
        assert dec["mirror"] is True
        assert dec["flip_axis"] == axis
        assert dec["rotation_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_non_affine_decomposes_at_the_requested_point(self):
        _, _, t = _model("polynomial3")
        near = t.decompose(pixelsize=1.0, at=(100.0, 100.0))
        far = t.decompose(pixelsize=1.0, at=(CHIP - 100, CHIP - 100))
        assert near["scale_major"] != far["scale_major"]

    def test_defaults_to_the_domain_center(self, model):
        _, _, _, t = model
        center = 0.5 * (t.domain[0] + t.domain[1])
        assert t.decompose() == t.decompose(at=center)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Transforms ride inside spline calibrations (JSON in an HDF5 attribute)
    and inside YAML calibrations written with `yaml.dump`, not `safe_dump`."""

    @pytest.mark.parametrize("encode, decode", [(json.dumps, json.loads)])
    def test_json_round_trip(self, model, encode, decode):
        _, src, _, t = model
        restored = transforms.from_dict(decode(encode(t.to_dict())))
        assert np.array_equal(restored.apply(src), t.apply(src))

    def test_yaml_round_trip(self, model):
        _, src, _, t = model
        restored = transforms.from_dict(yaml.safe_load(yaml.dump(t.to_dict())))
        assert np.array_equal(restored.apply(src), t.apply(src))

    def test_to_dict_holds_builtins_only(self, model):
        """A stray numpy scalar would make `yaml.dump` emit a
        `!!python/object/apply:numpy...` tag that `json.dumps` then rejects."""
        _, _, _, t = model

        def check(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    assert isinstance(k, str), k
                    check(v)
            elif isinstance(value, list):
                for v in value:
                    check(v)
            else:
                assert value is None or type(value) in (
                    bool,
                    int,
                    float,
                    str,
                ), f"{value!r} is {type(value)}"

        check(t.to_dict())
        assert "!!python" not in yaml.dump(t.to_dict())

    def test_from_dict_passes_transforms_through(self, model):
        _, _, _, t = model
        assert transforms.from_dict(t) is t

    def test_from_dict_rejects_an_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown transform model"):
            transforms.from_dict({"model": "pwl"})


# ---------------------------------------------------------------------------
# identity, allclose, is_plausible
# ---------------------------------------------------------------------------


class TestIdentityAndEquality:
    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_identity_is_a_no_op(self, model):
        t = transforms.identity(model)
        probe = _src(10, seed=8)
        assert np.allclose(t.apply(probe), probe, atol=1e-9)
        assert t.is_identity()

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_a_fitted_transform_is_not_the_identity(self, model):
        _, _, t = _model(model)
        assert not t.is_identity()

    def test_allclose_matches_only_the_same_model(self, model):
        _, _, _, t = model
        assert t.allclose(t)
        assert not t.allclose(transforms.identity())

    def test_allclose_rejects_a_different_model(self):
        affine = transforms.identity("affine")
        projective = transforms.identity("projective")
        assert not affine.allclose(projective)


class TestIsPlausible:
    def test_accepts_a_sane_transform(self, model):
        _, _, _, t = model
        assert transforms.is_plausible(t)

    def test_rejects_an_absurd_scale(self):
        big = transforms.AffineTransform(matrix=np.diag([3.0, 3.0, 1.0]))
        assert not transforms.is_plausible(big, domain=[[0, 0], [100, 100]])

    def test_accepts_a_mirror(self):
        """A negative determinant is a reflected channel, not an
        implausible one."""
        matrix = np.eye(3)
        matrix[:2, :2] = np.diag([-1.0, 1.0])
        mirrored = transforms.AffineTransform(matrix=matrix)
        assert transforms.is_plausible(mirrored, domain=[[0, 0], [100, 100]])

    def test_rejects_a_fold_over(self):
        """A flexible model on sparse points can become locally
        non-injective, which the determinant magnitude alone would miss."""
        powers = transforms._monomial_powers(2)
        forward = np.zeros((2, len(powers)))
        forward[0, powers.index((1, 0))] = 1.0
        forward[0, powers.index((2, 0))] = -0.02  # d/dx turns negative at x=25
        forward[1, powers.index((0, 1))] = 1.0
        folded = transforms.PolynomialTransform(
            degree=2,
            forward=forward,
            center=np.zeros(2),
            scale=1.0,
            reverse=forward.copy(),
            reverse_center=np.zeros(2),
            reverse_scale=1.0,
        )
        assert not transforms.is_plausible(folded, domain=[[0, 0], [100, 100]])


# ---------------------------------------------------------------------------
# warp_image
# ---------------------------------------------------------------------------


class TestWarpImage:
    IMAGE = (
        np.random.RandomState(3)
        .uniform(0, 100, size=(64, 80))
        .astype(np.float32)
    )
    MATRIX = np.array(
        [[1.01, -0.02, 3.5], [0.015, 0.99, -2.25], [0.0, 0.0, 1.0]]
    )

    def test_affine_path_is_bit_identical_to_scipy(self):
        """The interactive channel-sum preview goes through here, so the
        existing pixels must not move.

        ``matrix``/``offset`` are built exactly as the replaced
        ``picasso.localize._channel_warp`` built them.
        """
        from scipy.ndimage import affine_transform

        t = transforms.AffineTransform(matrix=self.MATRIX)
        legacy = self.MATRIX[:2, :]
        matrix, offset = legacy[:, :2][::-1, ::-1], legacy[:, 2][::-1]
        origin = np.array([5.0, 7.0])
        expected = affine_transform(
            self.IMAGE,
            matrix,
            offset=matrix @ origin + offset,
            output_shape=(40, 50),
            order=3,
            mode="constant",
            cval=0.0,
            output=np.float32,
        )
        got = transforms.warp_image(
            self.IMAGE, t, (40, 50), origin=(5, 7), dtype=np.float32
        )
        assert np.array_equal(expected, got)

    def test_pull_semantics_axes_are_swapped_not_inverted(self):
        """Output pixel ``(row, col)`` is sampled from the input at
        ``pull(col + x0, row + y0)``.

        The channel-sum preview relies on this: its output grid *is* the
        reference frame, so the reference -> channel transform goes in
        directly, with the axes swapped and **no inversion**.
        """
        t = transforms.AffineTransform(matrix=self.MATRIX)
        # A linear ramp: bilinear resampling reproduces a linear function
        # exactly, so the output value identifies the sampled position.
        rows, cols = np.mgrid[0:64, 0:80]
        image = 3.0 * cols + 5.0 * rows + 7.0
        out = transforms.warp_image(image, t, (64, 80), order=1)
        for row, col in [(3, 7), (11, 2), (40, 60)]:
            x, y = t.apply([[float(col), float(row)]])[0]
            assert out[row, col] == pytest.approx(3.0 * x + 5.0 * y + 7.0)

    def test_general_path_agrees_with_the_affine_path(self):
        """Same map, different resampler: `map_coordinates` on the
        inverse-mapped grid must reproduce `affine_transform`."""
        affine = transforms.AffineTransform(matrix=self.MATRIX)
        # a projective with a zero last row is the same map, but takes the
        # general code path
        projective = transforms.ProjectiveTransform(matrix=self.MATRIX)
        fast = transforms.warp_image(
            self.IMAGE, affine, (40, 50), origin=(5, 7), dtype=np.float32
        )
        general = transforms.warp_image(
            self.IMAGE, projective, (40, 50), origin=(5, 7), dtype=np.float32
        )
        assert np.allclose(fast, general, atol=1e-4)

    def test_identity_warp_returns_the_image(self):
        out = transforms.warp_image(
            self.IMAGE, transforms.identity(), dtype=np.float32
        )
        assert np.allclose(out, self.IMAGE, atol=1e-4)

    def test_row_blocking_does_not_change_the_result(self, monkeypatch):
        projective = transforms.ProjectiveTransform(matrix=self.MATRIX)
        whole = transforms.warp_image(self.IMAGE, projective, (40, 50))
        monkeypatch.setattr(transforms, "_WARP_BLOCK_PIXELS", 100)
        blocked = transforms.warp_image(self.IMAGE, projective, (40, 50))
        assert np.array_equal(whole, blocked)


# ---------------------------------------------------------------------------
# channel_jacobians
# ---------------------------------------------------------------------------


class TestChannelJacobians:
    def test_shape_and_layout(self):
        _, _, projective = _model("projective")
        transform_list = [transforms.identity(), projective]
        ref_xy = _src(25, seed=9)
        J = transforms.channel_jacobians(transform_list, ref_xy)
        assert J.shape == (25, 2, 4)
        assert J.dtype == np.float64
        # channel 0 is the identity reference
        assert np.allclose(J[:, 0, :], [1.0, 0.0, 0.0, 1.0])
        # [a00, a01, a10, a11], row-major over the 2x2
        assert np.allclose(
            J[:, 1, :], projective.jacobian(ref_xy).reshape(-1, 4)
        )

    def test_accepts_serialized_transforms(self):
        _, _, t = _model("polynomial2")
        ref_xy = _src(5, seed=9)
        from_objects = transforms.channel_jacobians([t], ref_xy)
        from_dicts = transforms.channel_jacobians([t.to_dict()], ref_xy)
        assert np.array_equal(from_objects, from_dicts)


# ---------------------------------------------------------------------------
# The translation model
# ---------------------------------------------------------------------------


class TestTranslation:
    """The 2-DOF model: what distinguishes it from an affine that happens to
    have an identity linear part."""

    def test_is_an_affine_so_every_affine_fast_path_takes_it(self):
        """The constant-Jacobian route through the spline kernels and the
        single-call route through warp_image both key off the class."""
        _, _, t = _model("translation")
        assert isinstance(t, transforms.AffineTransform)
        assert t.model == "translation"
        assert t.n_params == 2

    def test_fits_only_the_shift_and_leaves_rotation_in_the_residual(self):
        """A rotation an affine would absorb must show up as residual
        instead - that is the point of constraining the model."""
        src = _src(80, seed=3)
        rotated = _apply_matrix(AFFINE_TRUTH, src)
        t = transforms.estimate(src, rotated, "translation")
        assert np.allclose(t.jacobian(src), np.eye(2))
        resid = rotated - t.apply(src)
        assert np.sqrt(np.mean(np.sum(resid**2, axis=1))) > 1.0
        # ...while the affine, free to rotate, has none
        affine = transforms.estimate(src, rotated, "affine")
        assert np.allclose(affine.apply(src), rotated, atol=1e-8)

    def test_shift_is_the_mean_displacement(self):
        src = _src(40, seed=7)
        rng = np.random.RandomState(11)
        dst = src + [3.5, -2.25] + rng.normal(0.0, 0.5, src.shape)
        t = transforms.estimate(src, dst, "translation")
        assert np.allclose(t.shift, (dst - src).mean(axis=0))
        assert np.allclose(t.shift, t.translation)

    def test_a_single_correspondence_fixes_it(self):
        assert transforms.min_points("translation") == 1
        with pytest.warns(UserWarning):
            t = transforms.estimate(
                [[10.0, 20.0]], [[13.5, 17.75]], "translation"
            )
        assert np.allclose(t.shift, [3.5, -2.25])

    def test_a_non_identity_linear_part_is_rejected(self):
        with pytest.raises(ValueError, match="identity linear part"):
            transforms.TranslationTransform(matrix=AFFINE_TRUTH)

    def test_from_shift_round_trips_through_a_dict(self):
        t = transforms.TranslationTransform.from_shift(
            [3.5, -2.25], domain=[[0.0, 0.0], [CHIP, CHIP]]
        )
        data = t.to_dict()
        assert data["model"] == "translation"
        assert data["shift"] == [3.5, -2.25]
        # the same builtin-only guarantee the other models make
        assert yaml.safe_load(yaml.dump(data)) == data
        assert json.loads(json.dumps(data)) == data
        assert transforms.from_dict(data).allclose(t)

    def test_inverse_and_composition_stay_translations(self):
        """A translation must not silently widen into an affine: downstream
        code reads the model name back off the transform."""
        _, _, t = _model("translation")
        assert t.inverse().model == "translation"
        assert t.compose_translations(pre=(1.0, 2.0)).model == "translation"
        assert transforms.identity("translation").model == "translation"

    def test_never_compares_equal_to_the_matching_affine(self):
        """Same map, different model - a calibration fitted with a constrained
        model is not the same calibration as an unconstrained one."""
        t = transforms.TranslationTransform.from_shift([3.5, -2.25])
        affine = transforms.AffineTransform(matrix=t.matrix.copy())
        assert not t.allclose(affine)
        assert not affine.allclose(t)
