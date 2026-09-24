import numpy as np
import pytest
from skbeam.core import roi
from skbeam.core.utils import angle_grid, radial_grid, radius_to_twotheta

from pyCHX import chx_generic_functions
from pyCHX.chx_generic_functions import get_qval_qwid_dict, shift_mask

pytestmark = pytest.mark.portable


def _legacy_shift_mask(new_cen, new_mask, old_cen, old_roi_mask, limit_qnum=None):
    nsx, nsy = new_mask.shape
    down, up = new_cen[0], nsx - new_cen[0]
    left, right = new_cen[1], nsy - new_cen[1]
    x1, x2 = old_cen[0] - down, old_cen[0] + up
    y1, y2 = old_cen[1] - left, old_cen[1] + right
    cropped = old_roi_mask[x1:x2, y1:y2] * new_mask
    shifted = np.zeros_like(cropped)
    labels, _ = roi.extract_label_indices(cropped)
    for index, label in enumerate(np.unique(labels)):
        shifted[cropped == label] = index + 1
    if limit_qnum is not None:
        shifted[shifted > limit_qnum] = 0
    return shifted


def _legacy_qval_qwid_dict(roi_mask, setup_pargs, geometry):
    origin = setup_pargs["center"]
    radial = radial_grid(origin, roi_mask.shape)
    angles = np.degrees(angle_grid(origin, roi_mask.shape))
    two_theta = radius_to_twotheta(setup_pargs["Ldet"], setup_pargs["dpix"] * radial)
    q_map = chx_generic_functions.utils.twotheta_to_q(two_theta, setup_pargs["lambda_"])
    labels, _ = roi.extract_label_indices(roi_mask)
    qval_dict = {}
    qwid_dict = {}
    for index, label in enumerate(np.unique(labels)):
        q_values = q_map[roi_mask == label]
        if geometry == "saxs":
            qval_dict[index] = [(q_values.max() + q_values.min()) / 2]
            qwid_dict[index] = [q_values.max() - q_values.min()]
        elif geometry in {"ang_saxs", "flow_saxs"}:
            if geometry == "ang_saxs":
                angle_values = angles[roi_mask == label]
            else:
                center_row = origin[0]
                angle_values = angles[center_row:][roi_mask[center_row:] == label]
                if len(angle_values) == 0:
                    angle_values = angles[:center_row][roi_mask[:center_row] == label] + 180
            qval_dict[index] = np.zeros(2)
            qwid_dict[index] = np.zeros(2)
            qval_dict[index][0] = (q_values.max() + q_values.min()) / 2
            qwid_dict[index][0] = q_values.max() - q_values.min()
            if ((angle_values.max() * angle_values.min()) < 0) & (angle_values.max() > 90):
                qval_dict[index][1] = (angle_values.max() + angle_values.min()) / 2 - 180
                qwid_dict[index][1] = abs(angle_values.max() - angle_values.min() - 360)
            else:
                qval_dict[index][1] = (angle_values.max() + angle_values.min()) / 2
                qwid_dict[index][1] = abs(angle_values.max() - angle_values.min())
    return qval_dict, qwid_dict


@pytest.mark.parametrize("limit_qnum", [None, 2, 0])
def test_shift_mask_preserves_sparse_label_remapping_and_limit(limit_qnum):
    old_roi_mask = np.zeros((8, 9), dtype=np.int64)
    old_roi_mask[1:4, 1:4] = 1_000_000_000
    old_roi_mask[2:6, 4:6] = 7
    old_roi_mask[5:7, 2:5] = 42
    old_roi_mask[3, 3] = -4
    original = old_roi_mask.copy()
    new_mask = np.ones((5, 6), dtype=np.uint8)
    new_mask[0, :] = 0
    new_mask[:, -1] = 0

    actual = shift_mask(
        new_cen=[2, 3],
        new_mask=new_mask,
        old_cen=[4, 5],
        old_roi_mask=old_roi_mask,
        limit_qnum=limit_qnum,
    )
    expected = _legacy_shift_mask(
        new_cen=[2, 3],
        new_mask=new_mask,
        old_cen=[4, 5],
        old_roi_mask=old_roi_mask,
        limit_qnum=limit_qnum,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(old_roi_mask, original)
    assert actual.dtype == expected.dtype
    assert actual.flags.c_contiguous
    assert actual.flags.writeable


def test_shift_mask_preserves_narrow_output_dtype_and_ascending_order():
    old_roi_mask = np.asarray([[20, 0, 5], [20, 9, 5]], dtype=np.int16)
    new_mask = np.ones_like(old_roi_mask, dtype=bool)

    actual = shift_mask([0, 0], new_mask, [0, 0], old_roi_mask)

    np.testing.assert_array_equal(actual, [[3, 0, 1], [3, 2, 1]])
    assert actual.dtype == np.int16


@pytest.mark.parametrize("geometry", ["saxs", "ang_saxs", "flow_saxs"])
def test_grouped_q_metadata_matches_per_label_reference(geometry):
    roi_mask = np.zeros((12, 14), dtype=np.int64)
    roi_mask[1:4, 1:5] = 100_000
    roi_mask[4:9, 4:10] = 7
    roi_mask[8:11, :4] = 31
    roi_mask[2:5, 11:14] = 900
    setup_pargs = {"center": [6, 7], "dpix": 0.075, "Ldet": 5000.0, "lambda_": 1.0}

    actual = get_qval_qwid_dict(roi_mask, setup_pargs, geometry=geometry)
    expected = _legacy_qval_qwid_dict(roi_mask, setup_pargs, geometry=geometry)

    assert list(actual[0]) == list(expected[0])
    assert list(actual[1]) == list(expected[1])
    for index in actual[0]:
        np.testing.assert_array_equal(actual[0][index], expected[0][index])
        np.testing.assert_array_equal(actual[1][index], expected[1][index])


def test_saxs_q_metadata_does_not_construct_angle_map(monkeypatch):
    roi_mask = np.asarray([[0, 1], [2, 2]])
    setup_pargs = {"center": [1, 1], "dpix": 0.075, "Ldet": 5000.0, "lambda_": 1.0}

    def unexpected_angle_map(*args, **kwargs):
        raise AssertionError("isotropic SAXS does not use angles")

    monkeypatch.setattr(chx_generic_functions, "angle_grid", unexpected_angle_map)

    qval_dict, qwid_dict = get_qval_qwid_dict(roi_mask, setup_pargs, geometry="saxs")

    assert list(qval_dict) == [0, 1]
    assert list(qwid_dict) == [0, 1]


def test_q_metadata_preserves_empty_and_unknown_geometry_behavior():
    setup_pargs = {"center": [1, 1], "dpix": 0.075, "Ldet": 5000.0, "lambda_": 1.0}

    assert get_qval_qwid_dict(np.zeros((3, 3), dtype=int), setup_pargs) == ({}, {})
    assert get_qval_qwid_dict(np.ones((3, 3), dtype=int), setup_pargs, geometry="unknown") == ({}, {})


@pytest.mark.parametrize("center", [[-1, 1], [0.5, 1], [3.5, 1]])
def test_flow_q_metadata_uses_physical_half_plane_for_off_detector_and_fractional_centers(center):
    roi_mask = np.ones((3, 3), dtype=np.int64)
    setup_pargs = {"center": center, "dpix": 0.075, "Ldet": 5000.0, "lambda_": 1.0}
    rows, columns = np.nonzero(roi_mask)
    angles = np.degrees(np.arctan2(rows - center[0], columns - center[1]))
    selected = rows >= center[0]
    expected_angles = angles[selected] if np.any(selected) else angles + 180
    minimum = expected_angles.min()
    maximum = expected_angles.max()
    if minimum * maximum < 0 and maximum > 90:
        expected_center = (maximum + minimum) / 2 - 180
        expected_width = abs(maximum - minimum - 360)
    else:
        expected_center = (maximum + minimum) / 2
        expected_width = abs(maximum - minimum)

    qval_dict, qwid_dict = get_qval_qwid_dict(roi_mask, setup_pargs, geometry="flow_saxs")

    assert qval_dict[0][1] == pytest.approx(expected_center)
    assert qwid_dict[0][1] == pytest.approx(expected_width)
