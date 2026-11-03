import numpy as np
import pytest

from gaitlab.pose.skeleton import CANONICAL, J, side_joints, to_canonical


def test_mediapipe_feet_mapped():
    xy = np.arange(33 * 2, dtype=float).reshape(1, 33, 2)
    vis = np.ones((1, 33))
    out, v = to_canonical(xy, vis, "mediapipe33")
    assert out.shape == (1, len(CANONICAL), 2)
    np.testing.assert_array_equal(out[0, J["l_heel"]], xy[0, 29])
    np.testing.assert_array_equal(out[0, J["r_toe"]], xy[0, 32])


def test_halpe_feet_mapped():
    xy = np.arange(26 * 2, dtype=float).reshape(1, 26, 2)
    out, _ = to_canonical(xy, np.ones((1, 26)), "halpe26")
    np.testing.assert_array_equal(out[0, J["r_heel"]], xy[0, 25])
    np.testing.assert_array_equal(out[0, J["l_toe"]], xy[0, 20])


def test_coco_has_no_feet():
    out, v = to_canonical(np.zeros((2, 17, 2)), np.ones((2, 17)), "coco17")
    assert np.isnan(out[:, J["l_heel"]]).all()
    assert (v[:, J["r_toe"]] == 0).all()


def test_side_joints():
    sj = side_joints("l")
    assert sj["heel"] == J["l_heel"]
    with pytest.raises(ValueError):
        side_joints("x")
