"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        # Test with zeros
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        # Test with positive integers
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])),
    ],
)
def test_daily_mean(test_input, expected_result):
    """Test that mean function works correctly."""
    npt.assert_array_equal(daily_mean(test_input), expected_result)


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        # Test with zeros
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        # Test with positive integers
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([5, 6])),
    ],
)
def test_daily_max(test_input, expected_result):
    """Test that max function works correctly."""
    npt.assert_array_equal(daily_max(test_input), expected_result)


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        # Test with zeros
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        # Test with positive integers
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2])),
    ],
)
def test_daily_min(test_input, expected_result):
    """Test that min function works correctly."""
    npt.assert_array_equal(daily_min(test_input), expected_result)
