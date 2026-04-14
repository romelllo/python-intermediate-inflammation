"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
import tempfile
import os

from inflammation.models import daily_mean, daily_max, daily_min, load_csv


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


def test_load_csv():
    """Test that load_csv correctly loads a CSV file."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n")
        f.write("4,5,6\n")
        f.write("7,8,9\n")
        temp_file = f.name

    try:
        # Load the CSV file
        result = load_csv(temp_file)
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        npt.assert_array_equal(result, expected)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file)
