"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
import tempfile
import os
import math

from inflammation.models import (
    daily_mean,
    daily_max,
    daily_min,
    load_csv,
    patient_normalise,
)
from inflammation.compute_data import (
    compute_standard_deviation_by_day,
    analyse_data,
    CSVDataSource,
)


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


@pytest.mark.parametrize(
    "test, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
        )
    ],
)
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
    Test with a relative and absolute tolerance of 0.01."""

    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


def test_analyse_data():
    path = os.path.join(os.getcwd(), "data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected_output = [
        0.0,
        0.22510286,
        0.18157299,
        0.1264423,
        0.9495481,
        0.27118211,
        0.25104719,
        0.22330897,
        0.89680503,
        0.21573875,
        1.24235548,
        0.63042094,
        1.57511696,
        2.18850242,
        0.3729574,
        0.69395538,
        2.52365162,
        0.3179312,
        1.22850657,
        1.63149639,
        2.45861227,
        1.55556052,
        2.8214853,
        0.92117578,
        0.76176979,
        2.18346188,
        0.55368435,
        1.78441632,
        0.26549221,
        1.43938417,
        0.78959769,
        0.64913879,
        1.16078544,
        0.42417995,
        0.36019114,
        0.80801707,
        0.50323031,
        0.47574665,
        0.45197398,
        0.22070227,
    ]
    npt.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "data,expected_output",
    [
        ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
        ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
        ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
    ],
    ids=[
        "Two patients in same file",
        "Two patients in different files",
        "Two identical patients in two different files",
    ],
)
def test_compute_standard_deviation_by_day(data, expected_output):

    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected_output)


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
