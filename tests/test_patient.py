"""Tests for the Patient model."""

import pytest
from inflammation.models import Patient


@pytest.mark.parametrize(
    "name",
    [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
    ],
)
def test_create_patient(name):
    """Test that a patient can be created with a name."""
    p = Patient(name=name)
    assert p.name == name
