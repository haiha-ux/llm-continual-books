import pytest

from clb import thermal


@pytest.fixture(autouse=True)
def _fresh_duty_clock():
    thermal._last[0] = None
