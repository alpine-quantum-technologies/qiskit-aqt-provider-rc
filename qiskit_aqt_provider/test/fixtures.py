# This code is part of Qiskit.
#
# (C) Alpine Quantum Technologies GmbH 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pytest fixtures for the AQT Qiskit provider.

This module is exposed as pytest plugin for this project.
"""

import pytest

from qiskit_aqt_provider.aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource


@pytest.fixture(name="offline_simulator_no_noise")
def fixture_offline_simulator_no_noise() -> AQTResource:
    """Noiseless offline simulator resource."""
    provider = AQTProvider("")
    return provider.get_resource("default", "offline_simulator_no_noise")
