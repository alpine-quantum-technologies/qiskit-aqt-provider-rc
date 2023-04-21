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

from typing import Iterator, List, Tuple

import pytest
from qiskit.circuit import QuantumCircuit

from qiskit_aqt_provider.aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import ApiResource, OfflineSimulatorResource
from qiskit_aqt_provider.circuit_to_aqt import circuit_to_aqt


class MockSimulator(OfflineSimulatorResource):
    """Offline simulator that keeps track of the submitted circuits."""

    def __init__(self) -> None:
        super().__init__(
            AQTProvider(""),
            "default",
            ApiResource(name="mock_simulator", id="mock_simulator", type="offline_simulator"),
        )

        self.submit_call_args: List[Tuple[QuantumCircuit, int]] = []

    def submit(self, circuit: QuantumCircuit, shots: int) -> str:
        self.submit_call_args.append((circuit, shots))
        return super().submit(circuit, shots)

    @property
    def submitted_circuits(self) -> List[QuantumCircuit]:
        """Circuits passed to the resource for execution, in submission order."""
        return [circuit for circuit, _ in self.submit_call_args]


@pytest.fixture(name="offline_simulator_no_noise")
def fixture_offline_simulator_no_noise() -> Iterator[MockSimulator]:
    """Noiseless offline simulator resource."""
    resource = MockSimulator()
    yield resource

    # try to convert all circuits that were passed to the simulator
    # to the AQT API JSON format.
    for circuit, shots in resource.submit_call_args:
        try:
            _ = circuit_to_aqt(circuit, shots=shots)
        except Exception:  # pragma: no cover  # noqa: BLE001
            pytest.fail(f"Circuit cannot be converted to the AQT JSON format:\n{circuit}")
