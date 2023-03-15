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

from math import pi

import pytest
from qiskit import QuantumCircuit, transpile

from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.test.circuits import assert_circuits_equal


@pytest.mark.parametrize(
    "angle,expected_angle",
    [
        (pi / 3, pi / 3),
        (7 * pi / 5, -3 * pi / 5),
        (25 * pi, -pi),
        (22 * pi / 3, -2 * pi / 3),
    ],
)
def test_rx_wrap_angle(
    angle: float, expected_angle: float, offline_simulator_no_noise: AQTResource
) -> None:
    """Check that transpiled rotation gate angles are wrapped to [-π,π]."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    qc.measure_all()

    expected = QuantumCircuit(1)
    expected.r(expected_angle, 0, 0)
    expected.measure_all()

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert_circuits_equal(result, expected)


def test_rx_r_rewrite_simple(offline_simulator_no_noise: AQTResource) -> None:
    """Check that Rx gates are rewritten as R gates."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)
    qc.measure_all()

    expected = QuantumCircuit(1)
    expected.r(pi / 2, 0, 0)
    expected.measure_all()

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert_circuits_equal(result, expected)


def test_decompose_1q_rotations_simple(offline_simulator_no_noise: AQTResource) -> None:
    """Check that runs of single-qubit rotations are optimized as a ZXZ."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)
    qc.ry(pi / 2, 0)
    qc.measure_all()

    expected = QuantumCircuit(1)
    expected.rz(-pi / 2, 0)
    expected.r(pi / 2, 0, 0)
    expected.measure_all()

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert_circuits_equal(result, expected)
