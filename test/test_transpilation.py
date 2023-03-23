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
from typing import Final

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from qiskit import QuantumCircuit, transpile

from qiskit_aqt_provider.aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.test.circuits import (
    assert_circuits_equal,
    assert_circuits_equivalent,
    qft_circuit,
)
from qiskit_aqt_provider.transpiler_plugin import wrap_rxx_angle


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

    expected = QuantumCircuit(1)
    expected.r(expected_angle, 0, 0)

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert isinstance(result, QuantumCircuit)

    assert_circuits_equal(result, expected)


def test_rx_r_rewrite_simple(offline_simulator_no_noise: AQTResource) -> None:
    """Check that Rx gates are rewritten as R gates."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)

    expected = QuantumCircuit(1)
    expected.r(pi / 2, 0, 0)

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert isinstance(result, QuantumCircuit)  # only got one circuit back

    assert_circuits_equal(result, expected)


def test_decompose_1q_rotations_simple(offline_simulator_no_noise: AQTResource) -> None:
    """Check that runs of single-qubit rotations are optimized as a ZXZ."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)
    qc.ry(pi / 2, 0)

    expected = QuantumCircuit(1)
    expected.rz(-pi / 2, 0)
    expected.r(pi / 2, 0, 0)

    result = transpile(qc, offline_simulator_no_noise, optimization_level=3)
    assert isinstance(result, QuantumCircuit)  # only got one circuit back

    assert_circuits_equal(result, expected)


def test_rxx_wrap_angle_case0() -> None:
    """Snapshot test for Rxx(θ) rewrite with 0 <= θ <= π/2."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(pi / 2), (0, 1))

    expected = QuantumCircuit(2)
    expected.rxx(pi / 2, 0, 1)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


def test_rxx_wrap_angle_case0_negative() -> None:
    """Snapshot test for Rxx(θ) rewrite with -π/2 <= θ < 0."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(-pi / 2), (0, 1))

    expected = QuantumCircuit(2)
    expected.rz(pi, 0)
    expected.rxx(pi / 2, 0, 1)
    expected.rz(pi, 0)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


def test_rxx_wrap_angle_case1() -> None:
    """Snapshot test for Rxx(θ) rewrite with π/2 < θ <= 3π/2."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(3 * pi / 2), (0, 1))

    expected = QuantumCircuit(2)
    expected.rx(pi, 0)
    expected.rx(pi, 1)
    expected.rxx(pi / 2, 0, 1)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


def test_rxx_wrap_angle_case1_negative() -> None:
    """Snapshot test for Rxx(θ) rewrite with -3π/2 <= θ < -π/2."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(-3 * pi / 2), (0, 1))

    expected = QuantumCircuit(2)
    expected.rxx(pi / 2, 0, 1)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


def test_rxx_wrap_angle_case2() -> None:
    """Snapshot test for Rxx(θ) rewrite with θ > 3*π/2."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(18 * pi / 10), (0, 1))  # mod 2π = 9π/5 → -π/5

    expected = QuantumCircuit(2)
    expected.rz(pi, 0)
    expected.rxx(pi / 5, 0, 1)
    expected.rz(pi, 0)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


def test_rxx_wrap_angle_case2_negative() -> None:
    """Snapshot test for Rxx(θ) rewrite with θ < -3π/2."""
    result = QuantumCircuit(2)
    result.append(wrap_rxx_angle(-18 * pi / 10), (0, 1))  # mod 2π = π/5

    expected = QuantumCircuit(2)
    expected.rxx(pi / 5, 0, 1)

    assert_circuits_equal(result.decompose(), expected)
    assert_circuits_equivalent(result.decompose(), expected)


RXX_ANGLES: Final = [
    pi / 4,
    pi / 2,
    -pi / 2,
    3 * pi / 4,
    -3 * pi / 4,
    15 * pi / 8,
    -15 * pi / 8,
    33 * pi / 16,
    -33 * pi / 16,
]


@given(
    angle=st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1000 * pi,
        max_value=1000 * pi,
    )
)
@pytest.mark.parametrize("qubits", [3])
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
def test_rxx_wrap_angle_transpile(angle: float, qubits: int, optimization_level: int) -> None:
    """Check that Rxx angles are wrapped by the transpiler."""
    assume(abs(angle) > pi / 200)

    qc = QuantumCircuit(qubits)
    qc.rxx(angle, 0, 1)

    # we only need the backend's transpilation target for this test
    backend = AQTProvider("").get_resource("default", "offline_simulator_no_noise")
    trans_qc = transpile(qc, backend, optimization_level=optimization_level)
    assert isinstance(trans_qc, QuantumCircuit)

    assert_circuits_equivalent(trans_qc, qc)

    assert set(trans_qc.count_ops()) <= set(backend.configuration().basis_gates)
    num_rxx = trans_qc.count_ops().get("rxx")

    # in high optimization levels, the gate might be dropped
    assume(num_rxx is not None)
    assert num_rxx == 1

    # check that all Rxx have angles in [0, π/2]
    for operation in trans_qc.data:
        instruction = operation[0]
        if instruction.name == "rxx":
            (theta,) = instruction.params
            assert 0 <= float(theta) <= pi / 2
            break  # there's only one Rxx gate in the circuit
    else:  # pragma: no cover
        pytest.fail("Transpiled circuit contains no Rxx gate.")


@pytest.mark.parametrize("qubits", [1, 5, 10])
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
def test_qft_circuit_transpilation(
    qubits: int, optimization_level: int, offline_simulator_no_noise: AQTResource
) -> None:
    """Transpile a N-qubit QFT circuit for an AQT backend. Check that the angles are properly
    wrapped."""
    qc = qft_circuit(qubits)
    trans_qc = transpile(qc, offline_simulator_no_noise, optimization_level=optimization_level)
    assert isinstance(trans_qc, QuantumCircuit)

    assert set(trans_qc.count_ops()) <= set(offline_simulator_no_noise.configuration().basis_gates)

    for operation in trans_qc.data:
        instruction = operation[0]
        if instruction.name == "rxx":
            (theta,) = instruction.params
            assert 0 <= float(theta) <= pi / 2

        if instruction.name == "r":
            (theta, _) = instruction.params
            assert abs(theta) <= pi

    if optimization_level < 3 and qubits < 6:
        assert_circuits_equivalent(qc, trans_qc)
