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

import numpy.testing as npt
import pytest
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.providers import Provider

from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.transpiler_plugin import arbitrary_rxx_as_xx


def dummy_resource() -> AQTResource:
    """An AQT resource that can only be used as transpiler target."""

    class DummyProvider(Provider):
        portal_url = ""
        access_token = ""

    return AQTResource(
        DummyProvider(),
        workspace="dummy",
        resource={"name": "dummy", "id": "dummy", "type": "simulator"},
    )


@pytest.mark.parametrize(
    "angle,expected_angle",
    [
        (pi / 3, pi / 3),
        (7 * pi / 5, -3 * pi / 5),
        (25 * pi, -pi),
        (22 * pi / 3, -2 * pi / 3),
    ],
)
def test_rx_wrap_angle(angle: float, expected_angle: float) -> None:
    """Check that transpile rotation gate angles are wrapper to [-π,π]."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)

    expected = QuantumCircuit(1)
    expected.r(expected_angle, 0, 0)

    result = transpile(qc, dummy_resource(), optimization_level=3)

    msg = f"\nexpected:\n{expected}\nresult:\n{result}"
    assert expected == result, msg


def test_rx_r_rewrite_simple() -> None:
    """Check that Rx gates are rewritten as R gates."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)

    expected = QuantumCircuit(1)
    expected.r(pi / 2, 0, 0)

    result = transpile(qc, dummy_resource(), optimization_level=3)

    msg = f"\nexpected:\n{expected}\nresult:\n{result}"
    assert expected == result, msg


def test_decompose_1q_rotations_simple() -> None:
    """Check that runs of single-qubit rotations are optimized as a ZXZ."""
    qc = QuantumCircuit(1)
    qc.rx(pi / 2, 0)
    qc.ry(pi / 2, 0)

    expected = QuantumCircuit(1)
    expected.rz(-pi / 2, 0)
    expected.r(pi / 2, 0, 0)

    result = transpile(qc, dummy_resource(), optimization_level=3)

    msg = f"\nexpected:\n{expected}\nresult:\n{result}"
    assert expected == result, msg


def test_decompose_rxx_as_xx_simple() -> None:
    """Check that arbitrary-angle Rxx gates are rewritten in terms of Rxx(π/2) ones."""
    qc = QuantumCircuit(2)
    qc.rxx(pi / 3, 0, 1)

    expected = QuantumCircuit(2)
    expected.r(-pi, 0, 0)
    expected.r(pi / 2, pi / 2, 1)
    expected.r(pi / 2, 0, 1)
    expected.rxx(pi / 2, 0, 1)
    expected.rz(pi / 3, 1)
    expected.rxx(pi / 2, 0, 1)
    expected.r(-pi / 2, pi / 2, 1)
    expected.rz(pi / 2, 1)

    # use optimization level 0 such that no other transformation is done
    result = transpile(qc, dummy_resource(), optimization_level=0)

    msg = f"\nexpected:\n{expected}\nresult:\n{result}"
    assert expected == result, msg


def test_arbitrary_rxx_decomposition() -> None:
    backend = Aer.get_backend("unitary_simulator")

    qr = QuantumRegister(2)
    q0, q1 = qr._bits
    qc = arbitrary_rxx_as_xx(pi / 3, q0, q1)

    job = backend.run(qc)
    u_result = job.result().get_unitary()

    expected = QuantumCircuit(2)
    expected.rxx(pi / 3, 0, 1)

    # the decomposition adds a global phase to the unitary
    expected.global_phase = -pi / 2

    job = backend.run(expected)
    u_expected = job.result().get_unitary()

    npt.assert_allclose(u_result.data, u_expected.data, atol=1e-7)
