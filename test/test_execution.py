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

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class Simulator:
    def __init__(self) -> None:
        pass


def run_on_simulator(circuit: QuantumCircuit, shots: int) -> str:
    backend = AerSimulator(method="statevector")
    job = backend.run(circuit)
