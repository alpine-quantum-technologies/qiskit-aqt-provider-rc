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

from qiskit_aqt_provider import AQTProvider

if __name__ == "__main__":
    from math import pi

    aqt = AQTProvider("")
    backend = aqt.get_resource("default", "offline_simulator")

    qc = QuantumCircuit(2)
    qc.ry(pi / 2, 1)
    qc.measure_all()

    job = backend.run(qc, shots=200)
    print(job.result().get_counts())
