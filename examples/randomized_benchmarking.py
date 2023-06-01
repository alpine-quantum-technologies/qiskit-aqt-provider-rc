# This code is part of Qiskit.
#
# (C) Copyright Alpine Quantum Technologies GmbH 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Final

import numpy as np
from qiskit_experiments.library import StandardRB

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource

RANDOM_SEED: Final = 0


def standard_rb_one_qubit(backend: AQTResource) -> None:
    """Run a standard single-qubit randomized benchmarking experiment."""
    lengths = np.arange(1, 800, 200)
    num_samples = 10
    seed = RANDOM_SEED
    qubits = [0]

    exp = StandardRB(qubits, lengths, num_samples=num_samples, seed=seed, backend=backend)
    # The gate errors must be defined for the backend's native gate set:
    exp.analysis.options.gate_error_ratio = {"r": 1.0, "rz": 1.0, "rxx": 1.0}

    data = exp.run().block_for_results()

    # Retrieve the analysis result for the R gate
    result = data.analysis_results("EPG_r")

    # Should be better than the noise model prescribes
    assert result.quality == "good"  # noqa: S101
    assert result.value < 0.003  # noqa: S101

    # For information
    print(result)


if __name__ == "__main__":
    backend = AQTProvider("token").get_backend("offline_simulator_noise")
    standard_rb_one_qubit(backend)
