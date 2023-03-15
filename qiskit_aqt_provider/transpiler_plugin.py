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

import math
from typing import List, Optional

from qiskit.circuit.library import RGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


class RewriteRxRyAsR(TransformationPass):
    """Rewrite all `RXGate` and `RYGate` instances as `RGate`. Wrap the rotation angle to [-π, π]."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.gate_nodes():
            if node.name in {"rx", "ry"}:
                (theta,) = node.op.params
                phi = math.pi / 2 if node.name == "ry" else 0.0
                new_theta = math.atan2(math.sin(float(theta)), math.cos(float(theta)))
                dag.substitute_node(node, RGate(new_theta, phi))
        return dag


class AQTSchedulingPlugin(PassManagerStagePlugin):
    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: Optional[int] = None
    ) -> PassManager:
        passes: List[BasePass] = [
            # The Qiskit Target declares RX/RZ as basis gates.
            # This allows decomposing any run of rotations into the ZXZ form, taking
            # advantage of the free Z rotations.
            # Since the API expects R/RZ as single-qubit operations,
            # we rewrite all RX/RY gates as R gates after optimizations have been performed.
            RewriteRxRyAsR(),
        ]

        return PassManager(passes)
