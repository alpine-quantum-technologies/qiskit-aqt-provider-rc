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
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Type, Union, overload

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import RGate, RXGate, RXXGate, RZGate
from qiskit.circuit.tools import pi_check
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


@overload
def with_qiskit_error(
    run_func: Literal[None], /, exc_types: Tuple[Type[BaseException], ...] = ...
) -> Callable[
    [Callable[[TransformationPass, DAGCircuit], DAGCircuit]],
    Callable[[TransformationPass, DAGCircuit], DAGCircuit],
]:
    ...  # pragma: no cover


@overload
def with_qiskit_error(
    run_func: Callable[[BasePass, DAGCircuit], DAGCircuit],
    /,
    exc_types: Tuple[Type[BaseException], ...] = ...,
) -> Callable[[BasePass, DAGCircuit], DAGCircuit]:
    ...  # pragma: no cover


def with_qiskit_error(
    run_func: Optional[Callable[[BasePass, DAGCircuit], DAGCircuit]] = None,
    /,
    exc_types: Tuple[Type[BaseException], ...] = (Exception,),
) -> Union[
    Callable[
        [Callable[[BasePass, DAGCircuit], DAGCircuit]],
        Callable[[BasePass, DAGCircuit], DAGCircuit],
    ],
    Callable[[BasePass, DAGCircuit], DAGCircuit],
]:
    """Decorator for `BasePass.run` methods to raise `QiskitError` on execution errors.

    Args:
        run_func: method to decorate. If none, act as a decorator factory
        exc_types: which error types to re-raise as `QiskitError`.
    """
    # TODO: use ParamSpec once py310 is the lowest supported version

    def impl(
        func: Callable[[BasePass, DAGCircuit], DAGCircuit]
    ) -> Callable[[BasePass, DAGCircuit], DAGCircuit]:
        def wrapper(self: BasePass, dag: DAGCircuit) -> DAGCircuit:
            try:
                return func(self, dag)
            except exc_types as e:
                raise QiskitError from e

        return wrapper

    if run_func is not None:
        return impl(run_func)

    return impl  # pragma: no cover


def rewrite_rx_as_r(theta: float) -> Instruction:
    """Instruction equivalent to Rx(θ) as R(θ, φ) with θ ∈ [0, π] and φ ∈ [0, 2π]."""
    theta = math.atan2(math.sin(theta), math.cos(theta))
    phi = math.pi if theta < 0.0 else 0.0
    return RGate(abs(theta), phi)


class RewriteRxAsR(TransformationPass):
    """Rewrite Rx(θ) as R(θ, φ) with θ ∈ [0, π] and φ ∈ [0, 2π]."""

    @with_qiskit_error
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.gate_nodes():
            if node.name == "rx":
                (theta,) = node.op.params
                dag.substitute_node(node, rewrite_rx_as_r(float(theta)))
        return dag


class AQTSchedulingPlugin(PassManagerStagePlugin):
    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,  # noqa: ARG002
        optimization_level: Optional[int] = None,  # noqa: ARG002
    ) -> PassManager:
        passes: List[BasePass] = [
            # The Qiskit Target declares RX/RZ as basis gates.
            # This allows decomposing any run of rotations into the ZXZ form, taking
            # advantage of the free Z rotations.
            # Since the API expects R/RZ as single-qubit operations,
            # we rewrite all RX gates as R gates after optimizations have been performed.
            RewriteRxAsR(),
        ]

        return PassManager(passes)


@dataclass(frozen=True)
class CircuitInstruction:
    """Substitute for `qiskit.circuit.CircuitInstruction`.

    Contrary to its Qiskit counterpart, this type allows
    passing the qubits as integers.
    """

    gate: Gate
    qubits: Tuple[int, ...]


def _rxx_positive_angle(theta: float) -> List[CircuitInstruction]:
    """List of instructions equivalent to RXX(θ) with θ >= 0."""
    rxx = CircuitInstruction(RXXGate(abs(theta)), qubits=(0, 1))

    if theta >= 0:
        return [rxx]

    return [
        CircuitInstruction(RZGate(math.pi), (0,)),
        rxx,
        CircuitInstruction(RZGate(math.pi), (0,)),
    ]


def _emit_rxx_instruction(theta: float, instructions: List[CircuitInstruction]) -> Instruction:
    """Collect the passed instructions into a single one labeled 'Rxx(θ)'."""
    qc = QuantumCircuit(2, name=f"Rxx({pi_check(theta)})")
    for instruction in instructions:
        qc.append(instruction.gate, instruction.qubits)

    return qc.to_instruction()


def wrap_rxx_angle(theta: float) -> Instruction:
    """Instruction equivalent to RXX(θ) with θ ∈ [0, π/2]."""
    # fast path if -π/2 <= θ <= π/2
    if abs(theta) <= math.pi / 2:
        operations = _rxx_positive_angle(theta)
        return _emit_rxx_instruction(theta, operations)

    # exploit 2-pi periodicity of Rxx
    theta %= 2 * math.pi

    if abs(theta) <= math.pi / 2:
        operations = _rxx_positive_angle(theta)
    elif abs(theta) <= 3 * math.pi / 2:
        corrected_angle = theta - np.sign(theta) * math.pi
        operations = [
            CircuitInstruction(RXGate(math.pi), (0,)),
            CircuitInstruction(RXGate(math.pi), (1,)),
        ]
        operations.extend(_rxx_positive_angle(corrected_angle))
    else:
        corrected_angle = theta - np.sign(theta) * 2 * math.pi
        operations = _rxx_positive_angle(corrected_angle)

    return _emit_rxx_instruction(theta, operations)


class WrapRxxAngles(TransformationPass):
    """Wrap Rxx angles to [-π/2, π/2]."""

    @with_qiskit_error
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.gate_nodes():
            if node.name == "rxx":
                (theta,) = node.op.params

                if 0 <= float(theta) <= math.pi / 2:
                    continue

                rxx = wrap_rxx_angle(float(theta))
                dag.substitute_node(node, rxx)

        return dag


class AQTTranslationPlugin(PassManagerStagePlugin):
    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: Optional[int] = None,  # noqa: ARG002
    ) -> PassManager:
        translation_pm = common.generate_translation_passmanager(
            target=pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
        )

        passes: List[BasePass] = [WrapRxxAngles()]

        return translation_pm + PassManager(passes)
