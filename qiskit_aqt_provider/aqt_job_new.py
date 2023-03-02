# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, Alpine Quantum Technologies GmbH 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=protected-access

import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Set, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import JobV1
from qiskit.providers.jobstatus import JobStatus
from qiskit.result.result import Result

from qiskit_aqt_provider import circuit_to_aqt

if TYPE_CHECKING:
    from qiskit_aqt_provider.aqt_resource import AQTResource


# Tags for the status of AQT API jobs


@dataclass
class JobFinished:
    """The job finished successfully."""

    status: ClassVar = JobStatus.DONE
    samples: List[List[int]]


@dataclass
class JobFailed:
    """An error occurred during the job execution."""

    status: ClassVar = JobStatus.ERROR
    error: str


class JobQueued:
    """The job is queued."""

    status: ClassVar = JobStatus.QUEUED


class JobOngoing:
    """The job is running."""

    status: ClassVar = JobStatus.RUNNING


class JobCancelled:
    """The job was cancelled."""

    status = ClassVar = JobStatus.CANCELLED


class AQTJobNew(JobV1):
    def __init__(
        self,
        backend: "AQTResource",
        circuits: List[QuantumCircuit],
        shots: int,
    ):
        """Initialize a job instance.

        Parameters:
            backend (BaseBackend): Backend that job was executed on.
            circuits (List[QuantumCircuit]): List of circuits to execute.
            shots (int): Number of repetitions per circuit.
        """
        super().__init__(backend, str(uuid.uuid4()))

        self.shots = shots
        self.circuits = circuits

        self._jobs: Dict[
            str, Union[JobFinished, JobFailed, JobQueued, JobOngoing, JobCancelled]
        ] = {}
        self._jobs_lock = threading.Lock()

    def submit(self):
        """Submits a job for execution."""
        # do not parallelize to guarantee that the order is preserved in the _jobs dict
        for circuit in self.circuits:
            self._submit_single(circuit, self.shots)

    def status(self) -> JobStatus:
        """Query the job's status.

        The job status is aggregated from the status of the individual circuits running
        on the AQT resource."""
        # update the local job cache
        with ThreadPoolExecutor(thread_name_prefix="status_worker_") as pool:
            futures = [
                pool.submit(self._status_single, job_id) for job_id in self._jobs
            ]
            wait(futures, timeout=10.0)

        return self._aggregate_status()

    def result(self) -> Result:
        """Block until all circuits have been evaluated and return the combined result.

        In case of error, use `AQTJobNew.failed_jobs` to access the error messages of the
        failed circuit evaluations.

        Returns:
            The combined result of all circuit evaluations.

        Raises:
            RuntimeError: at least one circuit evaluation failed or was cancelled.
        """
        self.wait_for_final_state()  # one of DONE, CANCELLED, ERROR
        agg_status = self._aggregate_status()

        if agg_status is not JobStatus.DONE:
            raise RuntimeError(
                "An error occurred during at least one circuit evaluation."
            )

        results = []

        # jobs order is submission order
        for circuit, result in zip(self.circuits, self._jobs.values()):
            data = {}

            if isinstance(result, JobFinished):
                meas_map = _build_memory_mapping(circuit)

                # TODO: understand which classical register is used for the measurement
                # Take this register's initialization value into account.
                data["counts"] = _format_counts(result.samples, meas_map)

            results.append(
                {
                    "shots": self.shots,
                    "success": result.status is JobStatus.DONE,
                    "status": result.status.value,
                    "data": data,
                    "header": {
                        "memory_slots": circuit.num_clbits,
                        "name": circuit.name,
                        "metadata": circuit.metadata or {},
                    },
                }
            )

        return Result.from_dict(
            {
                "backend_name": self._backend.name,
                "backend_version": self._backend.version,
                "qobj_id": id(self.circuits),
                "job_id": self.job_id(),
                "success": agg_status is JobStatus.DONE,
                "results": results,
            }
        )

    @property
    def job_ids(self) -> Set[str]:
        """The AQT API identifiers of all the circuits evaluated in this Qiskit job."""
        return set(self._jobs)

    @property
    def failed_jobs(self) -> Dict[str, str]:
        """Map of failed job ids to error reports from the API."""
        with self._jobs_lock:
            return {
                job_id: payload.error
                for job_id, payload in self._jobs.items()
                if isinstance(payload, JobFailed)
            }

    def _submit_single(self, circuit: QuantumCircuit, shots: int) -> None:
        """Submit a single quantum circuit for execution on the backend.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to execute
            shots (int): Number of repetitions

        Returns:
            The AQT job identifier.
        """
        payload = circuit_to_aqt.circuit_to_aqt_new(circuit, shots=shots)
        job_id = self._backend.submit(payload)
        with self._jobs_lock:
            self._jobs[job_id] = JobQueued()

    def _status_single(self, job_id: str) -> None:
        """Query the status of a single circuit execution.
        Update the internal life-cycle tracker."""
        payload = self._backend.result(job_id)
        response = payload["response"]

        with self._jobs_lock:
            if response["status"] == "finished":
                self._jobs[job_id] = JobFinished(samples=response["result"])
            elif response["status"] == "error":
                self._jobs[job_id] = JobFailed(error=str(response["result"]))
            elif response["status"] == "queued":
                self._jobs[job_id] = JobQueued()
            elif response["status"] == "ongoing":
                self._jobs[job_id] = JobOngoing()
            else:
                raise RuntimeError(
                    f"API returned unknown job status: {response['status']}."
                )

    def _aggregate_status(self) -> JobStatus:
        """Aggregate the Qiskit job status from the status of the individual circuit evaluations."""

        # aggregate job status from individual circuits
        with self._jobs_lock:
            statuses = [payload.status for payload in self._jobs.values()]

        if any(s is JobStatus.ERROR for s in statuses):
            return JobStatus.ERROR

        if any(s is JobStatus.CANCELLED for s in statuses):
            return JobStatus.CANCELLED

        if any(s is JobStatus.RUNNING for s in statuses):
            return JobStatus.RUNNING

        if all(s is JobStatus.QUEUED for s in statuses):
            return JobStatus.QUEUED

        if all(s is JobStatus.DONE for s in statuses):
            return JobStatus.DONE

        # TODO: check for completeness
        return JobStatus.QUEUED


def _build_memory_mapping(circuit: QuantumCircuit) -> Dict[int, int]:
    """Scan the circuit for measurement instructions and collect qubit to classical bits mappings.

    This assumes that the `QuantumRegister` to `ClassicalRegister` mappings are consistent
    across all measurement operations in the circuit. If this is not the case, the later
    mappings take precedence.

    Parameters:
        circuit: the `QuantumCircuit` to analyze.

    Returns:
        the translation map for all measurement operations in the circuit.

    Examples:
        >>> qc = QuantumCircuit(2)
        >>> qc.measure_all()
        >>> _build_memory_mapping(qc)
        {0: 0, 1: 1}

        >>> qc = QuantumCircuit(2, 2)
        >>> _ = qc.measure([0, 1], [1, 0])
        >>> _build_memory_mapping(qc)
        {0: 1, 1: 0}

        >>> qc = QuantumCircuit(3, 2)
        >>> _ = qc.measure([0, 1], [0, 1])
        >>> _build_memory_mapping(qc)
        {0: 0, 1: 1}

        >>> qc = QuantumCircuit(4, 6)
        >>> _ = qc.measure([0, 1, 2, 3], [2, 3, 4, 5])
        >>> _build_memory_mapping(qc)
        {0: 2, 1: 3, 2: 4, 3: 5}

        >>> qc = QuantumCircuit(3, 4)
        >>> qc.measure_all(add_bits=False)
        >>> _build_memory_mapping(qc)
        {0: 0, 1: 1, 2: 2}

        >>> qc = QuantumCircuit(3, 3)
        >>> _ = qc.x(0)
        >>> _ = qc.measure([0], [2])
        >>> _ = qc.y(1)
        >>> _ = qc.measure([1], [1])
        >>> _ = qc.x(2)
        >>> _ = qc.measure([2], [0])
        >>> _build_memory_mapping(qc)
        {0: 2, 1: 1, 2: 0}
    """
    qu2cl: Dict[int, int] = {}

    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "measure":
            for qubit, clbit in zip(instruction.qubits, instruction.clbits):
                qu2cl[qubit.index] = clbit.index

    return qu2cl


def _shot_to_int(
    fluorescence_states: List[int], qubit_to_bit: Optional[Dict[int, int]] = None
) -> int:
    """Format the detected fluorescence states from a single shot as an integer.

    This follows the Qiskit ordering convention, where bit 0 in the classical register is mapped
    to bit 0 in the returned integer.

    An optional translation map from the quantum to the classical register can be applied.
    If provided, the map must be injective (i.e. map all qubits to classical bits).

    Parameters:
        fluorescence_states: detected fluorescence states for this shot
        qubit_to_bit: optional translation map from quantum register to classical register positions

    Returns:
        integral representation of the shot result, with the translation map applied.

    Examples:
       Without a translation map, the natural mapping is used (n -> n):

        >>> _shot_to_int([1])
        1

        >>> _shot_to_int([0, 0, 1])
        4

        >>> _shot_to_int([0, 1, 1])
        6

        Swap qubits 1 and 2 in the classical register:

        >>> _shot_to_int([0, 0, 1], {0: 0, 1: 2, 2: 1})
        2

        The map cannot be partial:

        >>> _shot_to_int([0, 0, 1], {1: 2, 2: 1})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        One can translate into a classical register larger than the
        qubit register.

        Warning: the classical register is always initialized to 0.

        >>> _shot_to_int([1], {0: 1})
        2

        >>> _shot_to_int([0, 1, 1], {0: 3, 1: 4, 2: 5}) == (0b110 << 3)
        True
    """
    tr_map = qubit_to_bit or {}

    if tr_map:
        if set(tr_map.keys()) != set(range(len(fluorescence_states))):
            raise ValueError("Map must be injective.")

        # allocate a zero-initialized classical register
        creg = [0] * (max(tr_map.values()) + 1)

        for src_index, dest_index in tr_map.items():
            creg[dest_index] = fluorescence_states[src_index]
    else:
        creg = fluorescence_states.copy()

    return (np.left_shift(1, np.arange(len(creg))) * creg).sum()


def _format_counts(
    samples: List[List[int]], qubit_to_bit: Optional[Dict[int, int]] = None
) -> Dict[str, int]:
    """Format all shots results from a circuit evaluation.

    The returned dictionary is compatible with Qiskit's `ExperimentResultData`
    `counts` field.

    Keys are hexadecimal string representations of the detected states, with the
    optional `QuantumRegister` to `ClassicalRegister` applied. Values are the occurrences
    of the keys.

    Parameters:
        samples: detected qubit fluorescence states for all shots
        qubit_to_bit: optional quantum to classical register translation map

    Returns:
        collected counts, for `ExperimentResultData`.

    Examples:
        >>> _format_counts([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        {'0x1': 2, '0x2': 1}

        >>> _format_counts([[1, 0, 0], [0, 1, 0], [1, 0, 0]], {0: 2, 1: 1, 2: 0})
        {'0x4': 2, '0x2': 1}
    """
    return dict(Counter(hex(_shot_to_int(shot, qubit_to_bit)) for shot in samples))
