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
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Set, Union

from qiskit import QuantumCircuit
from qiskit.providers import JobError, JobTimeoutError, JobV1
from qiskit.providers.jobstatus import JobStatus
from qiskit.result.models import ExperimentResult, ExperimentResultData
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
        for circuit in self.circuits:
            self._submit_single(circuit, self.shots)

    def status(self) -> JobStatus:
        # update the local job cache
        with ThreadPoolExecutor(thread_name_prefix="status_worker_") as pool:
            futures = [
                pool.submit(self._result_single, job_id) for job_id in self._jobs
            ]
            wait(futures, timeout=10.0)

        return self._aggregate_status()

    def result(self):
        self.wait_for_final_state()  # one of DONE, CANCELLED, ERROR
        agg_status = self._aggregate_status()

        results = []

        # jobs order is submission order
        for circuit, result in zip(self.circuits, self._jobs.values()):
            data = {}

            if isinstance(result, JobFinished):
                meas_map = _build_memory_mapping(circuit)
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
        """The AQT API identifiers of all the jobs for this Qiskit job."""
        return set(self._jobs)

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

    def _result_single(self, job_id: str) -> None:
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
    qu2cl = {}
    qubit_map = {}
    count = 0

    for bit in circuit.qubits:
        qubit_map[bit] = count
        count += 1
    clbit_map = {}
    count = 0
    for bit in circuit.clbits:
        clbit_map[bit] = count
        count += 1
    for instruction in circuit.data:
        if instruction[0].name == "measure":
            for index, qubit in enumerate(instruction[1]):
                qu2cl[qubit_map[qubit]] = clbit_map[instruction[2][index]]
    return qu2cl


def _rearrange_result(shots_results: List[int], qubit_to_bit: Dict[int, int]) -> str:
    length = max(qubit_to_bit.values()) + 1
    bin_output = list("0" * length)
    # convert sample entries to strings and pad the result list with "0" to the number
    # of classical qbits
    bin_input = list("".join([str(bit) for bit in shots_results]).ljust(length, "0"))

    bin_input.reverse()
    for qu, cl in qubit_to_bit.items():
        bin_output[cl] = bin_input[qu]
    return hex(int("".join(bin_output), 2))


def _format_counts(
    samples: List[List[int]], qubit_to_bit: Dict[int, int]
) -> Dict[str, int]:
    counts = {}
    for shots_results in samples:
        h_result = _rearrange_result(shots_results, qubit_to_bit)
        if h_result not in counts:
            counts[h_result] = 1
        else:
            counts[h_result] += 1
    return counts
