# Changelog

## Unreleased

## qiskit-aqt-provider v0.16.0

* Make the access token optional #80
* Add simple QAOA examples #81

## qiskit-aqt-provider v0.15.0

* Set default portal url to `https://arnica-stage.aqt.eu` #79

## qiskit-aqt-provider v0.14.0

* Add `AQTEstimator`, a specialized implementation of the `Estimator` primitive #71
* Add simple VQE example #71
* Update pinned dependencies #72
* Add `offline_simulator_noise` resource with basic noise model #73

## qiskit-aqt-provider v0.13.0

* Always raise `TranspilerError` on errors in the custom transpilation passes #57
* Add `AQTSampler`, a specialized implementation of the `Sampler` primitive #60
* Auto-generate and use Pydantic models for the API requests payloads #62
* Use server-side multi-circuits jobs API #63
* Add job completion progress bar #63
* Allow overriding any backend option in `AQTResource.run` #64
* Only return raw memory data when the `memory` option is set #64
* Implement the `ProviderV1` interface for `AQTProvider` #65
* Set User-Agent with package and platform information for HTTP requests #65
* Add py.typed marker file #66
* Rename package to `qiskit-aqt-provider-rc` #67

## qiskit-aqt-provider v0.12.0

* Use `ruff` instead of `pylint` as linter #51
* Publish release artifacts to PyPI #55

## qiskit-aqt-provider v0.11.0

* Expose the result polling period and timeout as backend options #46
* Support `qiskit.result.Result.get_memory()` to retrieve the raw results bitstrings #48

## qiskit-aqt-provider v0.10.0

* Add a Grover-based 3-SAT solver example #31
* Wrap `Rxx` angles to [0, π/2] instead of [-π/2, π/2] #37
* Wrap single-qubit rotation angles to [0, π] instead of [-π, π]  #39
* Remove provider for legacy API #40
* Automatically load environment variables from `.env` files #42

## qiskit-aqt-provider v0.9.0

* Fix and improve error handling from individual circuits #24
* Run the examples in the continuous integration pipeline #26
* Automatically create a Github release when a version tag is pushed #28
* Add `number_of_qubits` to the `quantum_circuit` job payload #29
* Fix the substitution circuit for wrapping the `Rxx` angles #30
* Connect to the internal Arnica on port 80 by default #33

## qiskit-aqt-provider v0.8.1

* Relax the Python version requirement #23

## qiskit-aqt-provider v0.8.0

* Allow the transpiler to decompose any series of single-qubit rotations as ZRZ #13
* Wrap single-qubit rotation angles to [-π, π] #13
* Add `offline_simulator_no_noise` resource (based on Qiskit-Aer simulator) to all workspaces #16
* Add simple execution tests #16
* Use native support for arbitrary-angle RXX gates #19
* Stricter validation of measurement operations #19
* Allow executing circuits with only measurement operations #19

## qiskit-aqt-provider v0.7.0

* Fix quantum/classical registers mapping #10
* Allow jobs with multiple circuits #10
* Use `poetry` for project setup #7

## qiskit-aqt-provider v0.6.1

* Fixes installation on windows #8

## qiskit-aqt-provider v0.6.0

* Initial support for the Arnica API #4
* Setup Mypy typechecker #3
