"""Capability discovery and in-memory CT task dispatch."""
import importlib.util
import time
from pathlib import Path
import torch

from inv_framework.api import execution
from inv_framework.ct.cases import list_ct_cases
from inv_framework.ct.registry import ALGORITHM_ALIASES, regularizer_records
from .schema import CTTaskRequest, CTTaskResult, normalize_request, load_request_case, request_device


def list_capabilities(*, context=None):
    """Discover CT methods; optional context screens recommendations, not requests."""
    astra_installed = importlib.util.find_spec("astra") is not None
    backends = {
        "torch": {"implemented": True, "dependency_installed": True, "runnable": True,
                  "geometries": ["parallel_2d"], "device": "cpu/cuda", "cuda_available": None},
        "astra": {"implemented": True, "dependency_installed": astra_installed,
                  "runnable": None if astra_installed else False, "geometries": ["parallel_2d", "fan_2d", "parallel_3d", "cone_3d"], "cuda_available": None, "device": "cpu/cuda"},
        "leap": {"implemented": True, "dependency_installed": importlib.util.find_spec("leapctype") is not None,
                 "runnable": importlib.util.find_spec("leapctype") is not None, "geometries": ["parallel_2d"], "reason": "Requires LLNL LEAP"},
    }
    backends['nasn'] = dict(implemented=True, dependency_installed=True,
        runnable=None, geometries=['parallel_2d'], device='cuda',
        reason='Requires CUDA toolkit and a working C++/CUDA extension compiler')
    algorithms = execution.solver_records()
    for record in algorithms:
        name = record["name"]
        from inv_framework.algorithm_catalog import CT_OBJECTIVES, fidelity_contract, parameter_description, resolve_recommendations
        record["fidelity"] = fidelity_contract("ct", name, record["parameter_names"])
        from inv_framework.algorithm_catalog import algorithm_documentation
        record["documentation"] = algorithm_documentation(name)
        record["entry_point"] = "inv_framework.ct.api.run_ct_task"
        _, defaults, _ = execution.load_algorithm_config(
            Path(__file__).resolve().parents[2] / "configs/algorithms" / f"{name}.yaml", name)
        record["objective_definition"] = CT_OBJECTIVES.get(name, record["objective"])
        for parameter in record["parameter_specs"]:
            parameter["default"] = defaults[parameter["name"]]
            declared_description = parameter["description"]
            parameter.update(parameter_description(parameter["name"], parameter["default"], parameter))
            if record["family"] == "deep":
                parameter["description"] = declared_description
                if parameter["name"] == "scale":
                    parameter["units"] = "guidance strength"
        resolve_recommendations("ct", record, context)
        backend = "astra" if name == "fdk" else "torch"
        record.update(backends=["astra"] if name == "fdk" else ["torch", "astra", "leap", "nasn"], default_parameters=defaults, implemented=record["build_reachable"],
                      execution_backend=backend, dependency_installed=backends[backend]["dependency_installed"],
                      runnable=None if backends[backend]["dependency_installed"] else False)
        from inv_framework.algorithm_catalog import describe_execution
        describe_execution("ct", record, context)
        selected = (context or {}).get('backend', backend)
        geometry = (context or {}).get('model', {}).get('geometry', {})
        status = record['runtime_status']
        record['execution_backend'] = selected
        if selected in backends:
            record['backend_capabilities'] = backends[selected]
            if geometry.get('type') and geometry['type'] not in backends[selected]['geometries']:
                status['reasons'].append(f"backend={selected} does not support {geometry['type']}")
            if (geometry.get('type') in {'fan_2d', 'parallel_3d', 'cone_3d'} or selected == 'nasn') and (context or {}).get('device', '').startswith('cpu'):
                status['reasons'].append('Selected geometry/backend requires CUDA')
            if selected == 'nasn':
                status['unresolved_conditions'].append(backends['nasn']['reason'])
        if status['reasons']:
            status['state'] = 'unavailable'
            record['runnable'] = False
        from inv_framework.solvers.deep.registry import ALGORITHMS
        if name in ALGORITHMS:
            from inv_framework.ct.deep import describe_execution as describe_deep
            describe_deep(record, context)
    fidelities = {}
    for record in algorithms:
        for fidelity, requirements in record["metadata"].get("fidelity_requirements", {}).items():
            fidelities.setdefault(fidelity, {})[record["name"]] = requirements
    return {"modality": "ct", "fidelities": fidelities, "algorithms": algorithms, "aliases": dict(ALGORITHM_ALIASES),
            "regularizers": regularizer_records(), "backends": backends,
            "default_backend": "torch",
            "deep_priors": {"entry_point": "inv_framework.ct.api.run_ct_task",
                            "resources": "docs/ct/deep_priors.md"}}


def run_ct_task(request: CTTaskRequest | dict) -> CTTaskResult:
    """Return expected request/backend/numerical failures; propagate implementation errors."""
    started = time.perf_counter()
    case_id = request.get("case_id", "") if isinstance(request, dict) else getattr(request, "case_id", "")
    summary = request.get("algorithm", "") if isinstance(request, dict) else getattr(request, "algorithm", "")
    try:
        request, name, parameters = normalize_request(request)
        summary = name
        if "data_fidelity" in parameters or "regularizer" in parameters:
            summary += f"(fidelity={parameters.get('data_fidelity', 'l2')}, reg={parameters.get('regularizer', 'none')})"
        device = request_device(request, name)
        input_case = load_request_case(request, device) if request.inputs is not None else None
        if input_case is not None:
            geometry = input_case.geometry["type"]
        else:
            cases = {row["case_id"]: row for row in list_ct_cases(data_root=request.data_root)}
            if request.case_id not in cases:
                raise execution.ConfigError(f"unknown CT case: {request.case_id}")
            geometry = cases[request.case_id]["geometry_type"]
        if request.backend == "leap" and importlib.util.find_spec("leapctype") is None:
            raise execution.BackendUnavailable("LLNL LEAP is not installed")
        if geometry == "cone_3d" and request.backend != "astra":
            raise execution.ConfigError("cone_3d requires backend=astra")
        device = request.device or ("cuda" if geometry in {"cone_3d", "parallel_3d", "fan_2d"} or request.backend == "nasn" else "cpu")
        if request.backend == "astra":
            if importlib.util.find_spec("astra") is None:
                raise execution.BackendUnavailable("astra-toolbox is not installed")
            if geometry == "cone_3d" or torch.device(device).type == "cuda":
                execution.require_astra_cuda(device)
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise execution.BackendUnavailable("PyTorch CUDA unavailable")
        result = execution.execute_case(name, request.case_id, parameters,
            device=device, data_root=request.data_root, backend=request.backend,
            fit_view_indices=request.fit_view_indices, heldout_view_indices=request.heldout_view_indices,
            stopping_policy=request.stopping_policy, problem_constraints=request.problem_constraints,
            callback=request.callback, input_case=input_case)
        diagnostics = result.get("diagnostics", {})
        effective_parameters = dict(diagnostics.get("parameters", parameters))
        native = diagnostics.get("convergence", {}).get("metadata", {})
        if name == 'cg' and parameters.get('lam_rel') is not None:
            effective_parameters.update({key: value for key, value in native.get('parameters', {}).items()
                                         if key in {'strength', 'lam_rel', 'operator_norm_squared'}})
        for key in ("step_size", "tau", "sigma", "rho"):
            if key in effective_parameters and native.get(key) is not None:
                effective_parameters[key] = native[key]
        diagnostics["effective_parameters"] = effective_parameters
        status = result.get("convergence_status")
        success = result["status"] == "success" and status not in {"cancelled", "resource_exhausted", "diverged", "numerical_error", "invalid_parameters"}
        bundle = result.get("bundle")
        metrics = {key: float(value) for key, value in result.get("metrics", {}).items()
                   if key in {"psnr", "ssim", "rmse", "relative_error", "data_residual", "measurement_norm", "held_out_projection_residual", "held_out_measurement_norm"}}
        return CTTaskResult("success" if success else "failed", request.case_id, summary, metrics,
                            time.perf_counter() - started, int(diagnostics.get("iterations_completed", 0)),
                            success and status == "converged",
                            bundle["reconstruction"].numpy() if bundle is not None else (result["partial_reconstruction"].numpy() if result.get("partial_reconstruction") is not None else None),
                            None if success else str(result.get("message", result.get("stopping_reason", status))),
                            convergence_status=status,
                            stopping_reason=diagnostics.get("stopping_reason", result.get("stopping_reason", status)),
                            reason_code=diagnostics.get("reason_code", status),
                            history=diagnostics.get("convergence", {}).get("trajectory", []),
                            effective_params=effective_parameters,
                            diagnostics=diagnostics)
    except (execution.ConfigError, execution.BackendUnavailable, execution.NumericalFailure, FileNotFoundError, ValueError, torch.cuda.OutOfMemoryError) as error:
        status = "unavailable" if isinstance(error, execution.BackendUnavailable) else "resource_exhausted" if isinstance(error, torch.cuda.OutOfMemoryError) else "numerical_error" if isinstance(error, execution.NumericalFailure) else "invalid_parameters"
        return CTTaskResult("failed", str(case_id), str(summary), {}, time.perf_counter() - started, 0, False,
                            error_message=str(error), convergence_status=status, stopping_reason=str(error), reason_code=status)
