#!/usr/bin/env python3
"""A single-file MCP server for the pip-installed DEVSIM simulator.

The server keeps each simulation in a separate worker process.  The worker owns
the DEVSIM global state, while this process owns the MCP protocol and job
handles.  Large values are written to the session artifact directory instead
of being sent inline through MCP.

Install the runtime dependencies with::

    pip install devsim mcp

Run with::

    python devsim_mcp.py

The default transport is MCP stdio.  ``--transport streamable-http`` is also
available for clients that support it.
"""

from __future__ import annotations

import argparse
import asyncio
import array
import contextlib
import copy
import hashlib
import io
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP < 2.0 compatibility
    from mcp.server.fastmcp import FastMCP as MCPServer


SERVER_VERSION = "0.1.0"
TERMINAL_JOB_STATES = {"completed", "failed", "cancelled"}
READ_ONLY_COMMANDS = {
    "get_circuit_equation_number",
    "get_circuit_node_list",
    "get_circuit_node_value",
    "get_circuit_solution_list",
    "get_contact_charge",
    "get_contact_current",
    "get_contact_equation_command",
    "get_contact_equation_list",
    "get_contact_list",
    "get_device_list",
    "get_dimension",
    "get_edge_model_list",
    "get_edge_model_values",
    "get_element_model_list",
    "get_element_model_values",
    "get_element_node_list",
    "get_equation_command",
    "get_equation_list",
    "get_equation_numbers",
    "get_interface_equation_command",
    "get_interface_equation_list",
    "get_interface_list",
    "get_interface_model_list",
    "get_interface_model_values",
    "get_material",
    "get_matrix_and_rhs",
    "get_mesh_list",
    "get_node_model_list",
    "get_node_model_values",
    "get_parameter",
    "get_parameter_list",
    "get_region_list",
}
MUTATING_COMMANDS = {
    "add_1d_contact",
    "add_1d_interface",
    "add_1d_mesh_line",
    "add_1d_region",
    "add_2d_contact",
    "add_2d_interface",
    "add_2d_mesh_line",
    "add_2d_region",
    "add_circuit_node",
    "circuit_alter",
    "circuit_element",
    "circuit_node_alias",
    "contact_edge_model",
    "contact_equation",
    "contact_node_model",
    "create_1d_mesh",
    "create_2d_mesh",
    "create_contact_from_interface",
    "create_device",
    "create_gmsh_mesh",
    "create_interface_from_nodes",
    "custom_equation",
    "delete_circuit",
    "delete_contact_equation",
    "delete_device",
    "delete_edge_model",
    "delete_element_model",
    "delete_equation",
    "delete_interface_equation",
    "delete_interface_model",
    "delete_mesh",
    "delete_node_model",
    "edge_average_model",
    "edge_from_node_model",
    "edge_model",
    "edge_solution",
    "element_from_edge_model",
    "element_from_node_model",
    "element_model",
    "element_pair_from_edge_model",
    "element_solution",
    "equation",
    "finalize_mesh",
    "interface_equation",
    "interface_model",
    "interface_normal_model",
    "load_devices",
    "node_model",
    "node_solution",
    "register_function",
    "reset_devsim",
    "set_circuit_node_value",
    "set_edge_values",
    "set_element_values",
    "set_initial_condition",
    "set_material",
    "set_node_value",
    "set_node_values",
    "set_parameter",
    "solve",
    "vector_element_model",
    "vector_gradient",
    "write_devices",
}


class ServerError(RuntimeError):
    """An error that is safe to return as a structured MCP result."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value) or math.isnan(value) or math.isinf(value)
    if isinstance(value, (list, tuple)):
        return all(_json_safe(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def _normalise(value: Any) -> Any:
    """Convert DEVSIM/Python values into JSON-safe small values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"nonfinite": str(value).lower()}
        return value
    if isinstance(value, complex):
        return {"real": _normalise(value.real), "imag": _normalise(value.imag)}
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, array.array):
        return [_normalise(v) for v in value]
    if hasattr(value, "tolist"):
        return _normalise(value.tolist())
    return str(value)


def _import_system_devsim() -> Any:
    """Import the installed DEVSIM package, never a sibling source tree."""
    module = importlib.import_module("devsim")
    origin_text = getattr(module, "__file__", None)
    script_dir = Path(__file__).resolve().parent
    if origin_text:
        origin = Path(origin_text).resolve()
        if origin == script_dir or script_dir in origin.parents:
            sys.modules.pop("devsim", None)
            sys.path[:] = [entry for entry in sys.path if Path(entry or os.curdir).resolve() != script_dir]
            importlib.invalidate_caches()
            module = importlib.import_module("devsim")
            origin_text = getattr(module, "__file__", None)
            if origin_text:
                origin = Path(origin_text).resolve()
                if origin == script_dir or script_dir in origin.parents:
                    raise ServerError("The DEVSIM import resolves to the MCP script directory; install DEVSIM in the active Python environment")
    return module


def _summary(values: Any) -> dict[str, Any]:
    """Return numeric summary information without exposing all field values."""
    if isinstance(values, array.array):
        values = list(values)
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return {"count": 1, "value": _normalise(values)}
    flat: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            flat.append(float(item))

    visit(values)
    result: dict[str, Any] = {"count": len(values)}
    if flat:
        result.update({"min": min(flat), "max": max(flat), "mean": sum(flat) / len(flat)})
    return result


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)
        (self.root / "checkpoints").mkdir(exist_ok=True)
        (self.root / "exports").mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            self._write_manifest([])

    def _read_manifest(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _write_manifest(self, entries: list[dict[str, Any]]) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _record(self, path: Path, kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        path = path.resolve()
        if not path.is_file():
            raise ServerError(f"Artifact was not created: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {
            "id": _new_id("artifact"),
            "kind": kind,
            "path": str(path.relative_to(self.root)),
            "size": path.stat().st_size,
            "sha256": digest,
            "created_at": _now(),
        }
        if metadata:
            entry["metadata"] = metadata
        entries = self._read_manifest()
        entries.append(entry)
        self._write_manifest(entries)
        return entry

    def safe_output(self, requested: str | None, default_name: str, directory: str = "results") -> Path:
        requested = requested or default_name
        candidate = Path(requested)
        if candidate.is_absolute():
            candidate = Path(candidate.name)
        candidate = (self.root / directory / candidate).resolve()
        directory_path = (self.root / directory).resolve()
        if directory_path not in candidate.parents and candidate != directory_path:
            raise ServerError("Output path must remain inside the session artifact directory")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def capture(self, path: Path, kind: str = "file") -> list[dict[str, Any]]:
        path = path.resolve()
        candidates = [path] if path.is_file() else sorted(path.parent.glob(path.name + "*"))
        return [self._record(item, kind) for item in candidates if item.is_file()]

    def write_json(self, name: str, value: Any, kind: str = "json") -> dict[str, Any]:
        path = self.safe_output(name, _new_id("data") + ".json")
        path.write_text(json.dumps(_normalise(value), indent=2, ensure_ascii=False), encoding="utf-8")
        return self._record(path, kind, {"mime_type": "application/json"})

    def write_text(self, name: str, text: str, kind: str = "text") -> dict[str, Any]:
        path = self.safe_output(name, _new_id("data") + ".txt", directory="exports")
        path.write_text(text, encoding="utf-8")
        return self._record(path, kind, {"mime_type": "text/plain"})

    def write_large(self, name: str, value: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write arrays as NPZ where NumPy is available, otherwise JSON."""
        path = self.safe_output(name, _new_id("array") + ".npz")
        try:
            import numpy as np

            np.savez_compressed(path, values=np.asarray(value))
            meta = {"format": "npz", "dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape)}
        except Exception:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(_normalise(value)), encoding="utf-8")
            meta = {"format": "json"}
        if metadata:
            meta.update(metadata)
        return self._record(path, "array", meta)

    def encode_value(self, value: Any, label: str = "value", inline_limit: int = 512) -> Any:
        if isinstance(value, array.array):
            if len(value) > inline_limit:
                return {"artifact": self.write_large(f"{label}.npz", value), **_summary(value)}
            return _normalise(value)
        if hasattr(value, "shape") and hasattr(value, "tolist"):
            size = int(getattr(value, "size", 0))
            if size > inline_limit:
                return {"artifact": self.write_large(f"{label}.npz", value), **_summary(value)}
            return _normalise(value)
        if isinstance(value, dict):
            return {str(k): self.encode_value(v, f"{label}_{k}", inline_limit) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) > inline_limit and all(isinstance(x, (int, float, complex)) for x in value):
                return {"artifact": self.write_large(f"{label}.npz", value), **_summary(value)}
            return [self.encode_value(v, f"{label}_{i}", inline_limit) for i, v in enumerate(value)]
        return _normalise(value)

    def list(self) -> list[dict[str, Any]]:
        return self._read_manifest()

    def read(self, artifact_id: str) -> dict[str, Any]:
        for entry in self._read_manifest():
            if entry.get("id") == artifact_id:
                path = (self.root / entry["path"]).resolve()
                if not path.is_file():
                    raise ServerError("Artifact file no longer exists")
                result = copy.deepcopy(entry)
                result["content"] = path.read_text(encoding="utf-8") if path.suffix in {".json", ".txt", ".py", ".csv"} else None
                return result
        raise ServerError(f"Unknown artifact: {artifact_id}")


def _capture_stdout() -> tuple[io.StringIO, contextlib.AbstractContextManager[Any]]:
    stream = io.StringIO()
    return stream, contextlib.redirect_stdout(stream)


class WorkerState:
    def __init__(self, artifact_root: str):
        self.ds = _import_system_devsim()
        self.store = ArtifactStore(Path(artifact_root))
        self.lock = threading.RLock()
        self.circuit_configured = False
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.active_job: str | None = None
        self.operation_log: list[dict[str, Any]] = []
        self.recipe_path = self.store.root / "operation_log.jsonl"

    def _record(self, command: str, arguments: dict[str, Any], include_models: list[str] | None = None) -> None:
        item: dict[str, Any] = {"command": command, "arguments": _normalise(arguments), "recorded_at": _now()}
        if include_models is not None:
            item["include_models"] = list(include_models)
        self.operation_log.append(item)
        with self.recipe_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _input_file(self, value: str) -> str:
        path = Path(value).expanduser().resolve()
        allowed = os.environ.get("DEVSIM_MCP_ALLOW_PATHS", "")
        own_artifact = path == self.store.root or self.store.root in path.parents
        if allowed and not own_artifact:
            roots = [Path(item).expanduser().resolve() for item in allowed.split(os.pathsep) if item]
            if not any(path == root or root in path.parents for root in roots):
                raise ServerError("Input path is outside DEVSIM_MCP_ALLOW_PATHS")
        if not path.is_file():
            raise ServerError(f"Input file does not exist: {path}")
        return str(path)

    def _native(
        self,
        command: str,
        arguments: dict[str, Any],
        record: bool = True,
        include_models: list[str] | None = None,
        artifact_kind: str = "devsim_export",
    ) -> Any:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", command) or command.startswith("_"):
            raise ServerError("Invalid native command name")
        function = getattr(self.ds, command, None)
        if not callable(function):
            raise ServerError(f"DEVSIM command is not available: {command}")
        if not _json_safe(arguments):
            raise ServerError("native_call accepts JSON values only; Python callables are not allowed")
        arguments = copy.deepcopy(arguments)
        if command == "load_devices" or command == "create_gmsh_mesh":
            if "file" in arguments:
                arguments["file"] = self._input_file(str(arguments["file"]))
        if command == "write_devices":
            requested = Path(str(arguments.get("file", ""))).expanduser()
            if requested.is_absolute() and (self.store.root == requested.resolve() or self.store.root in requested.resolve().parents):
                output = requested.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
            else:
                output = self.store.safe_output(arguments.get("file"), _new_id("device"), directory="results")
            arguments["file"] = str(output)
        if command == "get_matrix_and_rhs" and not arguments.get("format"):
            arguments["format"] = "csr"
        if record and command not in READ_ONLY_COMMANDS:
            self._record(command, arguments, include_models)
        stdout, redirect = _capture_stdout()
        with redirect:
            result = function(**arguments)
        logs = stdout.getvalue()
        if command in {"add_circuit_node", "circuit_element", "circuit_node_alias", "circuit_alter", "set_circuit_node_value"}:
            self.circuit_configured = True
        elif command in {"delete_circuit", "reset_devsim", "load_devices"}:
            self.circuit_configured = False
        if command == "write_devices":
            artifacts = self.store.capture(Path(arguments["file"]), artifact_kind)
            return {"value": self.store.encode_value(result), "logs": logs, "artifacts": artifacts}
        if logs:
            return {"value": self.store.encode_value(result), "logs": logs}
        return self.store.encode_value(result)

    def _mesh_define(self, config: dict[str, Any]) -> dict[str, Any]:
        mesh_name = str(config.get("mesh", config.get("name", "mesh")))
        dimension = int(config.get("dimension", 1))
        if dimension not in (1, 2):
            raise ServerError("Structured mesh creation supports dimensions 1 and 2; use mesh_import_gmsh for 3D")

        def need(item: dict[str, Any], keys: tuple[str, ...], what: str) -> None:
            missing = [key for key in keys if key not in item]
            if missing:
                raise ServerError(f"{what} needs keys: {', '.join(missing)}")

        # Validate before creating anything so a typo cannot leave an empty
        # mesh behind (DEVSIM has no transactional mesh builder).
        line_keys = ("pos", "ps")
        region_keys = ("tag1", "tag2", "region", "material") if dimension == 1 else ("region", "material")
        contact_keys = ("material", "name", "tag") if dimension == 1 else ("name", "material", "region")
        interface_keys = ("tag", "name") if dimension == 1 else ("name", "region0", "region1")
        for line in config.get("lines", []):
            need(line, line_keys, f"{dimension}D mesh line")
            for spacing_key in ("ps", "ns"):
                if spacing_key in line and float(line[spacing_key]) <= 0:
                    raise ServerError(f"{dimension}D mesh line {spacing_key} must be greater than 0")
        for region in config.get("regions", []):
            need(region, region_keys, f"{dimension}D region")
        for contact in config.get("contacts", []):
            need(contact, contact_keys, f"{dimension}D contact")
        for interface in config.get("interfaces", []):
            need(interface, interface_keys, f"{dimension}D interface")

        self._native(f"create_{dimension}d_mesh", {"mesh": mesh_name})
        try:
            if dimension == 1:
                for line in config.get("lines", []):
                    args = {"mesh": mesh_name, "pos": line["pos"], "ps": line["ps"]}
                    for key in ("tag", "ns"):
                        if key in line:
                            args[key] = line[key]
                    self._native("add_1d_mesh_line", args)
                for region in config.get("regions", []):
                    self._native("add_1d_region", {"mesh": mesh_name, **{k: region[k] for k in ("tag1", "tag2", "region", "material")}})
                for contact in config.get("contacts", []):
                    self._native("add_1d_contact", {"mesh": mesh_name, **{k: contact[k] for k in ("material", "name", "tag")}})
                for interface in config.get("interfaces", []):
                    self._native("add_1d_interface", {"mesh": mesh_name, **{k: interface[k] for k in ("tag", "name")}})
            else:
                for line in config.get("lines", []):
                    args = {"mesh": mesh_name, "pos": line["pos"], "ps": line["ps"], "ns": line.get("ns", line["ps"])}
                    if "dir" in line:
                        args["dir"] = line["dir"]
                    self._native("add_2d_mesh_line", args)
                for region in config.get("regions", []):
                    args = {"mesh": mesh_name, "region": region["region"], "material": region["material"]}
                    for key in ("xl", "xh", "yl", "yh", "bloat"):
                        if key in region:
                            args[key] = region[key]
                    self._native("add_2d_region", args)
                for contact in config.get("contacts", []):
                    args = {"mesh": mesh_name, "name": contact["name"], "material": contact["material"], "region": contact["region"]}
                    for key in ("xl", "xh", "yl", "yh", "bloat"):
                        if key in contact:
                            args[key] = contact[key]
                    self._native("add_2d_contact", args)
                for interface in config.get("interfaces", []):
                    args = {"mesh": mesh_name, "name": interface["name"], "region0": interface["region0"], "region1": interface["region1"]}
                    for key in ("xl", "xh", "yl", "yh", "bloat"):
                        if key in interface:
                            args[key] = interface[key]
                    self._native("add_2d_interface", args)
        except Exception:
            # A partially defined mesh would otherwise survive the failure and
            # block every retry with 'A mesh already exists with name ...'.
            try:
                self._native("delete_mesh", {"mesh": mesh_name}, record=False)
            except Exception:
                pass
            raise
        return {"mesh": mesh_name, "dimension": dimension}

    def _gmsh_import(self, config: dict[str, Any]) -> dict[str, Any]:
        mesh_name = str(config.get("mesh", "gmsh_mesh"))
        self._native("create_gmsh_mesh", {"mesh": mesh_name, "file": self._input_file(str(config["file"]))})
        for item in config.get("regions", []):
            self._native("add_gmsh_region", {"mesh": mesh_name, "gmsh_name": item["gmsh_name"], "region": item["region"], "material": item["material"]})
        for item in config.get("contacts", []):
            self._native("add_gmsh_contact", {"mesh": mesh_name, "gmsh_name": item["gmsh_name"], "name": item["name"], "region": item["region"], "material": item["material"]})
        for item in config.get("interfaces", []):
            self._native("add_gmsh_interface", {"mesh": mesh_name, "gmsh_name": item["gmsh_name"], "name": item["name"], "region0": item["region0"], "region1": item["region1"]})
        return {"mesh": mesh_name, "file": config["file"]}

    def _physics_apply(self, args: dict[str, Any]) -> dict[str, Any]:
        import devsim.python_packages.simple_physics as physics

        preset = args.get("preset", "silicon_drift_diffusion")
        device = str(args["device"])
        regions = list(args.get("regions", []))
        contacts = list(args.get("contacts", []))
        interfaces = list(args.get("interfaces", []))
        temperature = float(args.get("temperature", 300.0))
        region = str(args.get("region", regions[0] if regions else ""))
        if preset not in {"silicon_potential_only", "silicon_drift_diffusion", "oxide_potential_only"}:
            raise ServerError("Unknown physics preset")
        if not regions:
            raise ServerError("physics_apply needs at least one region")
        if contacts and not region:
            raise ServerError("physics_apply needs a region when contacts are given")

        created_solutions: list[str] = []

        def ensure_solution(region_name: str, name: str) -> bool:
            """Create a node solution and its edge models when it is missing."""
            if name in self.ds.get_node_model_list(device=device, region=region_name):
                return False
            physics.CreateSolution(device, region_name, name)
            created_solutions.append(f"{region_name}:{name}")
            return True

        for region_name in regions:
            if preset.startswith("silicon"):
                # SetSiliconParameters configures the full constant set,
                # including n1/p1/taun/taup that the SRH model needs.
                physics.SetSiliconParameters(device, region_name, temperature)
                ensure_solution(region_name, "Potential")
                physics.CreateSiliconPotentialOnly(device, region_name)
                if preset == "silicon_drift_diffusion":
                    electrons_created = ensure_solution(region_name, "Electrons")
                    holes_created = ensure_solution(region_name, "Holes")
                    if electrons_created or holes_created:
                        # Seed the carriers from equilibrium so the first
                        # drift-diffusion solve has a sane initial guess.
                        self.ds.set_node_values(
                            device=device, region=region_name, name="Electrons", init_from="IntrinsicElectrons"
                        )
                        self.ds.set_node_values(
                            device=device, region=region_name, name="Holes", init_from="IntrinsicHoles"
                        )
                    physics.CreateSiliconDriftDiffusion(device, region_name)
            else:
                physics.SetOxideParameters(device, region_name, temperature)
                physics.CreateOxidePotentialOnly(device, region_name)

        for contact in contacts:
            bias_name = f"{contact}_bias"
            # Contact equations read <contact>_bias; default it to 0 V unless
            # the caller already applied a bias.
            try:
                self.ds.get_parameter(device=device, name=bias_name)
            except Exception:
                self.ds.set_parameter(device=device, name=bias_name, value=0.0)
            if preset == "silicon_potential_only":
                physics.CreateSiliconPotentialOnlyContact(device, region, contact)
            elif preset == "silicon_drift_diffusion":
                # The potential boundary condition is a prerequisite for the
                # drift-diffusion contact model.
                physics.CreateSiliconPotentialOnlyContact(device, region, contact)
                physics.CreateSiliconDriftDiffusionAtContact(device, region, contact)
            else:
                physics.CreateOxideContact(device, region, contact)

        for interface in interfaces:
            if preset == "silicon_drift_diffusion":
                physics.CreateSiliconSiliconInterface(device, interface)
            else:
                physics.CreateSiliconOxideInterface(device, interface)
        self._record("physics_apply", {k: v for k, v in args.items() if k not in {"session_id"}})
        return {
            "preset": preset,
            "device": device,
            "regions": regions,
            "contacts": contacts,
            "interfaces": interfaces,
            "created_solutions": created_solutions,
        }

    def _field_get(self, args: dict[str, Any]) -> dict[str, Any]:
        location = str(args.get("location", "node"))
        command = {"node": "get_node_model_values", "edge": "get_edge_model_values", "element": "get_element_model_values", "interface": "get_interface_model_values"}.get(location)
        if command is None:
            raise ServerError("location must be node, edge, element, or interface")
        native_args = {k: args[k] for k in ("device", "region", "name", "interface") if k in args}
        if location == "interface":
            native_args.pop("region", None)
        values = self._native(command, native_args)
        if isinstance(values, dict) and "value" in values:
            values = values["value"]
        indices = args.get("indices")
        if indices is not None:
            values = [values[int(index)] for index in indices]
        result: dict[str, Any] = {"location": location, "device": args.get("device"), "region": args.get("region"), "name": args.get("name"), "summary": _summary(values)}
        result["values"] = self.store.encode_value(values, f"field_{args.get('name', 'model')}", int(args.get("inline_limit", 512)))
        return result

    def _topology(self) -> dict[str, Any]:
        devices = list(self.ds.get_device_list())
        result = {"devices": []}
        for device in devices:
            item: dict[str, Any] = {"name": device}
            try:
                item["dimension"] = self.ds.get_dimension(device=device)
            except Exception:
                item["dimension"] = None
            item["regions"] = list(self.ds.get_region_list(device=device))
            item["contacts"] = list(self.ds.get_contact_list(device=device))
            item["interfaces"] = list(self.ds.get_interface_list(device=device))
            result["devices"].append(item)
        result["meshes"] = list(self.ds.get_mesh_list())
        return result

    def _models_list(self, args: dict[str, Any]) -> dict[str, Any]:
        device, region = args["device"], args["region"]
        result = {
            "node": list(self.ds.get_node_model_list(device=device, region=region)),
            "edge": list(self.ds.get_edge_model_list(device=device, region=region)),
        }
        # Element models are not defined for 1D regions in DEVSIM.
        try:
            result["element"] = list(self.ds.get_element_model_list(device=device, region=region))
        except Exception:
            result["element"] = []
        return result

    def _equations_list(self, args: dict[str, Any]) -> dict[str, Any]:
        device, region = args["device"], args.get("region")
        result: dict[str, Any] = {}
        if region:
            result["equations"] = {name: self._native("get_equation_command", {"device": device, "region": region, "name": name}) for name in self.ds.get_equation_list(device=device, region=region)}
        if args.get("contact"):
            contact = args["contact"]
            result["contact_equations"] = {name: self._native("get_contact_equation_command", {"device": device, "contact": contact, "name": name}) for name in self.ds.get_contact_equation_list(device=device, contact=contact)}
        if args.get("interface"):
            interface = args["interface"]
            result["interface_equations"] = {name: self._native("get_interface_equation_command", {"device": device, "interface": interface, "name": name}) for name in self.ds.get_interface_equation_list(device=device, interface=interface)}
        return result

    def _measure_contact(self, args: dict[str, Any]) -> dict[str, Any]:
        device, contact, equation = args["device"], args["contact"], args["equation"]
        result: dict[str, Any] = {}
        if args.get("current", True):
            result["current"] = self._native("get_contact_current", {"device": device, "contact": contact, "equation": equation})
        if args.get("charge", False):
            result["charge"] = self._native("get_contact_charge", {"device": device, "contact": contact, "equation": equation})
        return result

    def _native_batch(self, operations: list[dict[str, Any]]) -> list[Any]:
        results = []
        for item in operations:
            if not isinstance(item, dict) or "command" not in item:
                raise ServerError("Each native_batch item needs command and arguments")
            results.append(self._native(str(item["command"]), dict(item.get("arguments", {}))))
        return results

    def _parameters_set(self, changes: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        for change in changes:
            results.append(self._native("set_parameter", dict(change)))
        return {"changed": len(changes), "results": results}

    def _solution_define(self, args: dict[str, Any]) -> Any:
        location = str(args.get("location", "node"))
        command = {"node": "node_solution", "edge": "edge_solution", "element": "element_solution"}.get(location)
        if command is None:
            raise ServerError("solution location must be node, edge, or element")
        result = self._native(command, {"device": args["device"], "region": args["region"], "name": args["name"]})
        if location == "node":
            # Equations reference a node solution through its Potential@n0 /
            # Potential@n1 edge models, and DEVSIM does not create those
            # implicitly.  Without this the first solve fails with
            # 'Value for "Potential@n0" not available'.
            self._native(
                "edge_from_node_model",
                {"device": args["device"], "region": args["region"], "node_model": args["name"]},
            )
        return result

    def _model_define(self, args: dict[str, Any]) -> Any:
        location = str(args.get("location", "node"))
        commands = {
            "node": "node_model",
            "edge": "edge_model",
            "element": "element_model",
            "contact_node": "contact_node_model",
            "contact_edge": "contact_edge_model",
            "interface": "interface_model",
        }
        command = commands.get(location)
        if command is None:
            raise ServerError("model location is invalid")
        if location == "interface":
            native_args = {"device": args["device"], "interface": args["interface"], "name": args["name"], "equation": args["equation"]}
        elif location.startswith("contact"):
            native_args = {"device": args["device"], "contact": args["contact"], "name": args["name"], "equation": args["equation"]}
        else:
            native_args = {"device": args["device"], "region": args["region"], "name": args["name"], "equation": args["equation"]}
        if "display_type" in args:
            native_args["display_type"] = args["display_type"]
        return self._native(command, native_args)

    def _equation_define(self, args: dict[str, Any]) -> Any:
        kind = str(args.get("kind", "bulk"))
        command = {"bulk": "equation", "contact": "contact_equation", "interface": "interface_equation"}.get(kind)
        if command is None:
            raise ServerError("equation kind must be bulk, contact, or interface")
        native_args = {k: v for k, v in args.items() if k not in {"kind"}}
        return self._native(command, native_args)

    def _solve(self, args: dict[str, Any]) -> Any:
        solve_args = dict(args)
        solve_args.pop("analysis", None)
        if "type" not in solve_args:
            analysis = args.get("analysis", "dc")
            solve_args["type"] = {"transient": "transient_bdf1"}.get(analysis, analysis)
        if solve_args.get("type") in {"ac", "noise"} and args.get("info"):
            solve_args.pop("info", None)
        return self._native("solve", solve_args)

    def _sweep(self, args: dict[str, Any], mode: str, job_id: str) -> dict[str, Any]:
        values: list[float]
        if mode == "bias":
            start, stop, step = float(args["start"]), float(args["stop"]), abs(float(args["step"]))
            if step == 0:
                raise ServerError("Sweep step cannot be zero")
            sign = 1 if stop >= start else -1
            values = []
            current = start
            while (current <= stop if sign > 0 else current >= stop):
                values.append(current)
                current += sign * step
            if not values or values[-1] != stop:
                values.append(stop)
            parameter = args.get("parameter", f"{args['contact']}_bias")
        elif mode == "frequency":
            values = [float(x) for x in args["values"]]
            parameter = None
        else:
            values = [float(x) for x in args["times"]]
            parameter = None
        rows: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            if self.cancel_events[job_id].is_set():
                raise InterruptedError("Job cancelled")
            self.jobs[job_id].update({"step": index + 1, "total_steps": len(values), "current_value": value})
            if mode == "bias":
                parameter_args = {"device": args["device"], "name": parameter, "value": value}
                if args.get("region"):
                    parameter_args["region"] = args["region"]
                self._native("set_parameter", parameter_args)
                solve_args = dict(args.get("solve", {"type": "dc", "absolute_error": 1.0, "relative_error": 1e-10, "maximum_iterations": 30}))
            elif mode == "frequency":
                solve_args = dict(args.get("solve", {"type": "ac"}))
                solve_args.update({"type": "ac", "frequency": value})
            else:
                solve_args = dict(args.get("solve", {"type": "transient_bdf1", "tdelta": value}))
                solve_args.update({"type": solve_args.get("type", "transient_bdf1"), "tdelta": value})
            solve_result = self._solve(solve_args)
            row: dict[str, Any] = {"value": value, "solve": solve_result}
            for measure in args.get("measure", []):
                if isinstance(measure, str):
                    measure = {"contact": args.get("contact"), "equation": measure, "name": measure}
                contact = measure.get("contact", args.get("contact"))
                equation = measure.get("equation", measure.get("name", "PotentialEquation"))
                if contact:
                    row[measure.get("name", equation)] = self._measure_contact({"device": args["device"], "contact": contact, "equation": equation, "current": measure.get("kind", "current") == "current", "charge": measure.get("kind") == "charge"})
            rows.append(row)
        artifact = self.store.write_json(f"sweep_{job_id}.json", rows, "sweep_data")
        return {"mode": mode, "rows": rows, "artifact": artifact}

    def _checkpoint_save(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", _new_id("checkpoint")))
        path = self.store.safe_output(name if name.endswith(".msh") else name + ".msh", name + ".msh", directory="checkpoints")
        result = self._native(
            "write_devices",
            {"file": str(path), "type": args.get("type", "devsim")},
            artifact_kind="checkpoint",
        )
        return {"checkpoint": result.get("artifacts", []) if isinstance(result, dict) else [], "result": result}

    def _checkpoint_load(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._input_file(str(args["file"]))
        self._native("reset_devsim", {})
        self._native("load_devices", {"file": path})
        return {"loaded": path, "topology": self._topology()}

    def _result_export(self, args: dict[str, Any]) -> dict[str, Any]:
        output_type = str(args.get("type", "vtk"))
        path = self.store.safe_output(args.get("file"), _new_id("export"), directory="results")
        native_args: dict[str, Any] = {"file": str(path), "type": output_type}
        if args.get("device"):
            native_args["device"] = args["device"]
        models = list(args.get("models", []))
        # include_test is a Python callback, so the MCP wrapper creates it
        # internally from a JSON model allow-list.
        if models and output_type in {"vtk", "tecplot"}:
            native_args["include_test"] = lambda name: name in models
            self._record("write_devices", {"file": str(path), "type": output_type, **({"device": args["device"]} if args.get("device") else {})}, models)
            stdout, redirect = _capture_stdout()
            with redirect:
                self.ds.write_devices(**native_args)
            return {"logs": stdout.getvalue(), "artifacts": self.store.capture(path, "result_export"), "models": models}
        result = self._native("write_devices", native_args, artifact_kind="result_export")
        return {
            "result": result.get("value") if isinstance(result, dict) else result,
            "logs": result.get("logs", "") if isinstance(result, dict) else "",
            "artifacts": result.get("artifacts", []) if isinstance(result, dict) else [],
        }

    def execute(self, operation: str, args: dict[str, Any], job_id: str | None = None) -> Any:
        if operation == "native_call":
            return self._native(str(args["command"]), dict(args.get("arguments", {})))
        if operation == "native_batch":
            return self._native_batch(list(args.get("operations", [])))
        if operation == "mesh_define":
            return self._mesh_define(dict(args))
        if operation == "mesh_import_gmsh":
            return self._gmsh_import(dict(args))
        if operation == "mesh_finalize":
            return self._native("finalize_mesh", {"mesh": args["mesh"]})
        if operation == "device_create":
            return self._native("create_device", {"mesh": args["mesh"], "device": args["device"]})
        if operation == "parameters_set":
            return self._parameters_set(list(args.get("changes", [])))
        if operation == "material_set":
            native_args = {k: args[k] for k in ("device", "region", "contact", "material") if k in args}
            return self._native("set_material", native_args)
        if operation == "solution_define":
            return self._solution_define(dict(args))
        if operation == "model_define":
            return self._model_define(dict(args))
        if operation == "equation_define":
            return self._equation_define(dict(args))
        if operation == "expression_validate":
            return {"expression": args["expression"], "result": self._native("symdiff", {"expr": args["expression"]})}
        if operation == "physics_apply":
            return self._physics_apply(dict(args))
        if operation == "solve_run":
            return self._solve(dict(args))
        if operation in {"bias_sweep", "frequency_sweep", "transient_sweep"}:
            if not job_id:
                raise ServerError("Sweep operations must run as jobs")
            mode = {"bias_sweep": "bias", "frequency_sweep": "frequency", "transient_sweep": "transient"}[operation]
            return self._sweep(dict(args), mode, job_id)
        if operation == "parameter_patch":
            checkpoint = None
            if self.ds.get_device_list():
                checkpoint = self._checkpoint_save({"name": f"pre_patch_{_new_id('state')}"})
            try:
                for change in args.get("changes", []):
                    change = dict(change)
                    change.pop("scope", None)
                    self._native("set_parameter", change)
                result: dict[str, Any] = {"changed": len(args.get("changes", []))}
                if args.get("solve_after"):
                    result["solve"] = self._solve(dict(args["solve_after"]))
                return result
            except Exception:
                if checkpoint:
                    try:
                        cp = checkpoint["checkpoint"][0]["path"]
                        self._native("reset_devsim", {})
                        self._native("load_devices", {"file": str(self.store.root / cp)})
                    except Exception:
                        pass
                raise
        if operation == "checkpoint_save":
            return self._checkpoint_save(dict(args))
        if operation == "checkpoint_load":
            return self._checkpoint_load(dict(args))
        if operation == "result_export":
            return self._result_export(dict(args))
        if operation == "field_get":
            return self._field_get(dict(args))
        if operation == "topology_get":
            return self._topology()
        if operation == "models_list":
            return self._models_list(dict(args))
        if operation == "equations_list":
            return self._equations_list(dict(args))
        if operation == "contact_measure":
            return self._measure_contact(dict(args))
        if operation == "circuit_inspect":
            if not self.circuit_configured:
                return {"nodes": [], "solutions": [], "values": {}}
            nodes = list(self.ds.get_circuit_node_list())
            solutions = list(self.ds.get_circuit_solution_list())
            values = {solution: {node: self.ds.get_circuit_node_value(solution=solution, node=node) for node in nodes} for solution in solutions}
            return {"nodes": nodes, "solutions": solutions, "values": values}
        if operation == "matrix_get":
            return self._native("get_matrix_and_rhs", {"format": args.get("format", "csr")})
        if operation == "recipe_run":
            steps = list(args.get("steps", []))
            results = []
            for index, step in enumerate(steps):
                if job_id and self.cancel_events[job_id].is_set():
                    raise InterruptedError("Job cancelled")
                if job_id:
                    self.jobs[job_id].update({"phase": "recipe", "step": index + 1, "total_steps": len(steps)})
                step = dict(step)
                op = str(step.pop("operation", step.pop("op", "native_call")))
                results.append({"operation": op, "result": self.execute(op, step, job_id)})
            return {"steps": results, "topology": self._topology()}
        raise ServerError(f"Unknown operation: {operation}")

    def execute_captured(self, operation: str, args: dict[str, Any], job_id: str | None = None) -> Any:
        """Keep all worker output away from the MCP stdio wire."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = self.execute(operation, args, job_id)
        logs = "".join(part for part in (stdout.getvalue(), stderr.getvalue()) if part)
        if not logs:
            return result
        if isinstance(result, dict):
            merged = dict(result)
            merged["logs"] = (str(merged.get("logs", "")) + logs)
            return merged
        return {"value": result, "logs": logs}

    def start_job(self, operation: str, args: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active_job:
                raise ServerError(f"A job is already running: {self.active_job}")
            job_id = _new_id("job")
            self.jobs[job_id] = {"job_id": job_id, "operation": operation, "status": "queued", "phase": "queued", "step": 0, "total_steps": None, "created_at": _now(), "updated_at": _now()}
            self.cancel_events[job_id] = threading.Event()
            self.active_job = job_id
            thread = threading.Thread(target=self._run_job, args=(job_id, operation, dict(args)), daemon=True)
            thread.start()
            return copy.deepcopy(self.jobs[job_id])

    def _run_job(self, job_id: str, operation: str, args: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job_id].update({"status": "running", "phase": operation, "started_at": _now(), "updated_at": _now()})
        try:
            # The control loop must still be able to answer job_status while
            # DEVSIM is solving.  The session has only one mutating job, so
            # the worker thread can run without holding the status lock.
            result = self.execute_captured(operation, args, job_id)
            with self.lock:
                self.jobs[job_id].update({"status": "completed", "result": result, "completed_at": _now(), "updated_at": _now()})
        except InterruptedError as exc:
            with self.lock:
                self.jobs[job_id].update({"status": "cancelled", "error": str(exc), "completed_at": _now(), "updated_at": _now()})
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "completed_at": _now(), "updated_at": _now()})
        finally:
            with self.lock:
                self.active_job = None
                self.cancel_events.pop(job_id, None)

    def status(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise ServerError(f"Unknown job: {job_id}")
            result = copy.deepcopy(self.jobs[job_id])
            result["updated_at"] = _now()
            return result

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise ServerError(f"Unknown job: {job_id}")
            event = self.cancel_events.get(job_id)
            if event:
                event.set()
            self.jobs[job_id]["cancel_requested"] = True
            return copy.deepcopy(self.jobs[job_id])

    def reset(self) -> dict[str, Any]:
        with self.lock:
            if self.active_job:
                raise ServerError("Cannot reset while a job is running")
            result = self._native("reset_devsim", {})
            return {"reset": True, "result": result}

    def export_script(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("name", "run_devsim.py"))
        if not name.endswith(".py"):
            name += ".py"
        operations = copy.deepcopy(self.operation_log)
        payload = json.dumps(operations, indent=2, ensure_ascii=False)
        script = f'''#!/usr/bin/env python3
"""Generated by devsim_mcp; runs without the MCP server."""
import argparse
import json
import os
import devsim
import devsim.python_packages.simple_physics as _physics

OPERATIONS = json.loads({payload!r})

def _ensure_solution(device, region, name):
    """Mirror solution_define: node solution plus its @n0/@n1 edge models."""
    if name in devsim.get_node_model_list(device=device, region=region):
        return False
    _physics.CreateSolution(device, region, name)
    return True

def _default_bias(device, contact):
    name = contact + "_bias"
    try:
        devsim.get_parameter(device=device, name=name)
    except Exception:
        devsim.set_parameter(device=device, name=name, value=0.0)

def apply_physics_preset(arguments):
    preset = arguments["preset"]
    device = arguments["device"]
    regions = arguments.get("regions", [])
    contacts = arguments.get("contacts", [])
    interfaces = arguments.get("interfaces", [])
    region = arguments.get("region", regions[0] if regions else "")
    temperature = arguments.get("temperature", 300.0)
    if preset == "silicon_potential_only":
        for item in regions:
            _physics.SetSiliconParameters(device, item, temperature)
            _ensure_solution(device, item, "Potential")
            _physics.CreateSiliconPotentialOnly(device, item)
        for item in contacts:
            _default_bias(device, item)
            _physics.CreateSiliconPotentialOnlyContact(device, region, item)
        for item in interfaces:
            _physics.CreateSiliconOxideInterface(device, item)
    elif preset == "silicon_drift_diffusion":
        for item in regions:
            _physics.SetSiliconParameters(device, item, temperature)
            _ensure_solution(device, item, "Potential")
            _physics.CreateSiliconPotentialOnly(device, item)
            created = _ensure_solution(device, item, "Electrons")
            created = _ensure_solution(device, item, "Holes") or created
            if created:
                devsim.set_node_values(device=device, region=item, name="Electrons", init_from="IntrinsicElectrons")
                devsim.set_node_values(device=device, region=item, name="Holes", init_from="IntrinsicHoles")
            _physics.CreateSiliconDriftDiffusion(device, item)
        for item in contacts:
            _default_bias(device, item)
            _physics.CreateSiliconPotentialOnlyContact(device, region, item)
            _physics.CreateSiliconDriftDiffusionAtContact(device, region, item)
        for item in interfaces:
            _physics.CreateSiliconSiliconInterface(device, item)
    elif preset == "oxide_potential_only":
        for item in regions:
            _physics.SetOxideParameters(device, item, temperature)
            _physics.CreateOxidePotentialOnly(device, item)
        for item in contacts:
            _physics.CreateOxideContact(device, region, item)
        for item in interfaces:
            _physics.CreateSiliconOxideInterface(device, item)
    else:
        raise ValueError("Unknown physics preset: " + str(preset))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    options = parser.parse_args()
    os.makedirs(options.output_dir, exist_ok=True)
    for item in OPERATIONS:
        command = item["command"]
        arguments = dict(item.get("arguments", {{}}))
        if command == "physics_apply":
            apply_physics_preset(arguments)
            continue
        if command == "write_devices":
            arguments["file"] = os.path.join(options.output_dir, os.path.basename(arguments["file"]))
            models = item.get("include_models")
            if models:
                arguments["include_test"] = lambda name, allowed=set(models): name in allowed
        if command == "load_devices" and not os.path.isabs(arguments.get("file", "")):
            arguments["file"] = os.path.join(options.output_dir, arguments["file"])
        result = getattr(devsim, command)(**arguments)
        if result is not None and command == "solve":
            print(json.dumps(result, default=str))

if __name__ == "__main__":
    main()
'''
        path = self.store.safe_output(name, "run_devsim.py", directory="exports")
        path.write_text(script, encoding="utf-8")
        entry = self.store._record(path, "python_script", {"standalone": True, "operation_count": len(operations)})
        return {"artifact": entry, "operation_count": len(operations), "warning": "Custom Python callables are not exportable through native JSON operations."}

    def runtime_info(self) -> dict[str, Any]:
        info = self.ds.get_parameter(name="info")
        return {"server_version": SERVER_VERSION, "devsim_version": getattr(self.ds, "__version__", None), "devsim_path": str(Path(self.ds.__file__).resolve()) if getattr(self.ds, "__file__", None) else None, "python_version": sys.version.split()[0], "info": _normalise(info), "commands": sorted(name for name in dir(self.ds) if not name.startswith("_") and callable(getattr(self.ds, name, None)))}


def _worker_main(requests: Any, responses: Any, artifact_root: str) -> None:
    # DEVSIM may print solver/library messages during import.  The worker's
    # stdout is inherited from the MCP process, so suppress it before the
    # request loop starts and capture operation output below.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        state = WorkerState(artifact_root)
    while True:
        request = requests.get()
        request_id = request.get("id")
        command = request.get("command")
        try:
            if command == "shutdown":
                responses.put({"id": request_id, "ok": True, "result": {"closed": True}})
                return
            if command == "runtime_info":
                result = state.runtime_info()
            elif command == "invoke":
                if state.active_job:
                    raise ServerError(f"A job is running: {state.active_job}")
                operation = str(request["operation"])
                if operation == "export_python_script":
                    result = state.export_script(dict(request.get("arguments", {})))
                elif operation == "reset":
                    result = state.reset()
                else:
                    with state.lock:
                        result = state.execute_captured(operation, dict(request.get("arguments", {})))
            elif command == "start_job":
                result = state.start_job(str(request["operation"]), dict(request.get("arguments", {})))
            elif command == "job_status":
                result = state.status(str(request["job_id"]))
            elif command == "job_cancel":
                result = state.cancel(str(request["job_id"]))
            elif command == "artifacts":
                result = state.store.list()
            elif command == "artifact_read":
                result = state.store.read(str(request["artifact_id"]))
            else:
                raise ServerError(f"Unknown worker command: {command}")
            responses.put({"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            responses.put({"id": request_id, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})


class SessionProxy:
    def __init__(self, session_id: str, root: Path, context: mp.context.BaseContext):
        self.session_id = session_id
        self.root = root
        self.requests = context.Queue()
        self.responses = context.Queue()
        self.process = context.Process(target=_worker_main, args=(self.requests, self.responses, str(root)), daemon=True)
        self.process.start()
        self.lock = threading.Lock()
        self._closed = False
        try:
            self._request("runtime_info")
        except Exception:
            self.close()
            raise

    def _request(self, command: str, timeout: float = 120.0, **payload: Any) -> Any:
        if not self.process.is_alive():
            raise ServerError("DEVSIM worker process is not running")
        request_id = _new_id("request")
        with self.lock:
            self.requests.put({"id": request_id, "command": command, **payload})
            deadline = time.monotonic() + timeout
            while True:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    response = self.responses.get(timeout=remaining)
                except queue.Empty as exc:
                    raise ServerError("Timed out waiting for DEVSIM worker") from exc
                if response.get("id") != request_id:
                    continue
                if not response.get("ok"):
                    error = response.get("error", {})
                    raise ServerError(f"{error.get('type', 'WorkerError')}: {error.get('message', 'unknown worker error')}")
                return response.get("result")

    def invoke(self, operation: str, arguments: dict[str, Any], timeout: float = 120.0) -> Any:
        return self._request("invoke", timeout, operation=operation, arguments=arguments)

    def start_job(self, operation: str, arguments: dict[str, Any]) -> Any:
        return self._request("start_job", operation=operation, arguments=arguments)

    def job_status(self, job_id: str) -> Any:
        return self._request("job_status", job_id=job_id)

    def job_cancel(self, job_id: str) -> Any:
        return self._request("job_cancel", job_id=job_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.is_alive():
                try:
                    self._request("shutdown", timeout=5)
                except Exception:
                    pass
        finally:
            # Always reap the process.  ``terminate`` alone can leave a
            # finished worker as a zombie until the parent exits.
            if self.process.is_alive():
                self.process.terminate()
            try:
                self.process.join(timeout=5)
            except Exception:
                pass
            for channel in (self.requests, self.responses):
                try:
                    channel.close()
                    channel.join_thread()
                except Exception:
                    pass


class SessionManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = mp.get_context("spawn")
        self.sessions: dict[str, SessionProxy] = {}
        self.default_session: str | None = None
        self.lock = threading.RLock()

    def _create_direct(self, name: str | None = None) -> dict[str, Any]:
        with self.lock:
            session_id = _new_id("session")
            session_root = self.root / session_id
            session: SessionProxy | None = None
            try:
                session = SessionProxy(session_id, session_root, self.context)
                self.sessions[session_id] = session
                if self.default_session is None or name == "default":
                    self.default_session = session_id
                info = session.invoke("topology_get", {})
                runtime = session._request("runtime_info")
                return {"session_id": session_id, "name": name or session_id, "root": str(session_root), "runtime": runtime, "topology": info}
            except Exception:
                self.sessions.pop(session_id, None)
                if self.default_session == session_id:
                    self.default_session = next(iter(self.sessions), None)
                if session is not None:
                    session.close()
                raise

    def create(self, name: str | None = None) -> dict[str, Any]:
        if threading.current_thread() is not threading.main_thread():
            raise ServerError("DEVSIM session creation must be called from the MCP server event loop")
        return self._create_direct(name)

    def get(self, session_id: str | None = None) -> SessionProxy:
        with self.lock:
            if session_id and session_id in self.sessions:
                return self.sessions[session_id]
            if self.default_session and self.default_session in self.sessions:
                return self.sessions[self.default_session]
            return self.sessions[self.create("default")["session_id"]]

    def close(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            session = self.sessions.pop(session_id, None)
            if not session:
                raise ServerError(f"Unknown session: {session_id}")
            session.close()
            if self.default_session == session_id:
                self.default_session = next(iter(self.sessions), None)
            return {"session_id": session_id, "closed": True}

    def close_all(self) -> None:
        with self.lock:
            for session in list(self.sessions.values()):
                session.close()
            self.sessions.clear()


DATA_ROOT = Path(os.environ.get("DEVSIM_MCP_DATA_DIR", Path.cwd() / ".devsim_mcp"))
MANAGER = SessionManager(DATA_ROOT)


def _success(result: Any, session_id: str | None = None) -> dict[str, Any]:
    return {"ok": True, "session_id": session_id, "result": result, "error": None}


def _failure(exc: Exception, session_id: str | None = None) -> dict[str, Any]:
    return {"ok": False, "session_id": session_id, "result": None, "error": {"type": type(exc).__name__, "message": str(exc), "recoverable": True}}


def _call(session_id: str | None, operation: str, arguments: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        result = session.invoke(operation, arguments, timeout=timeout)
        return _success(result, session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


def _start_job(session_id: str | None, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        result = session.start_job(operation, arguments)
        return _success(result, session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


async def _call_async(session_id: str | None, operation: str, arguments: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    # SessionProxy communication is intentionally kept on the MCP event-loop
    # thread.  Starting or talking to multiprocessing queues from AnyIO's
    # generic worker pool is not reliable on all supported MCP runtimes.
    return _call(session_id, operation, arguments, timeout)


async def _start_job_async(session_id: str | None, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return _start_job(session_id, operation, arguments)


server = MCPServer(
    name="devsim-mcp",
    version=SERVER_VERSION,
    description="Stateful MCP control plane for the pip-installed DEVSIM TCAD simulator.",
    instructions="Use recipe operations for normal simulations. Use native_call only for advanced JSON-compatible DEVSIM commands. Large data is returned as artifacts.",
)


@server.tool(name="runtime_info", description="Return DEVSIM and MCP runtime capabilities.", structured_output=True)
async def runtime_info() -> dict[str, Any]:
    try:
        session = MANAGER.get()
        result = session._request("runtime_info")
        return _success(result, session.session_id)
    except Exception as exc:
        return _failure(exc)


@server.tool(name="session_create", description="Create an isolated DEVSIM session.", structured_output=True)
async def session_create(name: str = "default") -> dict[str, Any]:
    try:
        result = MANAGER.create(name)
        return _success(result, result["session_id"])
    except Exception as exc:
        return _failure(exc)


@server.tool(name="session_status", description="Return topology and runtime status for a session.", structured_output=True)
async def session_status(session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "topology_get", {})


@server.tool(name="session_reset", description="Reset all DEVSIM state in a session.", structured_output=True)
async def session_reset(session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        result = session.invoke("reset", {})
        return _success(result, session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="session_close", description="Close an isolated DEVSIM session.", structured_output=True)
async def session_close(session_id: str) -> dict[str, Any]:
    try:
        return _success(MANAGER.close(session_id), session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="native_call", description="Call any JSON-compatible native DEVSIM command.", structured_output=True)
async def native_call(command: str, arguments: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "native_call", {"command": command, "arguments": arguments or {}})


@server.tool(name="native_batch", description="Execute a sequence of JSON-compatible native DEVSIM commands.", structured_output=True)
async def native_batch(operations: list[dict[str, Any]], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "native_batch", {"operations": operations})


@server.tool(name="mesh_define", description="Define a structured 1D or 2D mesh with regions, contacts, and interfaces.", structured_output=True)
async def mesh_define(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "mesh_define", config)


@server.tool(name="mesh_import_gmsh", description="Import a Gmsh mesh and map physical groups to DEVSIM objects.", structured_output=True)
async def mesh_import_gmsh(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "mesh_import_gmsh", config)


@server.tool(name="mesh_finalize", description="Finalize a mesh before creating a DEVSIM device.", structured_output=True)
async def mesh_finalize(mesh: str, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "mesh_finalize", {"mesh": mesh})


@server.tool(name="device_create", description="Create a DEVSIM device from a finalized mesh.", structured_output=True)
async def device_create(mesh: str, device: str, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "device_create", {"mesh": mesh, "device": device})


@server.tool(name="parameters_set", description="Set several global, device, or region parameters in one operation.", structured_output=True)
async def parameters_set(changes: list[dict[str, Any]], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "parameters_set", {"changes": changes})


@server.tool(name="material_set", description="Set a material on a region or contact.", structured_output=True)
async def material_set(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "material_set", config)


@server.tool(name="physics_list_presets", description="List physics presets provided by this single-file server.", structured_output=True)
async def physics_list_presets() -> dict[str, Any]:
    return _success({"presets": ["silicon_potential_only", "silicon_drift_diffusion", "oxide_potential_only"]})


@server.tool(name="physics_apply", description="Apply a built-in silicon or oxide physics preset.", structured_output=True)
async def physics_apply(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "physics_apply", config)


@server.tool(name="solution_define", description="Create a node, edge, or element solution variable.", structured_output=True)
async def solution_define(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "solution_define", config)


@server.tool(name="model_define", description="Create a node, edge, element, contact, or interface model.", structured_output=True)
async def model_define(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "model_define", config)


@server.tool(name="equation_define", description="Create a bulk, contact, or interface equation.", structured_output=True)
async def equation_define(config: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "equation_define", config)


@server.tool(name="expression_validate", description="Validate and symbolically simplify a DEVSIM expression.", structured_output=True)
async def expression_validate(expression: str, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "expression_validate", {"expression": expression})


@server.tool(name="recipe_validate", description="Validate recipe structure without running the simulation.", structured_output=True)
async def recipe_validate(recipe: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(recipe.get("steps", []), list):
        errors.append("recipe.steps must be a list")
    for index, step in enumerate(recipe.get("steps", [])):
        if not isinstance(step, dict) or not (step.get("operation") or step.get("op")):
            errors.append(f"steps[{index}] needs operation")
    return _success({"valid": not errors, "errors": errors}, session_id)


@server.tool(name="recipe_run", description="Run a declarative simulation recipe as a background job.", structured_output=True)
async def recipe_run(recipe: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _start_job_async(session_id, "recipe_run", recipe)


@server.tool(name="solve_run", description="Run one DEVSIM DC, AC, noise, or transient solve as a background job.", structured_output=True)
async def solve_run(options: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _start_job_async(session_id, "solve_run", options)


@server.tool(name="bias_sweep", description="Run a bias sweep and save tabular measurements as an artifact.", structured_output=True)
async def bias_sweep(options: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _start_job_async(session_id, "bias_sweep", options)


@server.tool(name="frequency_sweep", description="Run an AC frequency sweep and save tabular results.", structured_output=True)
async def frequency_sweep(options: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _start_job_async(session_id, "frequency_sweep", options)


@server.tool(name="transient_sweep", description="Run transient time points and save tabular results.", structured_output=True)
async def transient_sweep(options: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    return await _start_job_async(session_id, "transient_sweep", options)


@server.tool(name="parameter_patch", description="Apply parameter changes, optionally solve, and checkpoint before mutation.", structured_output=True)
async def parameter_patch(changes: list[dict[str, Any]], solve_after: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "parameter_patch", {"changes": changes, "solve_after": solve_after})


@server.tool(name="job_status", description="Return progress and result status for a background job.", structured_output=True)
async def job_status(job_id: str, session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        return _success(session.job_status(job_id), session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="job_wait", description="Wait for a background job until completion or timeout.", structured_output=True)
async def job_wait(job_id: str, timeout_seconds: float = 1.0, session_id: str | None = None) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    try:
        session = MANAGER.get(session_id)
        while True:
            status = session.job_status(job_id)
            if status.get("status") in TERMINAL_JOB_STATES or time.monotonic() >= deadline:
                return _success(status, session.session_id)
            await asyncio.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="job_cancel", description="Request cancellation between DEVSIM sweep steps.", structured_output=True)
async def job_cancel(job_id: str, session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        return _success(session.job_cancel(job_id), session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="topology_get", description="Inspect devices, meshes, regions, contacts, and interfaces.", structured_output=True)
async def topology_get(session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "topology_get", {})


@server.tool(name="models_list", description="List node, edge, and element models in a region.", structured_output=True)
async def models_list(device: str, region: str, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "models_list", {"device": device, "region": region})


@server.tool(name="equations_list", description="List equation definitions for a region, contact, or interface.", structured_output=True)
async def equations_list(device: str, region: str | None = None, contact: str | None = None, interface: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "equations_list", {"device": device, "region": region, "contact": contact, "interface": interface})


@server.tool(name="field_get", description="Get a model field, with large arrays stored as artifacts.", structured_output=True)
async def field_get(device: str, region: str | None, name: str, location: str = "node", interface: str | None = None, indices: list[int] | None = None, inline_limit: int = 512, session_id: str | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"device": device, "region": region, "name": name, "location": location, "inline_limit": inline_limit}
    if interface:
        arguments["interface"] = interface
    if indices is not None:
        arguments["indices"] = indices
    return await _call_async(session_id, "field_get", arguments)


@server.tool(name="contact_measure", description="Read contact current and/or charge for an equation.", structured_output=True)
async def contact_measure(device: str, contact: str, equation: str, current: bool = True, charge: bool = False, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "contact_measure", {"device": device, "contact": contact, "equation": equation, "current": current, "charge": charge})


@server.tool(name="circuit_inspect", description="Inspect circuit nodes, solutions, and node values.", structured_output=True)
async def circuit_inspect(session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "circuit_inspect", {})


@server.tool(name="matrix_get", description="Read the assembled DEVSIM matrix and RHS in CSR or CSC form.", structured_output=True)
async def matrix_get(format: str = "csr", session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "matrix_get", {"format": format})


@server.tool(name="checkpoint_save", description="Save a native DEVSIM checkpoint under the session artifact directory.", structured_output=True)
async def checkpoint_save(name: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "checkpoint_save", {"name": name} if name else {})


@server.tool(name="checkpoint_load", description="Load a native DEVSIM checkpoint from an allowed input path.", structured_output=True)
async def checkpoint_load(file: str, session_id: str | None = None) -> dict[str, Any]:
    return await _call_async(session_id, "checkpoint_load", {"file": file})


@server.tool(name="result_export", description="Export device results as devsim, devsim_data, VTK, or Tecplot files.", structured_output=True)
async def result_export(file: str | None = None, type: str = "vtk", device: str | None = None, models: list[str] | None = None, session_id: str | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"type": type, "models": models or []}
    if file:
        arguments["file"] = file
    if device:
        arguments["device"] = device
    return await _call_async(session_id, "result_export", arguments)


@server.tool(name="artifact_list", description="List saved result, checkpoint, data, and script artifacts.", structured_output=True)
async def artifact_list(session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        return _success(session._request("artifacts"), session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="artifact_read", description="Read a small text or JSON artifact by ID.", structured_output=True)
async def artifact_read(artifact_id: str, session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        return _success(session._request("artifact_read", artifact_id=artifact_id), session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.tool(name="export_python_script", description="Export recorded JSON-compatible operations as a standalone Python script.", structured_output=True)
async def export_python_script(name: str = "run_devsim.py", session_id: str | None = None) -> dict[str, Any]:
    try:
        session = MANAGER.get(session_id)
        return _success(session.invoke("export_python_script", {"name": name}), session.session_id)
    except Exception as exc:
        return _failure(exc, session_id)


@server.resource("devsim://capabilities", name="devsim-capabilities", description="DEVSIM MCP capability summary", mime_type="application/json")
async def capabilities_resource() -> str:
    return json.dumps({"server_version": SERVER_VERSION, "native_call": True, "large_data": "artifact", "plotting": False, "custom_python_callables": "trusted script or plugin only"}, indent=2)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True, mode="json")
    return _normalise(value)


def _stdio_response(request_id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": _model_dump(result)}, ensure_ascii=False, separators=(",", ":"))


def _stdio_error(request_id: Any, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}, ensure_ascii=False, separators=(",", ":"))


def _run_stdio_compat() -> None:
    """Serve the MCP legacy JSON-RPC handshake directly over stdin/stdout.

    This avoids a compatibility issue in some MCP 2.x stdio transport builds
    where the file-descriptor claiming layer does not deliver the first frame
    when launched through a pipe.  The wire format remains standard MCP JSON-RPC.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    initialized = False
    protocol_versions = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                sys.stdout.write(_stdio_error(None, -32700, "Parse error") + "\n")
                sys.stdout.flush()
                continue
            method = request.get("method")
            request_id = request.get("id")
            if method and method.startswith("notifications/"):
                continue
            try:
                if method == "initialize":
                    requested = str(request.get("params", {}).get("protocolVersion", "2025-11-25"))
                    negotiated = requested if requested in protocol_versions else "2025-11-25"
                    initialized = True
                    result = {
                        "protocolVersion": negotiated,
                        "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False}},
                        "serverInfo": {"name": server.name, "version": SERVER_VERSION},
                        "instructions": server.instructions,
                    }
                    output = _stdio_response(request_id, result)
                elif method == "ping":
                    output = _stdio_response(request_id, {})
                elif not initialized:
                    output = _stdio_error(request_id, -32002, "Server has not been initialized")
                elif method == "tools/list":
                    tools = loop.run_until_complete(server.list_tools())
                    output = _stdio_response(request_id, {"tools": [_model_dump(item) for item in tools]})
                elif method == "tools/call":
                    params = request.get("params", {})
                    result = loop.run_until_complete(server.call_tool(str(params.get("name", "")), dict(params.get("arguments", {}))))
                    output = _stdio_response(request_id, result)
                elif method == "resources/list":
                    resources = loop.run_until_complete(server.list_resources())
                    output = _stdio_response(request_id, {"resources": [_model_dump(item) for item in resources]})
                elif method == "resources/read":
                    uri = request.get("params", {}).get("uri", "")
                    result = loop.run_until_complete(server.read_resource(uri))
                    output = _stdio_response(request_id, {"contents": [_model_dump(item) for item in result]})
                else:
                    output = _stdio_error(request_id, -32601, f"Method not found: {method}")
            except Exception as exc:
                output = _stdio_error(request_id, -32603, str(exc))
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
    finally:
        loop.close()


def main() -> None:
    global DATA_ROOT, MANAGER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    DATA_ROOT = args.data_dir.resolve()
    if DATA_ROOT != MANAGER.root:
        MANAGER.close_all()
        MANAGER = SessionManager(DATA_ROOT)
    try:
        if args.transport == "stdio":
            _run_stdio_compat()
        elif args.transport == "sse":
            server.run("sse", host=args.host, port=args.port)
        else:
            server.run("streamable-http", host=args.host, port=args.port)
    finally:
        MANAGER.close_all()


if __name__ == "__main__":
    main()
