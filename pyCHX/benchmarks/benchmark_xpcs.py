"""Opt-in benchmark for the production XPCS compressed-data call path.

This module never runs as part of the test suite. Invoke it with
``python -m pyCHX.benchmarks.benchmark_xpcs --help``. Trials run in fresh
processes by default so peak-memory and lifetime state from one stage cannot
leak into the next stage's measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import multiprocessing
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from pyCHX import (
    Create_Report,
    chx_compress,
    chx_compress_analysis,
    chx_correlationc,
    chx_generic_functions,
)
from pyCHX._performance import (
    affinity_cpu_ids,
    available_memory_bytes,
    physical_core_count,
)
from pyCHX.chx_compress import Multifile, compress_eigerdata, mean_intensityc
from pyCHX.chx_compress_analysis import plot_each_ring_mean_intensityc
from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc
from pyCHX.chx_correlationp import cal_g2p
from pyCHX.Create_Report import export_xpcs_results_to_h5
from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time, show_C12

_CORE_STAGES = (
    "compression",
    "roi-intensity",
    "selected-pixels",
    "one-time",
    "two-time",
    "diagonal-means",
)
_ARTIFACT_STAGES = (
    "roi-intensity-plot",
    "g2-plot",
    "two-time-plot",
    "export",
)
_EXPERIMENTAL_STAGES = ("reader-reuse", "direct-fit", "g2-plot-reference", "export-raw")
_STAGES = _CORE_STAGES + _ARTIFACT_STAGES + _EXPERIMENTAL_STAGES
_ARTIFACT_REQUIRED_STAGES = _ARTIFACT_STAGES + ("g2-plot-reference", "export-raw")
_DEPENDENCIES = (
    "h5py",
    "lmfit",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "Pillow",
    "scikit-beam",
    "scipy",
    "threadpoolctl",
)


def _lifetime_rss_bytes():
    """Return the process lifetime high-water RSS, for context only."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _read_key_values(path):
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values[key] = value.strip()
    except OSError:
        pass
    return values


def _process_snapshot(pid):
    status = _read_key_values(f"/proc/{pid}/status")
    io_values = _read_key_values(f"/proc/{pid}/io")
    try:
        pss = int(_read_key_values(f"/proc/{pid}/smaps_rollup").get("Pss", "0 kB").split()[0]) * 1024
    except (OSError, ValueError):
        pss = 0

    def status_bytes(name):
        try:
            return int(status.get(name, "0 kB").split()[0]) * 1024
        except ValueError:
            return 0

    def io_bytes(name):
        try:
            return int(io_values.get(name, "0"))
        except ValueError:
            return 0

    try:
        threads = int(status.get("Threads", 0))
    except ValueError:
        threads = 0
    return {
        "rss_bytes": status_bytes("VmRSS"),
        "pss_bytes": pss,
        "threads": threads,
        "read_bytes": io_bytes("read_bytes"),
        "write_bytes": io_bytes("write_bytes"),
        "read_characters": io_bytes("rchar"),
        "write_characters": io_bytes("wchar"),
    }


def _descendant_pids(pid):
    """Return live descendants using Linux's per-task child list."""
    descendants = []
    pending = [pid]
    seen = {pid}
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child_text in children:
            try:
                child = int(child_text)
            except ValueError:
                continue
            if child not in seen:
                seen.add(child)
                descendants.append(child)
                pending.append(child)
    return descendants


def _process_tree_snapshots(pid=None):
    pid = os.getpid() if pid is None else pid
    return {child: _process_snapshot(child) for child in (pid, *_descendant_pids(pid))}


def _sum_process_snapshots(snapshots):
    snapshots = list(snapshots.values())
    keys = snapshots[0].keys()
    totals = {key: sum(snapshot[key] for snapshot in snapshots) for key in keys}
    totals["processes"] = len(snapshots)
    return totals


def _process_tree_snapshot(pid=None):
    return _sum_process_snapshots(_process_tree_snapshots(pid))


def _usage_snapshot():
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "cpu_user_seconds": self_usage.ru_utime + child_usage.ru_utime,
        "cpu_system_seconds": self_usage.ru_stime + child_usage.ru_stime,
        "minor_faults": self_usage.ru_minflt + child_usage.ru_minflt,
        "major_faults": self_usage.ru_majflt + child_usage.ru_majflt,
        "voluntary_context_switches": self_usage.ru_nvcsw + child_usage.ru_nvcsw,
        "involuntary_context_switches": self_usage.ru_nivcsw + child_usage.ru_nivcsw,
    }


def _subtract(after, before, keys):
    return {key: after[key] - before[key] for key in keys}


def _measure(name, function, logical_bytes_visited=0):
    """Measure one stage without seeding its peak from lifetime ``ru_maxrss``."""
    before_processes = _process_tree_snapshots()
    before_tree = _sum_process_snapshots(before_processes)
    sampled_process_ids = set(before_processes)
    before_usage = _usage_snapshot()
    peaks = {
        "rss_bytes": before_tree["rss_bytes"],
        "pss_bytes": before_tree["pss_bytes"],
        "threads": before_tree["threads"],
        "processes": before_tree["processes"],
    }
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.02):
            process_snapshots = _process_tree_snapshots()
            sampled_process_ids.update(process_snapshots)
            snapshot = _sum_process_snapshots(process_snapshots)
            for key, peak in peaks.items():
                peaks[key] = max(peak, snapshot[key])

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started_wall = time.perf_counter()
    started_monotonic_ns = time.monotonic_ns()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - started_wall
        stopped.set()
        monitor_thread.join()
    after_processes = _process_tree_snapshots()
    sampled_process_ids.update(after_processes)
    after_tree = _sum_process_snapshots(after_processes)
    after_usage = _usage_snapshot()
    for key, peak in peaks.items():
        peaks[key] = max(peak, after_tree[key])
    usage_delta = _subtract(after_usage, before_usage, before_usage)
    io_keys = ("read_bytes", "write_bytes", "read_characters", "write_characters")
    parent_pid = os.getpid()
    live_descendant_ids = sorted(pid for pid in after_processes if pid != parent_pid)
    io_delta = _subtract(after_processes[parent_pid], before_processes[parent_pid], io_keys)
    for child_pid in live_descendant_ids:
        child_after = after_processes[child_pid]
        child_before = before_processes.get(child_pid, {})
        for key in io_keys:
            io_delta[key] += max(0, child_after[key] - child_before.get(key, 0))
    return result, {
        "name": name,
        "seconds": elapsed,
        "started_monotonic_ns": started_monotonic_ns,
        "rss_before_bytes": before_tree["rss_bytes"],
        "rss_after_bytes": after_tree["rss_bytes"],
        "peak_rss_bytes": peaks["rss_bytes"],
        "peak_rss_increase_bytes": max(0, peaks["rss_bytes"] - before_tree["rss_bytes"]),
        "pss_before_bytes": before_tree["pss_bytes"],
        "peak_pss_bytes": peaks["pss_bytes"],
        "lifetime_self_peak_rss_bytes": _lifetime_rss_bytes(),
        "peak_threads": peaks["threads"],
        "peak_processes": peaks["processes"],
        "logical_bytes_visited": int(logical_bytes_visited),
        "os_io": io_delta,
        "os_io_accounting": {
            "complete": not live_descendant_ids,
            "method": "final_parent_delta_plus_live_descendants",
            "sampled_descendant_count": len(sampled_process_ids - {parent_pid}),
            "live_descendant_pids": live_descendant_ids,
        },
        "resource_usage": usage_delta,
    }


def _call_with_component_timers(function, patched_calls):
    """Time selected nested calls while retaining their original behavior."""
    components = {name: {"count": 0, "seconds": 0.0} for name, _, _ in patched_calls}

    def wrapper(name, original):
        if inspect.isgeneratorfunction(original):

            def timed_generator(*args, **kwargs):
                started = time.perf_counter()
                try:
                    yield from original(*args, **kwargs)
                finally:
                    components[name]["seconds"] += time.perf_counter() - started
                    components[name]["count"] += 1

            return timed_generator

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                components[name]["seconds"] += time.perf_counter() - started
                components[name]["count"] += 1

        return timed

    with contextlib.ExitStack() as stack:
        for name, owner, attribute in patched_calls:
            original = getattr(owner, attribute)
            stack.enter_context(mock.patch.object(owner, attribute, wrapper(name, original)))
        result = function()
    return result, components


def _sha256(path, block_size=8 * 1024**2):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path, hash_contents=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = {
        "path": os.fspath(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    if hash_contents:
        record["sha256"] = _sha256(path)
    return record


def _nearest_existing_path(path):
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _filesystem_record(path):
    path = Path(path).resolve()
    usage = shutil.disk_usage(_nearest_existing_path(path))
    record = {
        "path": os.fspath(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
    try:
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            separator = fields.index("-")
            mounts.append((fields[4], fields[separator + 1], fields[separator + 2]))
        mount, filesystem, source = max(
            (item for item in mounts if os.path.commonpath((os.fspath(path), item[0])) == item[0]),
            key=lambda item: len(item[0]),
        )
        record.update({"mount": mount, "filesystem": filesystem, "source": source})
    except (OSError, ValueError):
        pass
    return record


def _read_optional_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _binding_record(function):
    source = inspect.getsourcefile(function)
    record = {
        "name": function.__name__,
        "qualname": function.__qualname__,
        "module": function.__module__,
        "source": source,
        "callable_source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
    }
    if source and Path(source).is_file():
        record["source_file_sha256"] = _sha256(source)
    return record


def _numpy_configuration():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return output.getvalue()


def _working_tree_record(repository):
    """Record dirty runtime sources so benchmark results are reproducible."""
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    status_lines = completed.stdout.splitlines()
    source_files = []
    for line in status_lines:
        relative_path = line[3:]
        path = repository / relative_path
        if path.is_file() and path.suffix in {".py", ".pyx", ".pxd"}:
            record = _file_record(path, hash_contents=True)
            record["status"] = line[:2]
            source_files.append(record)
    return {
        "dirty": bool(status_lines),
        "status": status_lines,
        "runtime_source_files": source_files,
    }


def _environment_record(args):
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for dependency in _DEPENDENCIES:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    try:
        import matplotlib

        backend = matplotlib.get_backend()
    except ImportError:
        backend = None
    try:
        import numba

        try:
            threading_layer = numba.threading_layer()
        except ValueError:
            threading_layer = "not initialized"
        numba_threads = numba.get_num_threads()
    except ImportError:
        threading_layer = None
        numba_threads = None
    try:
        from threadpoolctl import threadpool_info

        thread_pools = threadpool_info()
    except ImportError:
        thread_pools = []
    cgroup = {
        "memory_current": _read_optional_text("/sys/fs/cgroup/memory.current"),
        "memory_max": _read_optional_text("/sys/fs/cgroup/memory.max"),
        "memory_swap_current": _read_optional_text("/sys/fs/cgroup/memory.swap.current"),
        "memory_swap_max": _read_optional_text("/sys/fs/cgroup/memory.swap.max"),
        "cpuset": _read_optional_text("/sys/fs/cgroup/cpuset.cpus.effective"),
        "cpu_max": _read_optional_text("/sys/fs/cgroup/cpu.max"),
        "pids_max": _read_optional_text("/sys/fs/cgroup/pids.max"),
    }
    affinity = affinity_cpu_ids()
    environment = {
        "commit": commit,
        "working_tree": _working_tree_record(repository),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dependency_versions": versions,
        "matplotlib_backend": backend,
        "modest_image_available": importlib.util.find_spec("modest_image") is not None,
        "numpy_configuration": _numpy_configuration(),
        "numba_threading_layer": threading_layer,
        "numba_threads": numba_threads,
        "thread_pools": thread_pools,
        "cpu_affinity": affinity,
        "logical_cpus": len(affinity),
        "physical_cores": physical_core_count(affinity),
        "available_memory_bytes": available_memory_bytes(),
        "meminfo": _read_key_values("/proc/meminfo"),
        "cgroup": cgroup,
        "numa": {
            "nodes_online": _read_optional_text("/sys/devices/system/node/online"),
            "mems_allowed_list": _read_key_values("/proc/self/status").get("Mems_allowed_list"),
        },
        "function_bindings": [
            _binding_record(function)
            for function in (
                compress_eigerdata,
                mean_intensityc,
                Get_Pixel_Arrayc,
                auto_two_Arrayc,
                cal_g2p,
                get_one_time_from_two_time,
            )
        ],
    }
    if args.notebook_reference:
        environment["notebook"] = _file_record(args.notebook_reference, hash_contents=True)
    paths = {
        "cmp": args.cmp,
        "report": args.output or Path.cwd(),
        "artifacts": args.artifact_dir,
        "compression_input": args.eiger_master,
        "compression_output": args.compression_output,
    }
    environment["filesystems"] = {
        name: _filesystem_record(path) for name, path in paths.items() if path is not None
    }
    return environment


def _parse_stages(values, include_compression=False, include_artifacts=False):
    if not values or values == ["all"]:
        stages = list(_CORE_STAGES if include_compression else _CORE_STAGES[1:])
        if include_artifacts:
            stages.extend(_ARTIFACT_STAGES)
        return stages
    stages = []
    for value in values:
        for stage in value.split(","):
            stage = stage.strip()
            if stage not in _STAGES:
                raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(_STAGES)}")
            if stage not in stages:
                stages.append(stage)
    return stages


def _preflight(stage, args, roi_mask, pixel_count):
    frame_count = args.end - args.beg
    labels, roi_pixel_counts = np.unique(roi_mask[roi_mask > 0], return_counts=True)
    roi_count = labels.size
    max_roi_pixels = int(roi_pixel_counts.max()) if roi_pixel_counts.size else 0
    float_bytes = np.dtype(np.float64).itemsize
    selected_pixel_bytes = frame_count * pixel_count * np.dtype(np.float64).itemsize
    result_bytes = frame_count * frame_count * roi_count * np.dtype(np.float64).itemsize
    square_work_bytes = frame_count * frame_count * np.dtype(np.float64).itemsize
    available_memory = available_memory_bytes()
    batch_count = min(roi_count, 16, max(1, int(available_memory * 0.05) // max(1, square_work_bytes)))
    two_time_scratch = batch_count * square_work_bytes + frame_count * max_roi_pixels * float_bytes
    num_levels = max(1, int(np.log((frame_count + 1) / max(1, args.num_buf - 1)) / np.log(2) + 1) + 1)
    one_time_state = num_levels * args.num_buf * pixel_count * float_bytes
    one_time_state += num_levels * args.num_buf * roi_count * float_bytes
    one_time_block_frames = max(
        1,
        min(
            1024, min(256 * 1024**2, max(8 * pixel_count, int(available_memory * 0.10))) // max(8, 8 * pixel_count)
        ),
    )
    one_time_buffer_count = (
        2
        if args.one_time_double_buffer is not False
        and (args.one_time_double_buffer_schedule is None or 1 in args.one_time_double_buffer_schedule)
        else 1
    )
    if one_time_buffer_count == 2:
        one_time_block_frames = max(1, one_time_block_frames // 2)
    one_time_block = one_time_buffer_count * one_time_block_frames * pixel_count * float_bytes
    roi_result = frame_count * roi_count * float_bytes
    roi_setup = roi_mask.size * (np.dtype(np.int64).itemsize + np.dtype(np.int64).itemsize)
    plot_roi_pixels = frame_count * frame_count * float_bytes
    memory_components = {
        "selected-pixels": {"selected_pixel_output": selected_pixel_bytes},
        "roi-intensity": {"roi_result": roi_result, "roi_lookup_and_labels": roi_setup},
        "reader-reuse": {
            "roi_result": roi_result,
            "roi_lookup_and_labels": roi_setup,
            "one_time_state": one_time_state,
            "one_time_input_block": one_time_block,
        },
        "one-time": {"one_time_state": one_time_state, "one_time_input_block": one_time_block},
        "two-time": {
            "selected_pixel_input": selected_pixel_bytes,
            "two_time_output": result_bytes,
            "batched_square_and_roi_scratch": two_time_scratch,
        },
        "diagonal-means": {
            "selected_pixel_input": selected_pixel_bytes,
            "two_time_output": result_bytes,
            "batched_square_and_roi_scratch": two_time_scratch,
        },
        "roi-intensity-plot": {
            "roi_result": roi_result,
            "roi_lookup_and_labels": roi_setup,
            "rendered_rgba": frame_count * max(1, roi_count) * 4,
        },
        "g2-plot": {"one_time_state": one_time_state, "one_time_input_block": one_time_block},
        "g2-plot-reference": {},
        "direct-fit": {},
        "two-time-plot": {
            "selected_pixel_input": selected_pixel_bytes,
            "two_time_output": result_bytes,
            "batched_square_and_roi_scratch": two_time_scratch,
            "selected_roi_and_render": plot_roi_pixels * 2,
        },
        "export": {
            "selected_pixel_input": selected_pixel_bytes,
            "two_time_output": result_bytes,
            "batched_square_and_roi_scratch": two_time_scratch,
        },
        "export-raw": {
            "selected_pixel_input": selected_pixel_bytes,
            "two_time_output": result_bytes,
            "batched_square_and_roi_scratch": two_time_scratch,
        },
    }
    required_disk = 0
    if stage == "compression":
        frame_count = _eiger_frame_count(args.eiger_master)
        detector_pixels = roi_mask.size
        segment_count = int(np.ceil(frame_count / max(1, args.num_sub * args.compression_bins)))
        worker_count = min(segment_count, 500, len(affinity_cpu_ids()))
        per_worker = detector_pixels * (
            np.dtype(np.bool_).itemsize + 2 * float_bytes + max(args.compression_bytes, 4)
        )
        memory_components[stage] = {
            "worker_masks_accumulators_and_frames": worker_count * per_worker,
            "parent_reduction_arrays": detector_pixels * (np.dtype(np.bool_).itemsize + float_bytes),
        }
        required_disk = 1024 + frame_count * (
            4 + roi_mask.size * (np.dtype(np.int32).itemsize + args.compression_bytes)
        )
    elif stage in {"export", "export-raw"}:
        required_disk = result_bytes
    components = memory_components[stage]
    required_memory = sum(components.values())
    if stage == "compression":
        destination = args.compression_output
    elif stage in _ARTIFACT_REQUIRED_STAGES:
        destination = args.artifact_dir
    else:
        destination = args.output or args.cmp
    free_disk = shutil.disk_usage(_nearest_existing_path(destination)).free
    record = {
        "estimated_required_memory_bytes": required_memory,
        "estimated_memory_components": components,
        "estimated_memory_traffic_bytes": {
            "selected_pixel_input": (
                selected_pixel_bytes
                if stage in {"two-time", "diagonal-means", "two-time-plot", "export", "export-raw"}
                else 0
            ),
            "two_time_output": (
                result_bytes
                if stage in {"two-time", "diagonal-means", "two-time-plot", "export", "export-raw"}
                else 0
            ),
            "cmp_input": (
                args.cmp.stat().st_size
                if stage in {"roi-intensity", "reader-reuse", "one-time", "selected-pixels"}
                else 0
            ),
        },
        "two_time_batch_count": batch_count,
        "one_time_block_frames": one_time_block_frames,
        "available_memory_bytes": available_memory,
        "memory_limit_fraction": args.memory_fraction,
        "estimated_required_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
    }
    if args.skip_preflight:
        record["enforced"] = False
        return record
    record["enforced"] = True
    if required_memory > available_memory * args.memory_fraction:
        raise RuntimeError(
            f"{stage} needs an estimated {required_memory / 1024**3:.2f} GiB, exceeding "
            f"{args.memory_fraction:.0%} of {available_memory / 1024**3:.2f} GiB available; "
            "use a smaller workload or --skip-preflight after reviewing the risk"
        )
    if required_disk > free_disk:
        raise RuntimeError(
            f"{stage} may need up to {required_disk / 1024**3:.2f} GiB, but the destination has "
            f"{free_disk / 1024**3:.2f} GiB free"
        )
    return record


def _load_inputs(args):
    roi_mask = np.load(args.roi, allow_pickle=False)
    norm = np.load(args.norm, allow_pickle=False) if args.norm else None
    imgsum = np.load(args.imgsum, allow_pickle=False) if args.imgsum else None
    bad_frames = np.load(args.bad_frames, allow_pickle=False) if args.bad_frames else []
    pixel_list = np.flatnonzero(roi_mask.ravel() > 0)
    return roi_mask, norm, imgsum, bad_frames, pixel_list


def _eiger_frame_count(master_filename):
    """Return the number of frames linked below an Eiger master data group."""
    with h5py.File(master_filename, "r") as h5file:
        return sum(dataset.shape[0] for dataset in h5file["entry/data"].values())


def _buffered_read_record(read_stats):
    cumulative_bytes = 0
    reads = []
    for offset, byte_count, seconds in read_stats:
        cumulative_bytes += byte_count
        reads.append(
            {
                "offset": offset,
                "bytes": byte_count,
                "seconds": seconds,
                "cumulative_bytes": cumulative_bytes,
            }
        )
    latencies = np.asarray([read["seconds"] for read in reads], dtype=np.float64)
    return {
        "count": len(reads),
        "bytes": cumulative_bytes,
        "seconds": float(latencies.sum()),
        "median_seconds": float(np.median(latencies)) if latencies.size else 0.0,
        "maximum_seconds": float(latencies.max()) if latencies.size else 0.0,
        "reads": reads,
    }


def _compression_destination(args, execution_mode, trial_number):
    base = args.compression_output
    suffix = base.suffix or ".cmp"
    return base.with_name(f"{base.stem}-{args.run_id}-{execution_mode}-trial-{trial_number}{suffix}")


def _artifact_trial_directory(args, stage, execution_mode, trial_number):
    directory = args.artifact_dir / args.run_id / stage / f"{execution_mode}-trial-{trial_number}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _artifact_records(directory, hash_contents=True):
    return [
        _file_record(path, hash_contents=hash_contents) for path in sorted(directory.iterdir()) if path.is_file()
    ]


def _hdf5_dataset_record(filename, key):
    """Record layout properties without reading a potentially huge dataset."""
    with h5py.File(filename, "r") as h5file:
        dataset = h5file[key]
        fill_value = dataset.fillvalue
        if isinstance(fill_value, np.generic):
            fill_value = fill_value.item()
        return {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "chunks": None if dataset.chunks is None else list(dataset.chunks),
            "compression": dataset.compression,
            "compression_options": dataset.compression_opts,
            "shuffle": dataset.shuffle,
            "fletcher32": dataset.fletcher32,
            "fill_value": fill_value,
        }


def _write_raw_contiguous_export(filename, key, value):
    """Write a benchmark-owned contiguous HDF5 baseline with timed boundaries."""
    components = {name: {"count": 0, "seconds": 0.0} for name in ("open", "array_write", "flush", "close")}

    def timed(name, function):
        started = time.perf_counter()
        try:
            return function()
        finally:
            components[name]["seconds"] += time.perf_counter() - started
            components[name]["count"] += 1

    h5file = timed("open", lambda: h5py.File(filename, "w"))
    try:
        timed("array_write", lambda: h5file.create_dataset(key, data=value))
        timed("flush", h5file.flush)
    finally:
        timed("close", h5file.close)
    return components


def _load_reference_g2(filename):
    """Load the small arrays needed to reproduce the three notebook g2 plots."""
    with h5py.File(filename, "r") as h5file:
        values = {key: h5file[key][:] for key in ("g2", "g2b", "taus", "tausb")}
        q_values = {
            int(key): np.asarray(value)
            for key, value in sorted(h5file["qval_dict"].attrs.items(), key=lambda item: int(item[0]))
        }
        fit_data = {}
        for key in ("g2_fit_paras", "g2b_fit_paras"):
            group = h5file[key]
            columns = [item.decode() for item in group["block0_items"][:]]
            fit_data[key] = [
                SimpleNamespace(best_values=dict(zip(columns, row))) for row in group["block0_values"][:]
            ]
    values["q_values"] = q_values
    values.update(fit_data)
    return values


def _stretched_fit_curves(taus, fit_results):
    curves = np.empty((len(taus), len(fit_results)), dtype=np.float64)
    for index, fit_result in enumerate(fit_results):
        parameters = fit_result.best_values
        curves[:, index] = parameters["baseline"] + parameters["beta"] * np.exp(
            -2 * (parameters["relaxation_rate"] * taus) ** parameters["alpha"]
        )
    return curves


def _load_selected_pixels(args, pixel_list, norm):
    with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
        compressed._reset_io_counters()
        data, metric = _measure(
            "selected-pixels-prerequisite",
            lambda: Get_Pixel_Arrayc(compressed, pixel_list, norm=norm).get_data(),
        )
        metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
    return data, metric


def _calculate_one_time(args, roi_mask, norm, imgsum, bad_frames):
    with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
        compressed._reset_io_counters()
        result, metric = _measure(
            "one-time-prerequisite",
            lambda: cal_g2p(
                compressed,
                roi_mask,
                bad_frame_list=bad_frames,
                num_buf=args.num_buf,
                imgsum=imgsum,
                norm=norm,
            ),
        )
        metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
    return result, metric


def _run_stage(stage, args, execution_mode, trial_number):
    roi_mask, norm, imgsum, bad_frames, pixel_list = _load_inputs(args)
    preflight = _preflight(stage, args, roi_mask, pixel_list.size)
    prefetch_schedule = getattr(args, "cmp_prefetch_schedule", None)
    cmp_prefetch = (
        bool(prefetch_schedule[(trial_number - 1) % len(prefetch_schedule)])
        if prefetch_schedule
        else args.cmp_prefetch
    )
    reader_dispatch_schedule = getattr(args, "reader_reuse_dispatch_schedule", None)
    reader_reuse_dispatch = (
        reader_dispatch_schedule[(trial_number - 1) % len(reader_dispatch_schedule)]
        if reader_dispatch_schedule
        else args.reader_reuse_dispatch
    )
    one_time_buffer_schedule = getattr(args, "one_time_double_buffer_schedule", None)
    one_time_double_buffer = (
        bool(one_time_buffer_schedule[(trial_number - 1) % len(one_time_buffer_schedule)])
        if one_time_buffer_schedule
        else args.one_time_double_buffer
    )
    if stage == "compression":
        detector_mask = np.load(args.mask, allow_pickle=False)
        metadata = json.loads(args.metadata.read_text())
        metadata["pixel_mask"] = detector_mask.copy()
        frame_count = _eiger_frame_count(args.eiger_master)
        destination = _compression_destination(args, execution_mode, trial_number)
        raw_bytes = frame_count * detector_mask.size * np.dtype(np.uint32).itemsize

        def run_compression():
            return _call_with_component_timers(
                lambda: compress_eigerdata(
                    np.empty(frame_count, dtype=np.uint8),
                    detector_mask.copy(),
                    metadata.copy(),
                    os.fspath(destination),
                    force_compress=True,
                    para_compress=True,
                    num_sub=args.num_sub,
                    bins=args.compression_bins,
                    nobytes=args.compression_bytes,
                    bad_pixel_threshold=args.bad_pixel_threshold,
                    bad_pixel_low_threshold=args.bad_pixel_low_threshold,
                    hot_pixel_threshold=args.hot_pixel_threshold,
                    dtypes="uid",
                    with_pickle=args.compression_with_pickle,
                    direct_load_data=True,
                    data_path=os.fspath(args.eiger_master),
                    images_per_file=args.images_per_file,
                    copy_rawdata=False,
                    reverse=not args.no_reverse,
                    rot90=args.rot90,
                    func_images_per_file=lambda _path: args.images_per_file,
                ),
                (
                    ("raw_staging", chx_compress, "copy_data"),
                    ("header", chx_compress, "create_compress_header"),
                    ("segments_and_reduction", chx_compress, "_iter_parallel_segment_results"),
                    ("cmp_publication", chx_compress, "_publish_compressed_segments"),
                    ("companion_publication", chx_compress, "_publish_file"),
                ),
            )

        (result, components), metric = _measure(stage, run_compression, raw_bytes)
        metric["components"] = components
        hash_output = args.compression_reference is not None
        metric["output"] = _file_record(destination, hash_contents=hash_output)
        if args.compression_with_pickle:
            metric["companion_output"] = _file_record(
                Path(os.fspath(destination) + ".pkl"), hash_contents=hash_output
            )
        if args.compression_reference is not None:
            reference = _file_record(args.compression_reference, hash_contents=True)
            metric["reference"] = reference
            metric["byte_identical"] = (
                metric["output"]["size_bytes"] == reference["size_bytes"]
                and metric["output"]["sha256"] == reference["sha256"]
            )
            if args.compression_with_pickle:
                companion_reference_path = Path(os.fspath(args.compression_reference) + ".pkl")
                companion_reference = _file_record(companion_reference_path, hash_contents=True)
                metric["companion_reference"] = companion_reference
                metric["companion_byte_identical"] = (
                    metric["companion_output"]["size_bytes"] == companion_reference["size_bytes"]
                    and metric["companion_output"]["sha256"] == companion_reference["sha256"]
                )
    elif stage == "roi-intensity":
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            compressed._buffered_read_stats = []
            compressed._roi_intensity_stats = {}
            compressed._buffered_read_block_size = int(args.cmp_read_block_mib * 1024**2)
            if cmp_prefetch is not None:
                compressed._roi_intensity_prefetch = cmp_prefetch
            result, metric = _measure(
                stage,
                lambda: mean_intensityc(compressed, roi_mask, sampling=args.sampling),
            )
            metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
            metric["components"] = compressed._roi_intensity_stats
            metric["components"]["prefetch"] = getattr(compressed, "_roi_intensity_prefetch", True)
            metric["buffered_reads"] = _buffered_read_record(compressed._buffered_read_stats)
    elif stage == "selected-pixels":
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            result, metric = _measure(
                stage,
                lambda: Get_Pixel_Arrayc(compressed, pixel_list, norm=norm).get_data(),
            )
            metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
    elif stage == "one-time":
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            compressed._one_time_stats = {}
            compressed._buffered_read_stats = []
            compressed._buffered_read_block_size = int(args.cmp_read_block_mib * 1024**2)
            if one_time_double_buffer is not None:
                compressed._one_time_double_buffer = one_time_double_buffer
            if cmp_prefetch is not None:
                compressed._roi_intensity_prefetch = cmp_prefetch
            result, metric = _measure(
                stage,
                lambda: cal_g2p(
                    compressed,
                    roi_mask,
                    bad_frame_list=bad_frames,
                    num_buf=args.num_buf,
                    imgsum=imgsum,
                    norm=norm,
                ),
            )
            metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
            metric["components"] = compressed._one_time_stats
            metric["buffered_reads"] = _buffered_read_record(compressed._buffered_read_stats)
    elif stage == "reader-reuse":
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            compressed._roi_intensity_stats = {}
            compressed._one_time_stats = {}
            compressed._buffered_read_stats = []
            compressed._buffered_read_block_size = int(args.cmp_read_block_mib * 1024**2)
            if one_time_double_buffer is not None:
                compressed._one_time_double_buffer = one_time_double_buffer
            if cmp_prefetch is not None:
                compressed._roi_intensity_prefetch = cmp_prefetch
            reader_state = {"beg_before": compressed.beg, "end_before": compressed.end}

            def run_reused_reader():
                intensity_started = time.perf_counter()
                intensity_result = mean_intensityc(compressed, roi_mask, sampling=args.sampling)
                intensity_seconds = time.perf_counter() - intensity_started
                if reader_reuse_dispatch == "buffered":
                    compressed._last_buffered_scan = None
                    compressed._one_time_use_buffered_reader = True
                elif reader_reuse_dispatch == "indexed":
                    compressed._one_time_use_buffered_reader = False
                one_time_started = time.perf_counter()
                one_time_result = cal_g2p(
                    compressed,
                    roi_mask,
                    bad_frame_list=bad_frames,
                    num_buf=args.num_buf,
                    imgsum=imgsum,
                    norm=norm,
                )
                return intensity_result, one_time_result, intensity_seconds, time.perf_counter() - one_time_started

            result, metric = _measure(stage, run_reused_reader)
            reader_state.update(
                {
                    "beg_after": compressed.beg,
                    "end_after": compressed.end,
                    "requested_dispatch": reader_reuse_dispatch,
                    "effective_one_time_dispatch": compressed._one_time_stats.get("reader"),
                    "completed_buffered_scan": (
                        list(compressed._last_buffered_scan)
                        if getattr(compressed, "_last_buffered_scan", None) is not None
                        else None
                    ),
                }
            )
            metric["components"] = {
                "roi_intensity": compressed._roi_intensity_stats,
                "one_time": compressed._one_time_stats,
                "roi_intensity_seconds": result[2],
                "one_time_seconds": result[3],
            }
            metric["reader_state"] = reader_state
            metric["cmp_logical_bytes_traversed"] = compressed._bytes_traversed
            metric["buffered_reads"] = _buffered_read_record(compressed._buffered_read_stats)
    elif stage == "direct-fit":
        reference = _load_reference_g2(args.reference_results)
        fit_kwargs = {
            "function": "stretched",
            "fit_range": None,
            "fit_variables": {
                "baseline": False,
                "beta": True,
                "alpha": True,
                "relaxation_rate": True,
            },
            "guess_values": {
                "baseline": 1.0,
                "beta": 0.2,
                "alpha": 1.0,
                "relaxation_rate": 1e-3,
            },
            "guess_limits": {
                "baseline": [0.9, 2],
                "alpha": [0, 2],
                "beta": [0.01, 0.4],
                "relaxation_rate": [1e-7, 1e7],
            },
        }

        result, metric = _measure(
            stage,
            lambda: chx_generic_functions.get_g2_fit_general(reference["g2"], reference["taus"], **fit_kwargs),
            reference["g2"].nbytes,
        )
        fit_results, fit_taus, fitted = result
        parameter_names = sorted(fit_results[0].best_values)
        parameter_values = np.asarray(
            [[fit_result.best_values[name] for name in parameter_names] for fit_result in fit_results],
            dtype=np.float64,
        )
        metric["parameter_names"] = parameter_names
        metric["parameter_values_sha256"] = hashlib.sha256(parameter_values.tobytes()).hexdigest()
        metric["fitted_values_sha256"] = hashlib.sha256(np.ascontiguousarray(fitted).tobytes()).hexdigest()
        metric["fit_taus_sha256"] = hashlib.sha256(np.ascontiguousarray(fit_taus).tobytes()).hexdigest()
    elif stage in {"two-time", "diagonal-means"}:
        data, selected_pixel_metric = _load_selected_pixels(args, pixel_list, norm)
        if stage == "two-time":
            thread_schedule = getattr(args, "two_time_thread_schedule", None)
            thread_count = (
                thread_schedule[(trial_number - 1) % len(thread_schedule)]
                if thread_schedule
                else getattr(args, "two_time_threads", None)
            )

            def calculate_two_time():
                blas_thread_count = getattr(args, "two_time_blas_threads", None)
                thread_override = (
                    mock.patch.object(chx_correlationc, "physical_core_count", return_value=thread_count)
                    if thread_count is not None
                    else contextlib.nullcontext()
                )
                original_upper_product = chx_correlationc._upper_two_time_product

                def upper_product_with_thread_limit(*call_args, **call_kwargs):
                    with chx_correlationc.threadpool_limits(limits=blas_thread_count, user_api="blas"):
                        return original_upper_product(*call_args, **call_kwargs)

                blas_override = (
                    mock.patch.object(
                        chx_correlationc,
                        "_upper_two_time_product",
                        upper_product_with_thread_limit,
                    )
                    if blas_thread_count is not None
                    else contextlib.nullcontext()
                )
                with thread_override, blas_override:
                    return _call_with_component_timers(
                        lambda: auto_two_Arrayc(data, roi_mask),
                        (
                            ("roi_gather_and_normalize", chx_correlationc, "_prepare_two_time_roi"),
                            ("symmetric_blas", chx_correlationc, "_upper_two_time_product"),
                            ("symmetric_scatter", chx_correlationc, "store_symmetric_two_time_batch"),
                        ),
                    )

            (result, components), metric = _measure(stage, calculate_two_time, data.nbytes)
            metric["components"] = components
            metric["two_time_thread_count"] = thread_count
        else:
            two_time, two_time_metric = _measure(
                "two-time-prerequisite",
                lambda: auto_two_Arrayc(data, roi_mask),
                data.nbytes,
            )
            result, metric = _measure(
                stage,
                lambda: get_one_time_from_two_time(two_time),
                two_time.nbytes,
            )
            metric["two_time_prerequisite"] = two_time_metric
            del two_time
        metric["selected_pixels_prerequisite"] = selected_pixel_metric
        del data
    elif stage == "roi-intensity-plot":
        artifact_dir = _artifact_trial_directory(args, stage, execution_mode, trial_number)
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            intensity_result, prerequisite = _measure(
                "roi-intensity-prerequisite",
                lambda: mean_intensityc(compressed, roi_mask, sampling=args.sampling),
            )
        intensities, _ = intensity_result
        times = args.beg + np.arange(intensities.shape[0]) * args.sampling

        def plot_intensities():
            return _call_with_component_timers(
                lambda: plot_each_ring_mean_intensityc(
                    times,
                    intensities,
                    save=True,
                    uid="benchmark",
                    path=os.fspath(artifact_dir) + os.sep,
                ),
                (
                    ("render_encode_write", Figure, "savefig"),
                    ("csv_write", chx_compress_analysis, "save_arrays"),
                ),
            )

        (result, components), metric = _measure(stage, plot_intensities, intensities.nbytes)
        metric["components"] = components
        metric["prerequisite"] = prerequisite
        metric["artifacts"] = _artifact_records(artifact_dir)
        plt.close("all")
        del intensities
    elif stage == "g2-plot":
        artifact_dir = _artifact_trial_directory(args, stage, execution_mode, trial_number)
        (g2, lag_steps), prerequisite = _calculate_one_time(args, roi_mask, norm, imgsum, bad_frames)
        q_values = {index: np.asarray([index], dtype=np.float64) for index in range(g2.shape[1])}

        def plot_g2():
            return _call_with_component_timers(
                lambda: chx_generic_functions.plot_g2_general(
                    {1: g2},
                    {1: lag_steps},
                    q_values,
                    filename="benchmark_g2",
                    path=os.fspath(artifact_dir) + os.sep,
                    return_fig=True,
                ),
                (
                    ("layout", chx_generic_functions, "_adjust_g2_page_layout"),
                    ("render_encode_write", Figure, "savefig"),
                    ("draw", Figure, "draw"),
                    ("montage", chx_generic_functions, "combine_images"),
                ),
            )

        (result, components), metric = _measure(stage, plot_g2, g2.nbytes)
        metric["components"] = components
        metric["prerequisite"] = prerequisite
        metric["artifacts"] = _artifact_records(artifact_dir)
        plt.close("all")
        del g2, lag_steps
    elif stage == "g2-plot-reference":
        artifact_dir = _artifact_trial_directory(args, stage, execution_mode, trial_number)
        reference = _load_reference_g2(args.reference_results)
        g2_fit = _stretched_fit_curves(reference["taus"], reference["g2_fit_paras"])
        g2b_fit = _stretched_fit_curves(reference["tausb"], reference["g2b_fit_paras"])

        def render_plot_suite():
            call_seconds = {}

            def timed_plot(name, *plot_args, **plot_kwargs):
                started = time.perf_counter()
                try:
                    return chx_generic_functions.plot_g2_general(*plot_args, **plot_kwargs)
                finally:
                    call_seconds[name] = time.perf_counter() - started

            common = {
                "qval_dict": reference["q_values"],
                "geometry": "ang_saxs",
                "path": os.fspath(artifact_dir) + os.sep,
                "ylabel": "g2",
                "return_fig": True,
            }
            timed_plot(
                "direct_fit",
                {1: reference["g2"], 2: g2_fit},
                {1: reference["taus"], 2: reference["taus"]},
                fit_res=reference["g2_fit_paras"],
                function="stretched",
                filename="benchmark_g2",
                append_name="_fit",
                vlim=[0.95, 1.05],
                **common,
            )
            timed_plot(
                "derived_fit",
                {1: reference["g2b"], 2: g2b_fit},
                {1: reference["tausb"], 2: reference["tausb"]},
                fit_res=reference["g2b_fit_paras"],
                function="stretched",
                filename="benchmark_g2",
                append_name="_b_fit",
                vlim=[0.95, 1.05],
                **common,
            )
            timed_plot(
                "comparison",
                {1: reference["g2"], 2: reference["g2b"]},
                {1: reference["taus"], 2: reference["tausb"]},
                g2_labels=["from_one_time", "from_two_time"],
                filename="benchmark_g2_two_g2",
                vlim=[0.99, 1.007],
                **common,
            )
            return call_seconds

        def plot_g2_suite():
            return _call_with_component_timers(
                render_plot_suite,
                (
                    ("layout", chx_generic_functions, "_adjust_g2_page_layout"),
                    ("render_encode_write", Figure, "savefig"),
                    ("draw", Figure, "draw"),
                    ("montage", chx_generic_functions, "combine_images"),
                ),
            )

        logical_bytes = sum(reference[key].nbytes for key in ("g2", "g2b", "taus", "tausb"))
        (plot_calls, components), metric = _measure(stage, plot_g2_suite, logical_bytes)
        metric["plot_calls"] = plot_calls
        metric["components"] = components
        metric["reference_results"] = _file_record(args.reference_results)
        metric["artifacts"] = _artifact_records(artifact_dir)
        plt.close("all")
        result = None
    elif stage in {"two-time-plot", "export", "export-raw"}:
        artifact_dir = _artifact_trial_directory(args, stage, execution_mode, trial_number)
        data, selected_pixel_metric = _load_selected_pixels(args, pixel_list, norm)
        two_time, two_time_metric = _measure(
            "two-time-prerequisite",
            lambda: auto_two_Arrayc(data, roi_mask),
            data.nbytes,
        )
        if stage == "two-time-plot":

            def plot_two_time():
                return _call_with_component_timers(
                    lambda: show_C12(
                        two_time,
                        q_ind=args.plot_q_index,
                        save=True,
                        uid="benchmark",
                        path=os.fspath(artifact_dir) + os.sep,
                        return_fig=True,
                    ),
                    (("render_encode_write", Figure, "savefig"),),
                )

            plotted_roi_bytes = two_time.shape[0] * two_time.shape[1] * two_time.dtype.itemsize
            (result, components), metric = _measure(stage, plot_two_time, plotted_roi_bytes)
            metric["components"] = components
            plt.close("all")
        elif stage == "export":
            filename = "benchmark_results.h5"
            output_path = artifact_dir / filename

            def export_two_time():
                return _call_with_component_timers(
                    lambda: export_xpcs_results_to_h5(
                        filename,
                        os.fspath(artifact_dir) + os.sep,
                        {"g12b": two_time},
                    ),
                    (
                        ("metadata_write", Create_Report, "_write_xpcs_metadata"),
                        ("array_write", Create_Report, "_write_xpcs_array"),
                        ("dataframe_write", Create_Report, "_write_xpcs_dataframe"),
                        ("open", Create_Report, "_open_xpcs_h5"),
                        ("close", Create_Report, "_close_xpcs_h5"),
                    ),
                )

            (result, components), metric = _measure(stage, export_two_time, two_time.nbytes)
            metric["components"] = components
            metric["durability"] = {"explicit_hdf5_flush": False, "fsync": False}
            metric["dataset"] = _hdf5_dataset_record(output_path, "g12b")
        else:
            output_path = artifact_dir / "benchmark_raw_contiguous.h5"

            def export_raw_two_time():
                return _write_raw_contiguous_export(output_path, "g12b", two_time)

            components, metric = _measure(stage, export_raw_two_time, two_time.nbytes)
            result = None
            metric["components"] = components
            metric["durability"] = {"explicit_hdf5_flush": True, "fsync": False}
            metric["dataset"] = _hdf5_dataset_record(output_path, "g12b")
        metric["selected_pixels_prerequisite"] = selected_pixel_metric
        metric["two_time_prerequisite"] = two_time_metric
        metric["artifacts"] = _artifact_records(
            artifact_dir,
            hash_contents=stage not in {"export", "export-raw"},
        )
        del data, two_time
    else:  # pragma: no cover - stage validation rejects this path
        raise ValueError(f"unsupported stage {stage!r}")
    metric.update(
        {
            "stage": stage,
            "trial": trial_number,
            "execution_mode": execution_mode,
            "cache_state": args.cache_state_label,
            "preflight": preflight,
        }
    )
    del result
    gc.collect()
    return metric


def _child_entry(connection, stages, arguments, execution_mode, repetitions, trial_start):
    try:
        args = argparse.Namespace(**arguments)
        environment = _environment_record(args)
        measurements = []
        for stage in stages:
            for trial_number in range(trial_start, trial_start + repetitions):
                measurements.append(_run_stage(stage, args, execution_mode, trial_number))
        connection.send({"measurements": measurements, "environment": environment})
    except Exception as error:  # noqa: BLE001 - forward worker failures to the parent
        connection.send(
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def _run_child(stages, args, execution_mode, repetitions, trial_start=1):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry,
        args=(child, stages, vars(args), execution_mode, repetitions, trial_start),
    )
    process.start()
    child.close()
    message = parent.recv()
    parent.close()
    process.join()
    if process.exitcode != 0 and "error" not in message:
        raise RuntimeError(f"benchmark worker exited with status {process.exitcode}")
    if "error" in message:
        raise RuntimeError(f"{message['error']}\n{message['traceback']}")
    return message


def _input_records(args):
    records = {
        "cmp": _file_record(args.cmp, hash_contents=args.hash_large_inputs),
        "roi": _file_record(args.roi, hash_contents=True),
    }
    for name in (
        "norm",
        "imgsum",
        "bad_frames",
        "reference_results",
        "eiger_master",
        "mask",
        "metadata",
        "compression_reference",
    ):
        path = getattr(args, name)
        if path:
            records[name] = _file_record(
                path,
                hash_contents=args.hash_large_inputs
                or name not in {"eiger_master", "reference_results", "compression_reference"},
            )
    return records


def _json_options(args, stages):
    excluded = {"output"}
    options = {key: value for key, value in vars(args).items() if key not in excluded}
    for key, value in options.items():
        if isinstance(value, Path):
            options[key] = os.fspath(value.resolve())
    options["stages"] = stages
    return options


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmp", required=True, type=Path)
    parser.add_argument("--roi", required=True, type=Path, help="NumPy .npy ROI-label mask")
    parser.add_argument("--norm", type=Path, help="Optional NumPy .npy per-pixel normalization")
    parser.add_argument("--imgsum", type=Path, help="Optional NumPy .npy frame-intensity normalization")
    parser.add_argument("--bad-frames", type=Path, help="Optional NumPy .npy bad-frame indices")
    parser.add_argument(
        "--reference-results",
        type=Path,
        help="Existing result HDF5 used by the opt-in representative g2 plotting stage",
    )
    parser.add_argument("--beg", type=int, default=0)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--num-buf", type=int, default=8)
    parser.add_argument(
        "--one-time-double-buffer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Benchmark-only override for overlapping one-time population and correlation",
    )
    parser.add_argument(
        "--one-time-double-buffer-schedule",
        type=int,
        choices=(0, 1),
        nargs="+",
        help="Benchmark-only per-trial one-time double-buffer states",
    )
    parser.add_argument(
        "--two-time-threads",
        type=int,
        help="Experimental exact two-time BLAS/Numba thread count; does not alter package defaults",
    )
    parser.add_argument(
        "--two-time-thread-schedule",
        type=int,
        nargs="+",
        help="Experimental per-trial exact thread counts, repeated cyclically",
    )
    parser.add_argument(
        "--two-time-blas-threads",
        type=int,
        help="Experimental per-product BLAS limit within the two-time thread limit",
    )
    parser.add_argument("--sampling", type=int, default=1, help="ROI-intensity frame sampling")
    parser.add_argument(
        "--reader-reuse-dispatch",
        choices=("automatic", "buffered", "indexed"),
        default="automatic",
        help="Benchmark-only dispatch selection for the reader-reuse stage",
    )
    parser.add_argument(
        "--reader-reuse-dispatch-schedule",
        choices=("automatic", "buffered", "indexed"),
        nargs="+",
        help="Benchmark-only per-trial reader dispatch modes, repeated cyclically",
    )
    parser.add_argument(
        "--cmp-read-block-mib",
        type=float,
        default=8,
        help="Private ROI-intensity sequential-read block size in MiB",
    )
    parser.add_argument(
        "--cmp-prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the internal bounded producer for sequential CMP ROI reads",
    )
    parser.add_argument(
        "--cmp-prefetch-schedule",
        type=int,
        choices=(0, 1),
        nargs="+",
        help="Benchmark-only per-trial prefetch states, repeated cyclically",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Write the JSON report without echoing it")
    parser.add_argument("--run-id", help="Unique benchmark run identifier used below artifact destinations")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Destination for opt-in plotting and export stage artifacts",
    )
    parser.add_argument("--plot-q-index", type=int, default=1)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=f"Stages or comma-separated stages ({', '.join(_STAGES)}; default: all available)",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--execution-mode",
        choices=("fresh-process", "same-process"),
        default="fresh-process",
        help="Use one worker per trial or repeat a stage within one worker",
    )
    parser.add_argument(
        "--cache-state-label",
        default="unknown",
        help="Externally characterized source-cache state; the harness never drops host caches",
    )
    parser.add_argument("--notebook-reference", type=Path, help="Read-only notebook whose SHA-256 is recorded")
    parser.add_argument("--hash-large-inputs", action="store_true", help="SHA-256 the CMP and Eiger master")
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--eiger-master", type=Path, help="Also benchmark compression from this Eiger master")
    parser.add_argument("--mask", type=Path, help="Detector mask required with --eiger-master")
    parser.add_argument("--metadata", type=Path, help="Compression metadata JSON required with --eiger-master")
    parser.add_argument("--compression-output", type=Path, help="Base name for benchmark CMP output")
    parser.add_argument("--compression-reference", type=Path, help="Optional byte-identity CMP reference")
    parser.add_argument(
        "--compression-with-pickle",
        action="store_true",
        help="Also produce and validate the compression companion pickle",
    )
    parser.add_argument("--images-per-file", type=int, default=100)
    parser.add_argument("--num-sub", type=int, default=128)
    parser.add_argument("--compression-bins", type=int, default=1)
    parser.add_argument("--compression-bytes", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--bad-pixel-threshold", type=float, default=1e15)
    parser.add_argument("--bad-pixel-low-threshold", type=float, default=0)
    parser.add_argument("--hot-pixel-threshold", type=float, default=2**30)
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument("--rot90", action="store_true")
    return parser


def _validate_arguments(parser, args, stages):
    if args.beg < 0 or args.end <= args.beg:
        parser.error("--end must be greater than nonnegative --beg")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.two_time_threads is not None and args.two_time_threads < 1:
        parser.error("--two-time-threads must be positive")
    if args.two_time_thread_schedule is not None and any(value < 1 for value in args.two_time_thread_schedule):
        parser.error("--two-time-thread-schedule values must be positive")
    if args.two_time_threads is not None and args.two_time_thread_schedule is not None:
        parser.error("--two-time-threads and --two-time-thread-schedule are mutually exclusive")
    if args.one_time_double_buffer is not None and args.one_time_double_buffer_schedule is not None:
        parser.error("--one-time-double-buffer and --one-time-double-buffer-schedule are mutually exclusive")
    if args.two_time_blas_threads is not None and args.two_time_blas_threads < 1:
        parser.error("--two-time-blas-threads must be positive")
    if args.sampling < 1:
        parser.error("--sampling must be positive")
    if args.cmp_read_block_mib <= 0:
        parser.error("--cmp-read-block-mib must be positive")
    if args.cmp_prefetch is not None and args.cmp_prefetch_schedule is not None:
        parser.error("--cmp-prefetch and --cmp-prefetch-schedule are mutually exclusive")
    if args.plot_q_index < 1:
        parser.error("--plot-q-index must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")
    if "compression" in stages and not all((args.eiger_master, args.mask, args.metadata, args.compression_output)):
        parser.error("the compression stage requires --eiger-master, --mask, --metadata, and --compression-output")
    if any(stage in {"direct-fit", "g2-plot-reference"} for stage in stages) and args.reference_results is None:
        parser.error("the direct-fit and g2-plot-reference stages require --reference-results")
    if any(stage in _ARTIFACT_REQUIRED_STAGES for stage in stages) and args.artifact_dir is None:
        parser.error("plotting and export stages require --artifact-dir")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        stages = _parse_stages(
            args.stages,
            include_compression=args.eiger_master is not None,
            include_artifacts=args.artifact_dir is not None,
        )
    except ValueError as error:
        parser.error(str(error))
    _validate_arguments(parser, args, stages)

    if args.run_id is None:
        args.run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{time.time_ns()}"
    controller_environment = _environment_record(args)

    measurements = []
    worker_environments = []
    if args.execution_mode == "fresh-process":
        for stage in stages:
            for trial_number in range(1, args.repetitions + 1):
                result = _run_child([stage], args, "fresh-process", 1, trial_start=trial_number)
                measurements.extend(result["measurements"])
                worker_environments.append(result["environment"])
    else:
        result = _run_child(stages, args, "same-process", args.repetitions)
        measurements.extend(result["measurements"])
        worker_environments.append(result["environment"])

    report = {
        "schema_version": 1,
        "environment": worker_environments[0],
        "controller_environment": controller_environment,
        "worker_environments": worker_environments,
        "options": _json_options(args, stages),
        "inputs": _input_records(args),
        "filesystems": {
            "source": _filesystem_record(args.cmp),
            "report_destination": _filesystem_record(args.output or Path.cwd()),
        },
        "measurements": measurements,
    }
    report_text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(report_text + "\n")
    if not args.quiet:
        print(report_text)


if __name__ == "__main__":
    main()
