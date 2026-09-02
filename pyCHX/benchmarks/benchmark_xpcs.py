"""Opt-in benchmark for the production XPCS compressed-data call path.

This module never runs as part of the test suite.  Invoke it with
``python -m pyCHX.benchmarks.benchmark_xpcs --help``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import threading
import time
from pathlib import Path

import numpy as np

from pyCHX.chx_compress import Multifile, compress_eigerdata
from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc
from pyCHX.chx_correlationp import cal_g2p
from pyCHX.Two_Time_Correlation_Function import get_one_time_from_two_time


def _rss_bytes():
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _process_rss_and_threads(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "Threads:")):
                key, value, *_ = line.split()
                fields[key.rstrip(":")] = int(value)
        return fields.get("VmRSS", 0) * 1024, fields.get("Threads", 0)
    except OSError:
        return 0, 0


def _measure(name, function, traversed_bytes):
    before_rss = _rss_bytes()
    peaks = {"rss": before_rss, "threads": threading.active_count(), "processes": 1}
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.02):
            pids = [os.getpid(), *(child.pid for child in multiprocessing.active_children())]
            snapshots = [_process_rss_and_threads(pid) for pid in pids]
            peaks["rss"] = max(peaks["rss"], sum(item[0] for item in snapshots))
            peaks["threads"] = max(peaks["threads"], sum(item[1] for item in snapshots))
            peaks["processes"] = max(peaks["processes"], len(pids))

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        monitor_thread.join()
    return result, {
        "name": name,
        "seconds": elapsed,
        "peak_rss_bytes": max(peaks["rss"], _rss_bytes()),
        "rss_before_bytes": before_rss,
        "peak_threads": peaks["threads"],
        "peak_processes": peaks["processes"],
        "bytes_traversed": traversed_bytes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmp", required=True, type=Path)
    parser.add_argument("--roi", required=True, type=Path, help="NumPy .npy ROI-label mask")
    parser.add_argument("--norm", type=Path, help="Optional NumPy .npy per-pixel normalization")
    parser.add_argument("--imgsum", type=Path, help="Optional NumPy .npy frame-intensity normalization")
    parser.add_argument("--bad-frames", type=Path, help="Optional NumPy .npy bad-frame indices")
    parser.add_argument("--beg", type=int, default=0)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--num-buf", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eiger-master", type=Path, help="Also benchmark compression from this Eiger master")
    parser.add_argument("--mask", type=Path, help="Detector mask required with --eiger-master")
    parser.add_argument("--metadata", type=Path, help="Compression metadata JSON required with --eiger-master")
    parser.add_argument("--compression-output", type=Path, help="Base name for temporary benchmark CMP output")
    parser.add_argument("--images-per-file", type=int, default=100)
    parser.add_argument("--compression-frames", type=int)
    parser.add_argument("--num-sub", type=int, default=128)
    parser.add_argument("--compression-bins", type=int, default=1)
    parser.add_argument("--compression-bytes", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--bad-pixel-threshold", type=float, default=1e15)
    parser.add_argument("--bad-pixel-low-threshold", type=float, default=0)
    parser.add_argument("--hot-pixel-threshold", type=float, default=2**30)
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument("--rot90", action="store_true")
    args = parser.parse_args(argv)

    roi_mask = np.load(args.roi, allow_pickle=False)
    norm = np.load(args.norm, allow_pickle=False) if args.norm else None
    imgsum = np.load(args.imgsum, allow_pickle=False) if args.imgsum else None
    bad_frames = np.load(args.bad_frames, allow_pickle=False) if args.bad_frames else []
    pixel_list = np.flatnonzero(roi_mask.ravel() > 0)
    measurements = []

    if args.eiger_master:
        if not args.mask or not args.metadata or not args.compression_output:
            parser.error("--eiger-master requires --mask, --metadata, and --compression-output")
        detector_mask = np.load(args.mask, allow_pickle=False)
        metadata = json.loads(args.metadata.read_text())
        metadata["pixel_mask"] = detector_mask.copy()
        frame_count = args.compression_frames or args.end
        raw_bytes = frame_count * detector_mask.size * np.dtype(np.uint32).itemsize
        for pass_name in ("cold", "warm"):
            destination = args.compression_output.with_name(
                f"{args.compression_output.stem}-{pass_name}{args.compression_output.suffix or '.cmp'}"
            )
            _, metric = _measure(
                f"compress_eigerdata_{pass_name}",
                lambda destination=destination: compress_eigerdata(
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
                    with_pickle=False,
                    direct_load_data=True,
                    data_path=os.fspath(args.eiger_master),
                    images_per_file=args.images_per_file,
                    copy_rawdata=False,
                    reverse=not args.no_reverse,
                    rot90=args.rot90,
                    func_images_per_file=lambda _path: args.images_per_file,
                ),
                raw_bytes,
            )
            measurements.append(metric)

    for pass_name in ("cold", "warm"):
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            data, metric = _measure(
                f"Get_Pixel_Arrayc_{pass_name}",
                lambda: Get_Pixel_Arrayc(compressed, pixel_list, norm=norm).get_data(),
                0,
            )
            metric["bytes_traversed"] = compressed._bytes_traversed
            measurements.append(metric)
        with Multifile(os.fspath(args.cmp), args.beg, args.end) as compressed:
            compressed._reset_io_counters()
            (_, _), metric = _measure(
                f"cal_g2p_{pass_name}",
                lambda: cal_g2p(
                    compressed,
                    roi_mask,
                    bad_frame_list=bad_frames,
                    num_buf=args.num_buf,
                    imgsum=imgsum,
                    norm=norm,
                ),
                0,
            )
            metric["bytes_traversed"] = compressed._bytes_traversed
            measurements.append(metric)
        two_time, metric = _measure(
            f"auto_two_Arrayc_{pass_name}", lambda: auto_two_Arrayc(data, roi_mask), data.nbytes
        )
        measurements.append(metric)
        _, metric = _measure(
            f"get_one_time_from_two_time_{pass_name}",
            lambda: get_one_time_from_two_time(two_time),
            two_time.nbytes,
        )
        measurements.append(metric)

    report = json.dumps(measurements, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
