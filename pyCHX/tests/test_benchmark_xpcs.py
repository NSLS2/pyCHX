import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from pyCHX.benchmarks import benchmark_xpcs
from pyCHX.chx_compress import init_compress_eigerdata

pytestmark = pytest.mark.portable


def test_parse_stages_defaults_exclude_unconfigured_compression():
    assert benchmark_xpcs._parse_stages(["all"]) == [
        "roi-intensity",
        "selected-pixels",
        "one-time",
        "two-time",
        "diagonal-means",
    ]
    assert benchmark_xpcs._parse_stages(["one-time,two-time", "one-time"]) == [
        "one-time",
        "two-time",
    ]
    assert benchmark_xpcs._parse_stages(["export", "export-raw"]) == ["export", "export-raw"]
    assert benchmark_xpcs._parse_stages(["reader-reuse"]) == ["reader-reuse"]
    assert benchmark_xpcs._parse_stages(["all"], include_artifacts=True)[-4:] == [
        "roi-intensity-plot",
        "g2-plot",
        "two-time-plot",
        "export",
    ]


def test_measure_reports_stage_peak_separately_from_lifetime_peak():
    result, metric = benchmark_xpcs._measure("small", lambda: np.arange(32), 256)

    np.testing.assert_array_equal(result, np.arange(32))
    assert metric["name"] == "small"
    assert metric["logical_bytes_visited"] == 256
    assert metric["peak_rss_bytes"] >= metric["rss_before_bytes"]
    assert metric["peak_rss_increase_bytes"] == metric["peak_rss_bytes"] - metric["rss_before_bytes"]
    assert metric["lifetime_self_peak_rss_bytes"] > 0
    assert set(metric["resource_usage"]) == {
        "cpu_user_seconds",
        "cpu_system_seconds",
        "minor_faults",
        "major_faults",
        "voluntary_context_switches",
        "involuntary_context_switches",
    }
    assert metric["os_io_accounting"]["complete"] is True


def test_measure_does_not_double_count_reaped_child_io(tmp_path):
    byte_count = 8 * 1024 * 1024
    output = tmp_path / "child-output.bin"
    code = (
        "import os,sys; "
        "f=open(sys.argv[1], 'wb'); "
        "f.write(b'x' * int(sys.argv[2])); "
        "f.flush(); os.fsync(f.fileno()); f.close()"
    )

    def write_in_child():
        subprocess.run([sys.executable, "-c", code, os.fspath(output), str(byte_count)], check=True)

    _, metric = benchmark_xpcs._measure("child-write", write_in_child)

    assert metric["os_io_accounting"]["complete"] is True
    assert metric["os_io_accounting"]["live_descendant_pids"] == []
    assert byte_count <= metric["os_io"]["write_characters"] < 2 * byte_count
    assert byte_count <= metric["os_io"]["write_bytes"] < 2 * byte_count


def test_buffered_read_record_preserves_latency_samples():
    record = benchmark_xpcs._buffered_read_record([(1024, 8, 0.2), (1032, 4, 0.5)])

    assert record == {
        "count": 2,
        "bytes": 12,
        "seconds": pytest.approx(0.7),
        "median_seconds": pytest.approx(0.35),
        "maximum_seconds": pytest.approx(0.5),
        "reads": [
            {"offset": 1024, "bytes": 8, "seconds": 0.2, "cumulative_bytes": 8},
            {"offset": 1032, "bytes": 4, "seconds": 0.5, "cumulative_bytes": 12},
        ],
    }


def test_component_timer_measures_generator_consumption():
    class Owner:
        @staticmethod
        def values():
            yield 1
            yield 2

    result, components = benchmark_xpcs._call_with_component_timers(
        lambda: list(Owner.values()),
        (("generator", Owner, "values"),),
    )

    assert result == [1, 2]
    assert components["generator"]["count"] == 1
    assert components["generator"]["seconds"] >= 0


def test_parse_stages_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        benchmark_xpcs._parse_stages(["plotting"])


def test_selected_pixel_stage_runs_in_fresh_process(tmp_path, capsys):
    frames = np.arange(1, 25, dtype=np.uint32).reshape(4, 2, 3)
    detector_mask = np.ones((2, 3), dtype=bool)
    roi_mask = np.asarray([[1, 1, 0], [2, 2, 2]], dtype=np.int64)
    cmp_path = tmp_path / "small.cmp"
    roi_path = tmp_path / "roi.npy"
    report_path = tmp_path / "report.json"
    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(cmp_path),
        with_pickle=False,
    )
    np.save(roi_path, roi_mask, allow_pickle=False)

    benchmark_xpcs.main(
        [
            "--cmp",
            str(cmp_path),
            "--roi",
            str(roi_path),
            "--end",
            "4",
            "--stages",
            "roi-intensity",
            "selected-pixels",
            "one-time",
            "two-time",
            "--output",
            str(report_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 1
    assert report["inputs"]["roi"]["sha256"]
    assert isinstance(report["environment"]["working_tree"]["dirty"], bool)
    assert isinstance(report["environment"]["working_tree"]["status"], list)
    assert isinstance(report["environment"]["working_tree"]["runtime_source_files"], list)
    assert report["worker_environments"][0]["function_bindings"][0]["source_file_sha256"]
    assert isinstance(report["worker_environments"][0]["thread_pools"], list)
    assert [measurement["stage"] for measurement in report["measurements"]] == [
        "roi-intensity",
        "selected-pixels",
        "one-time",
        "two-time",
    ]
    for measurement in report["measurements"]:
        assert measurement["execution_mode"] == "fresh-process"
        assert measurement["preflight"]["enforced"] is True
        assert measurement["preflight"]["estimated_required_memory_bytes"] > 0
    roi_measurement = report["measurements"][0]
    assert roi_measurement["buffered_reads"]["count"] >= 1
    assert roi_measurement["buffered_reads"]["bytes"] == cmp_path.stat().st_size - 1024
    assert roi_measurement["components"]["roi_setup_seconds"] >= 0
    one_time_measurement = report["measurements"][2]
    assert one_time_measurement["components"]["setup_seconds"] >= 0
    assert one_time_measurement["components"]["index_seconds"] >= 0
    assert one_time_measurement["components"]["populate_blocks_seconds"] >= 0
    assert one_time_measurement["components"]["correlate_blocks_seconds"] >= 0
    assert one_time_measurement["components"]["block_count"] >= 1
    two_time_measurement = report["measurements"][3]
    assert two_time_measurement["components"]["roi_gather_and_normalize"]["count"] == 2
    assert two_time_measurement["components"]["symmetric_blas"]["count"] == 2
    assert two_time_measurement["components"]["symmetric_scatter"]["count"] == 1


def test_reader_reuse_records_automatic_buffered_dispatch(tmp_path, capsys):
    frames = np.arange(1, 25, dtype=np.uint32).reshape(4, 2, 3)
    detector_mask = np.ones((2, 3), dtype=bool)
    roi_mask = np.asarray([[1, 1, 0], [2, 2, 2]], dtype=np.int64)
    cmp_path = tmp_path / "small.cmp"
    roi_path = tmp_path / "roi.npy"
    report_path = tmp_path / "report.json"
    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(cmp_path),
        with_pickle=False,
    )
    np.save(roi_path, roi_mask, allow_pickle=False)

    benchmark_xpcs.main(
        [
            "--cmp",
            str(cmp_path),
            "--roi",
            str(roi_path),
            "--end",
            "4",
            "--stages",
            "reader-reuse",
            "--output",
            str(report_path),
        ]
    )
    capsys.readouterr()

    metric = json.loads(report_path.read_text())["measurements"][0]
    assert metric["reader_state"] == {
        "beg_before": 0,
        "end_before": 4,
        "beg_after": 0,
        "end_after": 4,
        "requested_dispatch": "automatic",
        "effective_one_time_dispatch": "buffered",
        "completed_buffered_scan": [0, 4],
    }
    assert metric["components"]["roi_intensity_seconds"] >= 0
    assert metric["components"]["one_time_seconds"] >= 0


def test_two_time_thread_schedule_cycles_across_same_process_trials(tmp_path, capsys):
    frames = np.arange(1, 25, dtype=np.uint32).reshape(4, 2, 3)
    detector_mask = np.ones((2, 3), dtype=bool)
    roi_mask = np.asarray([[1, 1, 0], [2, 2, 2]], dtype=np.int64)
    cmp_path = tmp_path / "small.cmp"
    roi_path = tmp_path / "roi.npy"
    report_path = tmp_path / "report.json"
    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(cmp_path),
        with_pickle=False,
    )
    np.save(roi_path, roi_mask, allow_pickle=False)

    benchmark_xpcs.main(
        [
            "--cmp",
            str(cmp_path),
            "--roi",
            str(roi_path),
            "--end",
            "4",
            "--stages",
            "two-time",
            "--repetitions",
            "3",
            "--execution-mode",
            "same-process",
            "--two-time-thread-schedule",
            "2",
            "1",
            "--output",
            str(report_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(report_path.read_text())
    assert [measurement["two_time_thread_count"] for measurement in report["measurements"]] == [2, 1, 2]
    assert [measurement["trial"] for measurement in report["measurements"]] == [1, 2, 3]


def test_artifact_stages_write_isolated_plot_and_export_outputs(tmp_path, capsys):
    frames = np.arange(1, 25, dtype=np.uint32).reshape(4, 2, 3)
    detector_mask = np.ones((2, 3), dtype=bool)
    roi_mask = np.asarray([[1, 1, 0], [2, 2, 2]], dtype=np.int64)
    cmp_path = tmp_path / "small.cmp"
    roi_path = tmp_path / "roi.npy"
    report_path = tmp_path / "report.json"
    artifact_dir = tmp_path / "new" / "artifacts"
    reference_path = tmp_path / "reference.h5"
    init_compress_eigerdata(
        frames,
        detector_mask.copy(),
        {"pixel_mask": detector_mask.copy()},
        str(cmp_path),
        with_pickle=False,
    )
    np.save(roi_path, roi_mask, allow_pickle=False)
    with h5py.File(reference_path, "w") as h5file:
        h5file["g2"] = np.full((4, 4), 1.1)
        h5file["g2b"] = np.full((4, 4), 1.09)
        h5file["taus"] = np.arange(1, 5, dtype=np.float64)
        h5file["tausb"] = np.arange(1, 5, dtype=np.float64)
        q_values = h5file.create_dataset("qval_dict", (1,), dtype="i")
        for index, value in enumerate(((0.1, 0), (0.1, 90), (0.2, 0), (0.2, 90))):
            q_values.attrs[str(index)] = value
        for key in ("g2_fit_paras", "g2b_fit_paras"):
            group = h5file.create_group(key)
            group["block0_items"] = np.asarray([b"beta", b"relaxation_rate", b"alpha", b"baseline"])
            group["block0_values"] = np.tile([0.1, 0.01, 1.0, 1.0], (4, 1))

    benchmark_xpcs.main(
        [
            "--cmp",
            str(cmp_path),
            "--roi",
            str(roi_path),
            "--end",
            "4",
            "--stages",
            "roi-intensity-plot",
            "g2-plot",
            "two-time-plot",
            "g2-plot-reference",
            "export",
            "export-raw",
            "--artifact-dir",
            str(artifact_dir),
            "--reference-results",
            str(reference_path),
            "--output",
            str(report_path),
        ]
    )
    capsys.readouterr()

    report = json.loads(report_path.read_text())
    (
        plot_metric,
        g2_plot_metric,
        two_time_plot_metric,
        reference_plot_metric,
        export_metric,
        raw_export_metric,
    ) = report["measurements"]
    assert plot_metric["components"]["render_encode_write"]["count"] == 1
    assert plot_metric["components"]["csv_write"]["count"] == 1
    assert {record["path"].rsplit("/", 1)[-1] for record in plot_metric["artifacts"]} == {
        "benchmark_t_ROIs",
        "benchmark_t_ROIs.png",
    }
    assert g2_plot_metric["components"]["layout"]["count"] == 1
    assert g2_plot_metric["components"]["render_encode_write"]["count"] == 1
    assert two_time_plot_metric["components"]["render_encode_write"]["count"] == 1
    assert set(reference_plot_metric["plot_calls"]) == {"direct_fit", "derived_fit", "comparison"}
    assert reference_plot_metric["components"]["render_encode_write"]["count"] == 6
    assert reference_plot_metric["components"]["draw"]["count"] >= 6
    assert reference_plot_metric["components"]["montage"]["count"] == 3
    assert export_metric["components"]["array_write"]["count"] == 1
    assert export_metric["components"]["metadata_write"]["count"] == 0
    assert export_metric["components"]["dataframe_write"]["count"] == 0
    assert export_metric["components"]["open"]["count"] == 1
    assert export_metric["components"]["close"]["count"] == 1
    assert export_metric["dataset"]["chunks"] is None
    assert export_metric["dataset"]["compression"] is None
    assert export_metric["dataset"]["fill_value"] == 0
    assert export_metric["artifacts"][0]["path"].endswith("benchmark_results.h5")
    assert "sha256" not in export_metric["artifacts"][0]
    assert all(component["count"] == 1 for component in raw_export_metric["components"].values())
    assert raw_export_metric["dataset"] == export_metric["dataset"]
    raw_path = raw_export_metric["artifacts"][0]["path"]
    assert raw_path.endswith("benchmark_raw_contiguous.h5")
    export_path = export_metric["artifacts"][0]["path"]
    with h5py.File(export_path, "r") as exported, h5py.File(raw_path, "r") as raw:
        np.testing.assert_array_equal(exported["g12b"][:], raw["g12b"][:])
