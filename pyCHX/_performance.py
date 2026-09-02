"""Private CPU, memory, and compiled kernels used by performance-sensitive paths."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads


def affinity_cpu_ids():
    """Return the logical CPUs on which this process may run."""
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def physical_core_count(cpu_ids=None):
    """Count physical cores in the current CPU affinity mask.

    Linux exposes package/core identifiers in sysfs.  Falling back to the
    affinity-sized logical count is conservative on platforms without that
    topology information and, importantly, never escapes a scheduler cpuset.
    """
    if cpu_ids is None:
        cpu_ids = affinity_cpu_ids()
    cpu_ids = tuple(cpu_ids)
    cores = set()
    try:
        for cpu in cpu_ids:
            topology = f"/sys/devices/system/cpu/cpu{cpu}/topology"
            with open(os.path.join(topology, "physical_package_id")) as stream:
                package = int(stream.read())
            with open(os.path.join(topology, "core_id")) as stream:
                core = int(stream.read())
            cores.add((package, core))
    except (OSError, ValueError):
        return max(1, len(cpu_ids))
    return max(1, len(cores))


def available_memory_bytes():
    """Best-effort available-memory estimate honoring common cgroup limits."""
    available = None
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pass

    for current_name, maximum_name in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            with open(current_name) as stream:
                current = int(stream.read().strip())
            with open(maximum_name) as stream:
                maximum_text = stream.read().strip()
            if maximum_text != "max":
                cgroup_available = max(0, int(maximum_text) - current)
                available = cgroup_available if available is None else min(available, cgroup_available)
        except (OSError, ValueError):
            continue
    return max(1, available or 512 * 1024**2)


@contextmanager
def numba_thread_limit(limit):
    """Temporarily constrain Numba's parallel worker pool."""
    previous = get_num_threads()
    limit = max(1, min(int(limit), previous))
    set_num_threads(limit)
    try:
        yield
    finally:
        set_num_threads(previous)


@njit(cache=True, nogil=True)
def sparse_scatter_normalized(
    positions,
    values,
    lookup,
    output,
    output_row,
    source_frame,
    norm_1d,
    norm_2d,
    norm_columns,
    imgsum,
    mean_int_sets,
    qind,
    normalization_flags,
):
    """Scatter one sparse frame into a selected-pixel row with fused norms."""
    has_norm_1d = normalization_flags[0]
    has_norm_2d = normalization_flags[1]
    has_imgsum = normalization_flags[2]
    has_mean = normalization_flags[3]
    for source_index in range(positions.size):
        position = positions[source_index]
        if position < 0 or position >= lookup.size:
            continue
        destination = lookup[position]
        if destination < 0:
            continue
        divisor = 1.0
        if has_mean:
            divisor *= mean_int_sets[source_frame, qind[destination] - 1]
        if has_imgsum:
            divisor *= imgsum[source_frame]
        if has_norm_2d:
            divisor *= norm_2d[source_frame, norm_columns[destination]]
        elif has_norm_1d:
            divisor *= norm_1d[norm_columns[destination]]
        output[output_row, destination] = values[source_index] / divisor


@njit(cache=True, nogil=True)
def sparse_add_image(positions, values, flattened_output):
    for index in range(positions.size):
        position = positions[index]
        if 0 <= position < flattened_output.size:
            flattened_output[position] += values[index]


@njit(cache=True, nogil=True)
def sparse_roi_sums(positions, values, roi_lookup, output_row):
    for index in range(positions.size):
        position = positions[index]
        if 0 <= position < roi_lookup.size:
            roi_index = roi_lookup[position]
            if roi_index >= 0:
                output_row[roi_index] += values[index]


@njit(cache=True, nogil=True)
def sparse_frame_sum(values):
    total = 0.0
    for index in range(values.size):
        total += values[index]
    return total


@njit(cache=True, nogil=True)
def process_one_time_block(
    frames,
    buf,
    correlation,
    past_intensity,
    future_intensity,
    images_per_level,
    track_level,
    current,
    bad_counts,
    level_offsets,
    intensity_buf,
    correlation_all,
    past_all,
    future_all,
    calculate_error,
):
    """Update one ROI's multi-tau state for a contiguous frame block."""
    num_levels, num_bufs, pixel_count = buf.shape
    for frame_index in range(frames.shape[0]):
        current[0] = (1 + current[0]) % num_bufs
        buffer_number = current[0] - 1
        for pixel in range(pixel_count):
            buf[0, buffer_number, pixel] = frames[frame_index, pixel]

        level = 0
        processing = True
        while processing:
            images_per_level[level] += 1
            minimum_delay = num_bufs // 2 if level else 0
            future = buf[level, buffer_number]
            future_bad = False
            for pixel in range(pixel_count):
                if np.isnan(future[pixel]):
                    future_bad = True
                    break

            future_sum = 0.0
            if not future_bad and not calculate_error:
                for pixel in range(pixel_count):
                    future_sum += future[pixel]
                intensity_buf[level, buffer_number] = future_sum

            stop_delay = min(images_per_level[level], num_bufs)
            for delay in range(minimum_delay, stop_delay):
                time_index = level * num_bufs // 2 + delay
                delay_number = (buffer_number - delay) % num_bufs
                past = buf[level, delay_number]
                bad = future_bad
                if not bad:
                    for pixel in range(pixel_count):
                        if np.isnan(past[pixel]):
                            bad = True
                            break
                local_index = time_index - level_offsets[level]
                normalize = images_per_level[level] - delay - bad_counts[level, local_index]
                if bad:
                    bad_counts[level, local_index] += 1
                elif calculate_error:
                    for pixel in range(pixel_count):
                        product = past[pixel] * future[pixel]
                        correlation_all[time_index, pixel] += (
                            product - correlation_all[time_index, pixel]
                        ) / normalize
                        past_all[time_index, pixel] += (past[pixel] - past_all[time_index, pixel]) / normalize
                        future_all[time_index, pixel] += (
                            future[pixel] - future_all[time_index, pixel]
                        ) / normalize
                else:
                    product_sum = 0.0
                    past_sum = intensity_buf[level, delay_number]
                    for pixel in range(pixel_count):
                        product_sum += past[pixel] * future[pixel]
                    product_mean = product_sum / pixel_count
                    past_mean = past_sum / pixel_count
                    future_mean = future_sum / pixel_count
                    correlation[time_index] += (product_mean - correlation[time_index]) / normalize
                    past_intensity[time_index] += (past_mean - past_intensity[time_index]) / normalize
                    future_intensity[time_index] += (future_mean - future_intensity[time_index]) / normalize

            if level + 1 >= num_levels:
                processing = False
            else:
                level += 1
                if not track_level[level]:
                    track_level[level] = True
                    processing = False
                else:
                    previous = 1 + (current[level - 1] - 2) % num_bufs
                    current[level] = 1 + current[level] % num_bufs
                    buffer_number = current[level] - 1
                    for pixel in range(pixel_count):
                        buf[level, buffer_number, pixel] = (
                            buf[level - 1, previous - 1, pixel] + buf[level - 1, current[level - 1] - 1, pixel]
                        ) / 2.0
                    track_level[level] = False


@njit(cache=True, nogil=True, error_model="numpy")
def mirror_and_normalize_two_time(matrix, row_norm, pixel_count):
    """Normalize the computed upper triangle and mirror it in place."""
    frame_count = matrix.shape[0]
    for row in range(frame_count):
        for column in range(row, frame_count):
            value = matrix[row, column]
            value /= row_norm[column]
            value /= row_norm[row]
            value /= pixel_count
            matrix[row, column] = value
            matrix[column, row] = value


@njit(cache=True, nogil=True)
def mirror_two_time(matrix):
    """Copy an upper-triangular symmetric BLAS result to its lower half."""
    frame_count = matrix.shape[0]
    for row in range(frame_count):
        for column in range(row + 1, frame_count):
            matrix[column, row] = matrix[row, column]


@njit(cache=True, nogil=True, parallel=True)
def diagonal_nanmean(g12):
    """Reduce upper diagonals of a C-order (time, time, ROI) array."""
    frame_count = g12.shape[0]
    roi_count = g12.shape[2]
    output = np.empty((frame_count, roi_count), dtype=np.float64)
    for task in prange(frame_count * roi_count):
        delay = task // roi_count
        roi_index = task - delay * roi_count
        total = 0.0
        count = 0
        for frame in range(frame_count - delay):
            value = g12[frame, frame + delay, roi_index]
            if not np.isnan(value):
                total += value
                count += 1
        output[delay, roi_index] = total / count if count else np.nan
    return output


@njit(cache=True, nogil=True, parallel=True)
def normalize_diagonal_means(diagonal_means, norms, pixel_counts):
    frame_count, roi_count = diagonal_means.shape
    output = np.empty_like(diagonal_means)
    prefixes = np.empty((frame_count + 1, roi_count), dtype=np.float64)
    nan_prefixes = np.zeros((frame_count + 1, roi_count), dtype=np.int64)
    prefixes[0, :] = 0.0
    for frame in range(frame_count):
        for roi_index in range(roi_count):
            value = norms[frame, roi_index]
            if np.isnan(value):
                prefixes[frame + 1, roi_index] = prefixes[frame, roi_index]
                nan_prefixes[frame + 1, roi_index] = nan_prefixes[frame, roi_index] + 1
            else:
                prefixes[frame + 1, roi_index] = prefixes[frame, roi_index] + value
                nan_prefixes[frame + 1, roi_index] = nan_prefixes[frame, roi_index]
    for task in prange(frame_count * roi_count):
        delay = task // roi_count
        roi_index = task - delay * roi_count
        count = frame_count - delay
        future_nans = nan_prefixes[frame_count, roi_index] - nan_prefixes[delay, roi_index]
        past_nans = nan_prefixes[count, roi_index]
        if future_nans or past_nans:
            output[delay, roi_index] = np.nan
        else:
            future_mean = (prefixes[frame_count, roi_index] - prefixes[delay, roi_index]) / count
            past_mean = prefixes[count, roi_index] / count
            output[delay, roi_index] = diagonal_means[delay, roi_index] / (
                future_mean * past_mean * pixel_counts[roi_index]
            )
    return output
