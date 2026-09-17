import operator
import os
import pickle as pkl
import shutil
import struct
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count

import dill
import matplotlib.pyplot as plt
import numpy as np
import skbeam.core.roi as roi
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from tqdm import tqdm

from pyCHX._performance import (
    physical_core_count,
    sparse_add_image,
    sparse_frame_sum,
    sparse_roi_sums,
)
from pyCHX.chx_generic_functions import (
    copy_data,
    create_time_slice,
    delete_data,
    get_detector,
    get_eigerImage_per_file,
    get_sid_filenames,
    load_data,
)
from pyCHX.chx_handlers import EigerImages, db
from pyCHX.chx_libs import RUN_GUI
from pyCHX.config import get_compressed_data_dir


def run_dill_encoded(what):
    fun, args = dill.loads(what)
    return fun(*args)


def apply_async(pool, fun, args, callback=None):
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback=callback)


def _make_pool(task_count):
    """Create no more worker processes than either tasks or available CPUs."""
    if task_count < 1:
        raise ValueError("at least one multiprocessing task is required")
    return Pool(processes=min(task_count, _available_cpu_count()))


def _available_cpu_count():
    """Return affinity-constrained physical cores, avoiding SMT oversubscription."""
    detected = cpu_count()
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return detected
    return min(detected, physical_core_count(affinity))


def _write_sparse_frame(stream, positions, values):
    """Write one sparse CMP frame using the legacy native binary layout."""
    stream.write(np.asarray(len(positions), dtype=np.uint32).tobytes())
    if len(positions):
        stream.write(np.asarray(positions, dtype=np.int32).tobytes())
        stream.write(np.ascontiguousarray(values).tobytes())


def _publish_file(source, destination):
    """Copy *source* beside *destination* and atomically publish it."""
    destination = os.path.abspath(destination)
    destination_dir = os.path.dirname(destination)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination_dir,
        prefix=".%s." % os.path.basename(destination),
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        shutil.copymode(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _staged_init_compress_eigerdata(images, mask, md, filename, new_path, **kwargs):
    """Run serial compression locally, then publish its completed output."""
    staging_dir = tempfile.mkdtemp(prefix="pychx-compress-", dir=new_path)
    staged_filename = os.path.join(staging_dir, os.path.basename(filename))
    try:
        result = init_compress_eigerdata(images, mask, md, staged_filename, **kwargs)
        _publish_file(staged_filename, filename)
        if kwargs.get("with_pickle", True):
            _publish_file(staged_filename + ".pkl", filename + ".pkl")
        return result
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _collect_pool_results(pool, results, show_progress=False):
    """Collect keyed async results and always reap the worker processes."""
    try:
        pool.close()
        keys = list(sorted(results))
        if show_progress:
            keys = tqdm(keys)
        return [results[key].get() for key in keys]
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _frame_bin_edges(frame_count, bins):
    """Return consecutive, non-overlapping frame ranges for compression."""
    try:
        bins = operator.index(bins)
    except TypeError as error:
        raise TypeError("bins must be an integer") from error
    if bins < 1:
        raise ValueError("bins must be at least one")

    starts = np.arange(0, frame_count, bins, dtype=np.int64)
    stops = np.minimum(starts + bins, frame_count)
    return np.column_stack((starts, stops))


def _read_eiger_contiguous(images, start, stop):
    """Read a contiguous Eiger range with one HDF5 slice per data file."""
    pieces = []
    while start < stop:
        file_number = start // images.images_per_file
        dataset = images._entry[f"data_{file_number + 1:06d}"]
        local_start = start - file_number * images.images_per_file
        local_stop = min(dataset.shape[0], stop - file_number * images.images_per_file)
        pieces.append(dataset[local_start:local_stop])
        start += local_stop - local_start
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces, axis=0)


def _iter_binned_images(images, start, stop, bins, reverse=False, rot90=False):
    """Yield binned images, using bounded HDF5 block reads when available."""
    direct_eiger = all(hasattr(images, attribute) for attribute in ("_entry", "images_per_file"))
    if not direct_eiger:
        sliced = images[start:stop]
        if bins == 1:
            yield from sliced
            return
        for local_start, local_stop in _frame_bin_edges(stop - start, bins):
            yield np.average(sliced[local_start:local_stop], axis=0)
        return

    first_key = images.valid_keys[0]
    frame_bytes = int(np.prod(images._entry[first_key].shape[1:])) * images._entry[first_key].dtype.itemsize
    target_frames = max(1, (256 * 1024**2) // max(1, frame_bytes))
    block_frames = max(bins, (target_frames // bins) * bins)
    for block_start in range(start, stop, block_frames):
        block_stop = min(stop, block_start + block_frames)
        block = _read_eiger_contiguous(images, block_start, block_stop)
        if reverse:
            block = block[:, ::-1, :]
        if rot90:
            block = np.rot90(block, axes=(1, 2))
        if bins == 1:
            yield from block
            continue
        for local_start, local_stop in _frame_bin_edges(len(block), bins):
            yield np.average(block[local_start:local_stop], axis=0)


def _publish_compressed_segments(staged_filename, destination, segment_count):
    """Concatenate segment files directly into an atomic destination sibling."""
    sources = [staged_filename + "-header"] + [
        staged_filename + "_temp-%i.tmp" % index for index in range(segment_count)
    ]
    if not all(os.path.exists(source) for source in sources):
        # Retains compatibility with callers/tests that replace the public
        # combination and publication helpers.
        combine_compressed(staged_filename, segment_count, del_old=True)
        _publish_file(staged_filename, destination)
        return
    destination = os.path.abspath(destination)
    descriptor, temporary = tempfile.mkstemp(
        dir=os.path.dirname(destination),
        prefix=".%s." % os.path.basename(destination),
        suffix=".tmp",
    )
    source_mode = os.stat(sources[0]).st_mode
    try:
        with os.fdopen(descriptor, "wb") as output:
            for source_name in sources:
                with open(source_name, "rb") as source:
                    shutil.copyfileobj(source, output, length=16 * 1024**2)
                os.remove(source_name)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, source_mode)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def map_async(pool, fun, args):
    return pool.map_async(run_dill_encoded, (dill.dumps((fun, args)),))


def pass_FD(FD, n):
    # FD.rdframe(n)
    try:
        FD.seekimg(n)
    except Exception:
        pass
        return False


def go_through_FD(FD):
    if not pass_FD(FD, FD.beg):
        for i in range(FD.beg, FD.end):
            pass_FD(FD, i)
    else:
        pass


def compress_eigerdata(
    images,
    mask,
    md,
    filename=None,
    force_compress=False,
    bad_pixel_threshold=1e15,
    bad_pixel_low_threshold=0,
    hot_pixel_threshold=2**30,
    nobytes=2,
    bins=1,
    bad_frame_list=None,
    para_compress=False,
    num_sub=128,
    dtypes="uid",
    reverse=True,
    rot90=False,
    num_max_para_process=500,
    with_pickle=False,
    direct_load_data=True,
    data_path=None,
    images_per_file=100,
    copy_rawdata=True,
    new_path="/tmp/",
    func_images_per_file=get_eigerImage_per_file,
):
    """
    Init 2016, YG@CHX
    DEV 2018, June, make images_per_file a dummy, will be determined by get_eigerImage_per_file if direct_load_data
                    Add copy_rawdata opt.

    """

    end = len(_frame_bin_edges(len(images), bins))
    if filename is None:
        filename = os.path.join(get_compressed_data_dir(), "uid_%s.cmp" % md["uid"])
    if dtypes != "uid":
        para_compress = False
    else:
        if para_compress:
            images = "foo"
            # para_compress=   True
    # print( dtypes )
    if direct_load_data:
        images_per_file = func_images_per_file(data_path)
        if data_path is None:
            sud = get_sid_filenames(db[md["uid"]])
            data_path = sud[2][0]
    if force_compress:
        print("Create a new compress file with filename as :%s." % filename)
        if para_compress:
            # stop connection to be before forking... (let it reset again); 11/09/2024 this seems to fail with
            # 'registry doesn't have attribute disconnect... -> try making this optional; this might have been a
            # leftover: if compression happens "natuarally" (not as force_compress=True) this disconnect/reconnect
            # is already missing...we definitely had this error before...
            try:
                db.reg.disconnect()
                db.mds.reset_connection()
            except Exception:
                pass
            print("Using a multiprocess to compress the data.")
            return para_compress_eigerdata(
                images,
                mask,
                md,
                filename,
                bad_pixel_threshold=bad_pixel_threshold,
                hot_pixel_threshold=hot_pixel_threshold,
                bad_pixel_low_threshold=bad_pixel_low_threshold,
                nobytes=nobytes,
                bins=bins,
                num_sub=num_sub,
                dtypes=dtypes,
                rot90=rot90,
                reverse=reverse,
                num_max_para_process=num_max_para_process,
                with_pickle=with_pickle,
                direct_load_data=direct_load_data,
                data_path=data_path,
                images_per_file=images_per_file,
                copy_rawdata=copy_rawdata,
                new_path=new_path,
            )
        else:
            return _staged_init_compress_eigerdata(
                images,
                mask,
                md,
                filename,
                new_path,
                bad_pixel_threshold=bad_pixel_threshold,
                hot_pixel_threshold=hot_pixel_threshold,
                bad_pixel_low_threshold=bad_pixel_low_threshold,
                nobytes=nobytes,
                bins=bins,
                with_pickle=with_pickle,
                direct_load_data=direct_load_data,
                data_path=data_path,
                images_per_file=images_per_file,
                reverse=reverse,
                rot90=rot90,
            )
    else:
        if not os.path.exists(filename):
            print("Create a new compress file with filename as :%s." % filename)
            if para_compress:
                print("Using a multiprocess to compress the data.")
                return para_compress_eigerdata(
                    images,
                    mask,
                    md,
                    filename,
                    bad_pixel_threshold=bad_pixel_threshold,
                    hot_pixel_threshold=hot_pixel_threshold,
                    bad_pixel_low_threshold=bad_pixel_low_threshold,
                    nobytes=nobytes,
                    bins=bins,
                    num_sub=num_sub,
                    dtypes=dtypes,
                    reverse=reverse,
                    rot90=rot90,
                    num_max_para_process=num_max_para_process,
                    with_pickle=with_pickle,
                    direct_load_data=direct_load_data,
                    data_path=data_path,
                    images_per_file=images_per_file,
                    copy_rawdata=copy_rawdata,
                    new_path=new_path,
                )
            else:
                return _staged_init_compress_eigerdata(
                    images,
                    mask,
                    md,
                    filename,
                    new_path,
                    bad_pixel_threshold=bad_pixel_threshold,
                    hot_pixel_threshold=hot_pixel_threshold,
                    bad_pixel_low_threshold=bad_pixel_low_threshold,
                    nobytes=nobytes,
                    bins=bins,
                    with_pickle=with_pickle,
                    direct_load_data=direct_load_data,
                    data_path=data_path,
                    images_per_file=images_per_file,
                    reverse=reverse,
                    rot90=rot90,
                )
        else:
            print("Using already created compressed file with filename as :%s." % filename)
            beg = 0
            return read_compressed_eigerdata(
                mask,
                filename,
                beg,
                end,
                bad_pixel_threshold=bad_pixel_threshold,
                hot_pixel_threshold=hot_pixel_threshold,
                bad_pixel_low_threshold=bad_pixel_low_threshold,
                bad_frame_list=bad_frame_list,
                with_pickle=with_pickle,
                direct_load_data=direct_load_data,
                data_path=data_path,
                images_per_file=images_per_file,
            )


def read_compressed_eigerdata(
    mask,
    filename,
    beg,
    end,
    bad_pixel_threshold=1e15,
    hot_pixel_threshold=2**30,
    bad_pixel_low_threshold=0,
    bad_frame_list=None,
    with_pickle=False,
    direct_load_data=False,
    data_path=None,
    images_per_file=100,
):
    """
    Read already compress eiger data
    Return
        mask
        avg_img
        imsum
        bad_frame_list

    """
    # should use try and except instead of with_pickle in the future!
    CAL = False
    if not with_pickle:
        CAL = True
    else:
        try:
            with open(filename + ".pkl", "rb") as stream:
                mask, avg_img, imgsum, bad_frame_list_ = pkl.load(stream)
        except Exception:
            CAL = True
    if CAL:
        with Multifile(filename, beg, end) as FD:
            imgsum = np.zeros(FD.end - FD.beg, dtype=np.float64)
            avg_img = np.zeros([FD.md["ncols"], FD.md["nrows"]], dtype=np.float64)
            supplied_bad = set() if bad_frame_list is None else set(np.atleast_1d(bad_frame_list).tolist())
            detected_bad = []
            good_count = 0
            flattened_average = avg_img.ravel()
            for output_index, frame_index in enumerate(range(FD.beg, FD.end)):
                positions, values = FD._raw_frame_view(frame_index)
                frame_sum = sparse_frame_sum(values)
                imgsum[output_index] = frame_sum
                is_bad = frame_sum > bad_pixel_threshold or frame_sum <= bad_pixel_low_threshold
                if is_bad:
                    detected_bad.append(frame_index)
                if not is_bad and frame_index not in supplied_bad:
                    sparse_add_image(positions, values, flattened_average)
                    good_count += 1
            bad_frame_list_ = np.unique(np.asarray([*supplied_bad, *detected_bad], dtype=np.int64))
            if good_count:
                avg_img /= good_count
            else:
                avg_img.fill(np.nan)

    return mask, avg_img, imgsum, bad_frame_list_


def para_compress_eigerdata(
    images,
    mask,
    md,
    filename,
    num_sub=128,
    bad_pixel_threshold=1e15,
    hot_pixel_threshold=2**30,
    bad_pixel_low_threshold=0,
    nobytes=4,
    bins=1,
    dtypes="uid",
    reverse=True,
    rot90=False,
    num_max_para_process=500,
    cpu_core_number=0,
    with_pickle=True,
    direct_load_data=False,
    data_path=None,
    images_per_file=100,
    copy_rawdata=True,
    new_path="/tmp/",
):

    data_path_ = data_path
    raw_data_copied = False
    if dtypes == "uid":
        uid = md["uid"]  # images
        if not direct_load_data:
            detector = get_detector(db[uid])
            images_ = load_data(uid, detector, reverse=reverse, rot90=rot90)
        else:
            # print('Here for images_per_file: %s'%images_per_file)
            # images_ = EigerImages( data_path, images_per_file=images_per_file)
            # print('here')
            if not copy_rawdata:
                images_ = EigerImages(data_path, images_per_file, md)
            else:
                print("Due to a IO problem running on GPFS. The raw data will be copied to /tmp/")
                print("Copying...")
                copy_data(data_path, new_path)
                # print(data_path, new_path)
                new_master_file = os.path.join(new_path, os.path.basename(data_path))
                data_path_ = new_master_file
                images_ = EigerImages(new_master_file, images_per_file, md)
                raw_data_copied = True
                # print(md)
            try:
                N = len(images_)
            finally:
                images_.close()

        if not direct_load_data:
            N = len(images_)

    else:
        N = len(images)

    if cpu_core_number == 0:
        cpu_core_number = _available_cpu_count()
    else:
        cpu_core_number = min(cpu_core_number, _available_cpu_count())

    raw_image_count = N
    N = len(_frame_bin_edges(raw_image_count, bins))
    Nf = int(np.ceil(N / num_sub))
    if Nf > cpu_core_number:
        print("The process number is larger than %s (current server's core threads)" % cpu_core_number)
        num_sub_old = num_sub
        num_sub = int(np.ceil(N / cpu_core_number))
        Nf = int(np.ceil(N / num_sub))
        print("The sub compressed file number was changed from %s to %s" % (num_sub_old, num_sub))
    staging_dir = tempfile.mkdtemp(prefix="pychx-compress-", dir=new_path)
    staged_filename = os.path.join(staging_dir, os.path.basename(filename))
    try:
        create_compress_header(md, staged_filename + "-header", nobytes, bins, rot90=rot90)
        segment_results = _iter_parallel_segment_results(
            images=images,
            mask=mask,
            md=md,
            filename=staged_filename,
            num_sub=num_sub,
            bad_pixel_threshold=bad_pixel_threshold,
            hot_pixel_threshold=hot_pixel_threshold,
            bad_pixel_low_threshold=bad_pixel_low_threshold,
            nobytes=nobytes,
            bins=bins,
            dtypes=dtypes,
            num_max_para_process=num_max_para_process,
            reverse=reverse,
            rot90=rot90,
            direct_load_data=direct_load_data,
            data_path=data_path_,
            images_per_file=images_per_file,
            image_count=raw_image_count,
            segment_count=Nf,
        )

        imgsum = np.zeros(N)
        bad_frame_list = np.zeros(N, dtype=bool)
        good_count = 0
        for i, segment_result in segment_results:
            mask_, avg_img_, imgsum_, bad_frame_list_ = segment_result
            imgsum[i * num_sub : (i + 1) * num_sub] = imgsum_
            bad_frame_list[i * num_sub : (i + 1) * num_sub] = bad_frame_list_
            segment_good_count = len(imgsum_) - np.count_nonzero(bad_frame_list_)
            if i == 0:
                mask = mask_
                avg_img = np.zeros_like(avg_img_, dtype=np.float64)
            else:
                mask *= mask_
            if segment_good_count and not np.any(np.isnan(avg_img_)):
                avg_img += avg_img_ * segment_good_count
                good_count += segment_good_count

        bad_frame_list = np.where(bad_frame_list)[0]
        if good_count:
            avg_img /= good_count
        else:
            avg_img.fill(np.nan)

        if len(bad_frame_list):
            print("Bad frame list are: %s" % bad_frame_list)
        else:
            print("No bad frames are involved.")
        print("Combining the seperated compressed files together...")
        _publish_compressed_segments(staged_filename, filename, Nf)
        if with_pickle:
            staged_pickle = staged_filename + ".pkl"
            with open(staged_pickle, "wb") as stream:
                pkl.dump([mask, avg_img, imgsum, bad_frame_list], stream)
            _publish_file(staged_pickle, filename + ".pkl")
        return mask, avg_img, imgsum, bad_frame_list
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if raw_data_copied:
            delete_data(data_path, new_path)


def combine_compressed(filename, Nf, del_old=True):
    old_files = [filename + "-header"]
    for i in range(Nf):
        old_files.append(filename + "_temp-%i.tmp" % i)
    combine_binary_files(filename, old_files, del_old)


def combine_binary_files(filename, old_files, del_old=False):
    """Combine binary files together"""
    with open(filename, "wb") as destination:
        for source_name in old_files:
            with open(source_name, "rb") as source:
                shutil.copyfileobj(source, destination)
            if del_old:
                os.remove(source_name)


def para_segment_compress_eigerdata(
    images,
    mask,
    md,
    filename,
    num_sub=100,
    bad_pixel_threshold=1e15,
    hot_pixel_threshold=2**30,
    bad_pixel_low_threshold=0,
    nobytes=4,
    bins=1,
    dtypes="images",
    reverse=True,
    rot90=False,
    num_max_para_process=50,
    direct_load_data=False,
    data_path=None,
    images_per_file=100,
):
    """
    parallelly compressed eiger data without header, this function is for parallel compress
    """
    if dtypes == "uid":
        uid = md["uid"]
        if not direct_load_data:
            detector = get_detector(db[uid])
            image_count = len(load_data(uid, detector, reverse=reverse, rot90=rot90))
        else:
            probe = EigerImages(data_path, images_per_file, md)
            try:
                image_count = len(probe)
            finally:
                probe.close()
    else:
        image_count = len(images)

    # N = int( np.ceil( N/ bins  ) )
    num_sub *= bins
    Nf = int(np.ceil(image_count / num_sub))
    print("It will create %i temporary files for parallel compression." % Nf)

    if Nf > num_max_para_process:
        print("The segment count %s exceeds the concurrent worker limit %s" % (Nf, num_max_para_process))
    worker_count = min(Nf, num_max_para_process, _available_cpu_count())
    print("Pool processes: %s" % worker_count)
    pool = Pool(
        processes=worker_count,
        initializer=_compression_worker_init,
        initargs=(
            images,
            mask,
            md,
            bad_pixel_threshold,
            hot_pixel_threshold,
            bad_pixel_low_threshold,
            nobytes,
            bins,
            dtypes,
            reverse,
            rot90,
            direct_load_data,
            data_path,
            images_per_file,
        ),
    )
    result = {}
    try:
        for i in range(Nf):
            start = i * num_sub
            stop = min((i + 1) * num_sub, image_count)
            result[i] = pool.apply_async(
                _compression_worker_run,
                ((filename + "_temp-%i.tmp" % i, start, stop),),
            )
        pool.close()
        for async_result in result.values():
            async_result.wait()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    return result


_COMPRESSION_WORKER_CONTEXT = None


def _compression_worker_init(
    images,
    mask,
    md,
    bad_pixel_threshold,
    hot_pixel_threshold,
    bad_pixel_low_threshold,
    nobytes,
    bins,
    dtypes,
    reverse,
    rot90,
    direct_load_data,
    data_path,
    images_per_file,
):
    """Initialize a persistent compression worker once."""
    global _COMPRESSION_WORKER_CONTEXT
    apply_reverse = False
    apply_rot90 = False
    if dtypes == "uid":
        if direct_load_data:
            images = EigerImages(data_path, images_per_file, md)
            apply_reverse = reverse
            apply_rot90 = rot90
        else:
            detector = get_detector(db[md["uid"]])
            images = load_data(md["uid"], detector, reverse=reverse, rot90=rot90)
    _COMPRESSION_WORKER_CONTEXT = (
        images,
        np.asarray(mask),
        bad_pixel_threshold,
        hot_pixel_threshold,
        bad_pixel_low_threshold,
        nobytes,
        bins,
        apply_reverse,
        apply_rot90,
    )


def _compression_worker_run(descriptor):
    filename, start, stop = descriptor
    (
        images,
        mask,
        bad_pixel_threshold,
        hot_pixel_threshold,
        bad_pixel_low_threshold,
        nobytes,
        bins,
        reverse,
        rot90,
    ) = _COMPRESSION_WORKER_CONTEXT
    return _compress_segment(
        images,
        mask.copy(),
        filename,
        bad_pixel_threshold,
        hot_pixel_threshold,
        bad_pixel_low_threshold,
        nobytes,
        bins,
        start,
        stop,
        reverse,
        rot90,
    )


def _compression_worker_run_indexed(descriptor):
    segment_index, filename, start, stop = descriptor
    return segment_index, _compression_worker_run((filename, start, stop))


def _iter_parallel_segment_results(
    *,
    images,
    mask,
    md,
    filename,
    num_sub,
    bad_pixel_threshold,
    hot_pixel_threshold,
    bad_pixel_low_threshold,
    nobytes,
    bins,
    dtypes,
    reverse,
    rot90,
    num_max_para_process,
    direct_load_data,
    data_path,
    images_per_file,
    image_count,
    segment_count,
):
    """Yield completed segment reductions in order while workers stay alive."""
    raw_segment_size = num_sub * bins
    descriptors = [
        (
            segment_index,
            filename + "_temp-%i.tmp" % segment_index,
            segment_index * raw_segment_size,
            min((segment_index + 1) * raw_segment_size, image_count),
        )
        for segment_index in range(segment_count)
    ]
    worker_count = min(segment_count, num_max_para_process, _available_cpu_count())
    print("Pool processes: %s" % worker_count)
    pool = Pool(
        processes=worker_count,
        initializer=_compression_worker_init,
        initargs=(
            images,
            mask,
            md,
            bad_pixel_threshold,
            hot_pixel_threshold,
            bad_pixel_low_threshold,
            nobytes,
            bins,
            dtypes,
            reverse,
            rot90,
            direct_load_data,
            data_path,
            images_per_file,
        ),
    )
    try:
        iterator = pool.imap(_compression_worker_run_indexed, descriptors, chunksize=1)
        pool.close()
        yield from iterator
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _compress_segment(
    images,
    mask,
    filename,
    bad_pixel_threshold,
    hot_pixel_threshold,
    bad_pixel_low_threshold,
    nobytes,
    bins,
    start,
    stop,
    reverse=False,
    rot90=False,
):
    """Compress one raw-frame range without constructing another reader."""
    if nobytes == 2:
        dtype = np.int16
    elif nobytes == 4:
        dtype = np.int32
    elif nobytes == 8:
        dtype = np.float64
    else:
        print("Wrong type of nobytes, only support 2 [np.int16] or 4 [np.int32]")
        dtype = np.int32
    if bins != 1:
        dtype = np.float64

    output_count = len(_frame_bin_edges(stop - start, bins))
    imgsum = np.zeros(output_count)
    avg_img = np.zeros(mask.shape, dtype=np.float64)
    good_count = 0
    with open(filename, "wb") as stream:
        for output_index, source_image in enumerate(
            _iter_binned_images(images, start, stop, bins, reverse, rot90)
        ):
            image = np.asarray(source_image, dtype=dtype)
            mask &= image < hot_pixel_threshold
            flattened = image.ravel()
            positions = np.flatnonzero((flattened > 0) & mask.ravel())
            values = flattened[positions]
            imgsum[output_index] = values.sum()
            if (
                len(positions) == 0
                or imgsum[output_index] > bad_pixel_threshold
                or imgsum[output_index] <= bad_pixel_low_threshold
            ):
                _write_sparse_frame(stream, (), ())
            else:
                avg_img.ravel()[positions] += values
                good_count += 1
                _write_sparse_frame(stream, positions, values)
    if good_count:
        avg_img /= good_count
    else:
        avg_img.fill(np.nan)
    bad_frames = (imgsum > bad_pixel_threshold) | (imgsum <= bad_pixel_low_threshold)
    sys.stdout.write("#")
    sys.stdout.flush()
    return mask, avg_img, imgsum, bad_frames


def segment_compress_eigerdata(
    images,
    mask,
    md,
    filename,
    bad_pixel_threshold=1e15,
    hot_pixel_threshold=2**30,
    bad_pixel_low_threshold=0,
    nobytes=4,
    bins=1,
    N1=None,
    N2=None,
    dtypes="images",
    reverse=True,
    rot90=False,
    direct_load_data=False,
    data_path=None,
    images_per_file=100,
):
    """
    Create a compressed eiger data without header, this function is for parallel compress
    for parallel compress don't pass any non-scalar parameters
    """
    owned_images = None
    apply_reverse = False
    apply_rot90 = False
    if dtypes == "uid":
        uid = md["uid"]
        if not direct_load_data:
            detector = get_detector(db[uid])
            source = load_data(uid, detector, reverse=reverse, rot90=rot90)
        else:
            owned_images = EigerImages(data_path, images_per_file, md)
            source = owned_images
            apply_reverse = reverse
            apply_rot90 = rot90
    else:
        source = images
    start = 0 if N1 is None else N1
    stop = len(source) if N2 is None else min(N2, len(source))
    try:
        return _compress_segment(
            source,
            mask,
            filename,
            bad_pixel_threshold,
            hot_pixel_threshold,
            bad_pixel_low_threshold,
            nobytes,
            bins,
            start,
            stop,
            apply_reverse,
            apply_rot90,
        )
    finally:
        if owned_images is not None:
            owned_images.close()


def create_compress_header(md, filename, nobytes=4, bins=1, rot90=False):
    """
    Create the head for a compressed eiger data, this function is for parallel compress
    """
    fp = open(filename, "wb")
    # Make Header 1024 bytes
    # md = images.md
    if bins != 1:
        nobytes = 8
    flag = True
    # print(   list(md.keys())   )
    # print(md)
    if "pixel_mask" in list(md.keys()):
        sx, sy = md["pixel_mask"].shape[0], md["pixel_mask"].shape[1]
    elif "img_shape" in list(md.keys()):
        sx, sy = md["img_shape"][0], md["img_shape"][1]
    else:
        sx, sy = 2167, 2070  # by default for 4M
    # print(flag)
    klst = [
        "beam_center_x",
        "beam_center_y",
        "count_time",
        "detector_distance",
        "frame_time",
        "incident_wavelength",
        "x_pixel_size",
        "y_pixel_size",
    ]
    vs = [0, 0, 0, 0, 0, 0, 75, 75]
    for i, k in enumerate(klst):
        if k in list(md.keys()):
            vs[i] = md[k]
    if flag:
        if rot90:
            Header = struct.pack(
                "@16s8d7I916x",
                b"Version-COMP0001",
                vs[0],
                vs[1],
                vs[2],
                vs[3],
                vs[4],
                vs[5],
                vs[6],
                vs[7],
                nobytes,
                sx,
                sy,
                0,
                sx,
                0,
                sy,
            )

        else:
            Header = struct.pack(
                "@16s8d7I916x",
                b"Version-COMP0001",
                vs[0],
                vs[1],
                vs[2],
                vs[3],
                vs[4],
                vs[5],
                vs[6],
                vs[7],
                # md['beam_center_x'],md['beam_center_y'], md['count_time'], md['detector_distance'],
                # #md['frame_time'],md['incident_wavelength'], md['x_pixel_size'],md['y_pixel_size'],
                nobytes,
                sy,
                sx,
                0,
                sy,
                0,
                sx,
            )

    fp.write(Header)
    fp.close()


def init_compress_eigerdata(
    images,
    mask,
    md,
    filename,
    bad_pixel_threshold=1e15,
    hot_pixel_threshold=2**30,
    bad_pixel_low_threshold=0,
    nobytes=4,
    bins=1,
    with_pickle=True,
    reverse=True,
    rot90=False,
    direct_load_data=False,
    data_path=None,
    images_per_file=100,
):
    """
    Compress the eiger data

    Create a new mask by remove hot_pixel
    Do image average
    Do each image sum
    Find badframe_list for where image sum above bad_pixel_threshold
    Generate a compressed data with filename

    if bins!=1, will bin the images with bin number as bins

    Header contains 1024 bytes ['Magic value', 'beam_center_x', 'beam_center_y', 'count_time', 'detector_distance',
       'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size',
       bytes per pixel (either 2 or 4 (Default)),
       Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End ]

    Return
        mask
        avg_img
        imsum
        bad_frame_list

    """
    fp = open(filename, "wb")
    # Make Header 1024 bytes
    # md = images.md
    if bins != 1:
        nobytes = 8
    if "count_time" not in list(md.keys()):
        md["count_time"] = 0
    if "detector_distance" not in list(md.keys()):
        md["detector_distance"] = 0
    if "frame_time" not in list(md.keys()):
        md["frame_time"] = 0
    if "incident_wavelength" not in list(md.keys()):
        md["incident_wavelength"] = 0
    if "y_pixel_size" not in list(md.keys()):
        md["y_pixel_size"] = 0
    if "x_pixel_size" not in list(md.keys()):
        md["x_pixel_size"] = 0
    if "beam_center_x" not in list(md.keys()):
        md["beam_center_x"] = 0
    if "beam_center_y" not in list(md.keys()):
        md["beam_center_y"] = 0

    if not rot90:
        Header = struct.pack(
            "@16s8d7I916x",
            b"Version-COMP0001",
            md["beam_center_x"],
            md["beam_center_y"],
            md["count_time"],
            md["detector_distance"],
            md["frame_time"],
            md["incident_wavelength"],
            md["x_pixel_size"],
            md["y_pixel_size"],
            nobytes,
            md["pixel_mask"].shape[1],
            md["pixel_mask"].shape[0],
            0,
            md["pixel_mask"].shape[1],
            0,
            md["pixel_mask"].shape[0],
        )
    else:
        Header = struct.pack(
            "@16s8d7I916x",
            b"Version-COMP0001",
            md["beam_center_x"],
            md["beam_center_y"],
            md["count_time"],
            md["detector_distance"],
            md["frame_time"],
            md["incident_wavelength"],
            md["x_pixel_size"],
            md["y_pixel_size"],
            nobytes,
            md["pixel_mask"].shape[0],
            md["pixel_mask"].shape[1],
            0,
            md["pixel_mask"].shape[0],
            0,
            md["pixel_mask"].shape[1],
        )

    fp.write(Header)

    owned_images = None
    apply_reverse = False
    apply_rot90 = False
    if direct_load_data:
        owned_images = EigerImages(data_path, images_per_file, md)
        images = owned_images
        apply_reverse = reverse
        apply_rot90 = rot90

    Nimg_ = len(images)
    avg_img = np.zeros(mask.shape, dtype=np.float64)
    Nopix = float(avg_img.size)
    n = 0
    good_count = 0
    frac = 0.0
    if nobytes == 2:
        dtype = np.int16
    elif nobytes == 4:
        dtype = np.int32
    elif nobytes == 8:
        dtype = np.float64
    else:
        print("Wrong type of nobytes, only support 2 [np.int16] or 4 [np.int32]")
        dtype = np.int32

    time_edge = _frame_bin_edges(Nimg_, bins)
    Nimg = len(time_edge)

    imgsum = np.zeros(Nimg)
    if bins != 1:
        print("The frames will be binned by %s" % bins)

    try:
        image_iterator = _iter_binned_images(
            images,
            0,
            Nimg_,
            bins,
            reverse=apply_reverse,
            rot90=apply_rot90,
        )
        for n, image in enumerate(tqdm(image_iterator, total=Nimg)):
            mask &= image < hot_pixel_threshold
            p = np.where((np.ravel(image) > 0) & np.ravel(mask))[0]  # don't use masked data
            v = np.ravel(np.array(image, dtype=dtype))[p]
            dlen = len(p)
            imgsum[n] = v.sum()
            if (imgsum[n] > bad_pixel_threshold) or (imgsum[n] <= bad_pixel_low_threshold):
                _write_sparse_frame(fp, (), ())
            else:
                np.ravel(avg_img)[p] += v
                good_count += 1
                frac += dlen / Nopix
                _write_sparse_frame(fp, p, v)
    finally:
        fp.close()
        if owned_images is not None:
            owned_images.close()
    if good_count:
        frac /= good_count
    else:
        frac = 0.0
    print("The fraction of pixel occupied by photon is %6.3f%% " % (100 * frac))
    if good_count:
        avg_img /= good_count
    else:
        avg_img.fill(np.nan)

    bad_frame_list = np.where(
        (np.array(imgsum) > bad_pixel_threshold) | (np.array(imgsum) <= bad_pixel_low_threshold)
    )[0]
    # bad_frame_list1 = np.where( np.array(imgsum) > bad_pixel_threshold  )[0]
    # bad_frame_list2 = np.where( np.array(imgsum) < bad_pixel_low_threshold  )[0]
    # bad_frame_list =   np.unique( np.concatenate( [bad_frame_list1, bad_frame_list2]) )

    if len(bad_frame_list):
        print("Bad frame list are: %s" % bad_frame_list)
    else:
        print("No bad frames are involved.")
    if with_pickle:
        with open(filename + ".pkl", "wb") as stream:
            pkl.dump([mask, avg_img, imgsum, bad_frame_list], stream)
    return mask, avg_img, imgsum, bad_frame_list


"""    Description:

    This is code that Mark wrote to open the multifile format
    in compressed mode, translated to python.
    This seems to work for DALSA, FCCD and EIGER in compressed mode.
    It should be included in the respective detector.i files
    Currently, this refers to the compression mode being '6'
    Each file is image descriptor files chunked together as follows:
            Header (1024 bytes)
    |--------------IMG N begin--------------|
    |                   Dlen
    |---------------------------------------|
    |       Pixel positions (dlen*4 bytes   |
    |      (0 based indexing in file)       |
    |---------------------------------------|
    |    Pixel data(dlen*bytes bytes)       |
    |    (bytes is found in header          |
    |    at position 116)                   |
    |--------------IMG N end----------------|
    |--------------IMG N+1 begin------------|
    |----------------etc.....---------------|


     Header contains 1024 bytes version name, 'beam_center_x', 'beam_center_y', 'count_time', 'detector_distance',
           'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size',
           bytes per pixel (either 2 or 4 (Default)),
           Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End,



"""


class Multifile:
    """The class representing the multifile.
    The recno is in 1 based numbering scheme (first record is 1)
    This is efficient for reading in increasing order.
    Note: reading same image twice in a row is like reading an earlier
    numbered image and means the program starts for the beginning again.

    """

    def __init__(self, filename, beg, end, reverse=False):
        """Multifile initialization. Open the file.
        Here I use the read routine which returns byte objects
        (everything is an object in python). I use struct.unpack
        to convert the byte object to other data type (int object
        etc)
        NOTE: At each record n, the file cursor points to record n+1
        """
        self.FID = open(filename, "rb")
        #        self.FID.seek(0,os.SEEK_SET)
        self.filename = filename
        # br: bytes read
        br = self.FID.read(1024)
        if len(br) != 1024:
            self.FID.close()
            raise ValueError("malformed compressed file: incomplete 1024-byte header")
        self.beg = beg
        self.end = end
        self.reverse = reverse
        ms_keys = [
            "beam_center_x",
            "beam_center_y",
            "count_time",
            "detector_distance",
            "frame_time",
            "incident_wavelength",
            "x_pixel_size",
            "y_pixel_size",
            "bytes",
            "nrows",
            "ncols",
            "rows_begin",
            "rows_end",
            "cols_begin",
            "cols_end",
        ]

        version = struct.unpack("@16s", br[:16])[0]
        if version != b"Version-COMP0001":
            self.FID.close()
            raise ValueError("unsupported compressed file header")
        md_temp = struct.unpack("@8d7I916x", br[16:])
        self.md = dict(zip(ms_keys, md_temp))

        self.imgread = 0
        self.recno = 0
        self._mmap = None
        self._frame_offsets = None
        self._bytes_traversed = 0

        if reverse:
            nrows = self.md["nrows"]
            ncols = self.md["ncols"]
            self.md["nrows"] = ncols
            self.md["ncols"] = nrows
            rbeg = self.md["rows_begin"]
            rend = self.md["rows_end"]
            cbeg = self.md["cols_begin"]
            cend = self.md["cols_end"]
            self.md["rows_begin"] = cbeg
            self.md["rows_end"] = cend
            self.md["cols_begin"] = rbeg
            self.md["cols_end"] = rend

        # some initialization stuff
        self.byts = self.md["bytes"]
        if self.byts == 2:
            self.valtype = np.uint16
        elif self.byts == 4:
            self.valtype = np.uint32
        elif self.byts == 8:
            self.valtype = np.float64
        else:
            self.FID.close()
            raise ValueError("malformed compressed file: bytes per value must be 2, 4, or 8")
        # now convert pieces of these bytes to our data
        first_length = np.fromfile(self.FID, dtype=np.int32, count=1)
        if first_length.size != 1:
            self.FID.close()
            raise ValueError("malformed compressed file: missing first frame")
        self.dlen = first_length[0]
        if self.dlen < 0:
            self.FID.close()
            raise ValueError("malformed compressed file: negative sparse-frame length")

        # now read first image
        # print "Opened file. Bytes per data is {0img.shape = (self.rows,self.cols)}".format(self.byts)

    def _readHeader(self):
        length = np.fromfile(self.FID, dtype=np.int32, count=1)
        if length.size != 1:
            raise ValueError("malformed compressed file: truncated frame header")
        self.dlen = length[0]
        if self.dlen < 0:
            raise ValueError("malformed compressed file: negative sparse-frame length")

    def _readImageRaw(self):

        p = np.fromfile(self.FID, dtype=np.int32, count=self.dlen)
        v = np.fromfile(self.FID, dtype=self.valtype, count=self.dlen)
        if p.size != self.dlen or v.size != self.dlen:
            raise ValueError("malformed compressed file: truncated sparse-frame payload")
        self.imgread = 1
        return (p, v)

    def _ensure_index(self):
        """Build and validate an in-memory frame-offset index on first use."""
        if self._frame_offsets is not None:
            return self._frame_offsets
        import mmap

        if self.FID.closed:
            raise ValueError("I/O operation on closed compressed file")
        mapped = mmap.mmap(self.FID.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = []
        position = 1024
        file_size = len(mapped)
        try:
            for frame in range(self.end):
                if position + 4 > file_size:
                    raise ValueError("malformed compressed file: requested frame range extends past end of file")
                length = struct.unpack_from("@i", mapped, position)[0]
                if length < 0:
                    raise ValueError("malformed compressed file: negative sparse-frame length")
                next_position = position + 4 + length * (4 + self.byts)
                if next_position > file_size:
                    raise ValueError("malformed compressed file: truncated sparse-frame payload")
                offsets.append(position)
                position = next_position
        except BaseException:
            mapped.close()
            raise
        self._mmap = mapped
        self._frame_offsets = np.asarray(offsets, dtype=np.int64)
        self._bytes_traversed += 4 * len(offsets)
        return self._frame_offsets

    def _raw_frame_view(self, n):
        """Return read-only zero-copy position/value views for private consumers."""
        if n < self.beg or n >= self.end:
            raise IndexError("Error, record out of range")
        offsets = self._ensure_index()
        offset = int(offsets[n])
        length = struct.unpack_from("@i", self._mmap, offset)[0]
        positions = np.frombuffer(self._mmap, dtype=np.int32, count=length, offset=offset + 4)
        values = np.frombuffer(
            self._mmap,
            dtype=self.valtype,
            count=length,
            offset=offset + 4 + length * np.dtype(np.int32).itemsize,
        )
        positions.flags.writeable = False
        values.flags.writeable = False
        self._bytes_traversed += 4 + length * (np.dtype(np.int32).itemsize + self.byts)
        return positions, values

    def _iter_raw_frames(self, indices):
        self._ensure_index()
        for index in indices:
            yield index, self._raw_frame_view(index)

    def _reset_io_counters(self):
        self._bytes_traversed = 0

    def _readImage(self):
        p, v = self._readImageRaw()
        img = np.zeros((self.md["ncols"], self.md["nrows"]))
        np.put(np.ravel(img), p, v)
        return img

    def seekimg(self, n=None):
        """Position file to read the nth image.
        For now only reads first image ignores n
        """
        # the logic involving finding the cursor position
        if n is None:
            n = self.recno
        if n < self.beg or n >= self.end:
            raise IndexError("Error, record out of range")
        # print (n, self.recno, self.FID.tell() )
        if (n == self.recno) and (self.imgread == 0):
            pass  # do nothing

        else:
            if n <= self.recno:  # ensure cursor less than search pos
                self.FID.seek(1024, os.SEEK_SET)
                self.dlen = np.fromfile(self.FID, dtype=np.int32, count=1)[0]
                self.recno = 0
                self.imgread = 0
                if n == 0:
                    return
            # have to iterate on seeking since dlen varies
            # remember for rec recno, cursor is always at recno+1
            if self.imgread == 0:  # move to next header if need to
                self.FID.seek(self.dlen * (4 + self.byts), os.SEEK_CUR)
            for i in range(self.recno + 1, n):
                # the less seeks performed the faster
                # print (i)
                self.dlen = np.fromfile(self.FID, dtype=np.int32, count=1)[0]
                # print 's',self.dlen
                self.FID.seek(self.dlen * (4 + self.byts), os.SEEK_CUR)

            # we are now at recno in file, read the header and data
            # self._clearImage()
            self._readHeader()
            self.imgread = 0
            self.recno = n

    def rdframe(self, n):
        if self.seekimg(n) != -1:
            return self._readImage()

    def rdrawframe(self, n):
        if self.seekimg(n) != -1:
            return self._readImageRaw()

    def close(self):
        """Close the compressed-data file."""
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # A private zero-copy view may briefly outlive this object.
                pass
            self._mmap = None
            self._frame_offsets = None
        if not self.FID.closed:
            self.FID.close()

    def reopen(self):
        """Reopen a closed compressed file and reset its sequential cursor."""
        if not self.FID.closed:
            return self
        replacement = type(self)(self.filename, self.beg, self.end, self.reverse)
        self.__dict__.update(replacement.__dict__)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_closed"] = self.FID.closed
        state.pop("FID", None)
        state.pop("_mmap", None)
        state.pop("_frame_offsets", None)
        return state

    def __setstate__(self, state):
        replacement = type(self)(state["filename"], state["beg"], state["end"], state["reverse"])
        self.__dict__.update(replacement.__dict__)
        if state.get("_closed", False):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class Multifile_Bins(object):
    """
    Bin a compressed file with bins number
    See Multifile for details for Multifile_class
    """

    def __init__(self, FD, bins=100):
        """
        FD: the handler of a compressed Eiger frames
        bins: bins number
        """

        self.FD = FD
        if (FD.end - FD.beg) % bins:
            print("Please give a better bins number and make the length of FD/bins= integer")
        else:
            self.bins = bins
            self.md = FD.md
            # self.beg = FD.beg
            self.beg = 0
            Nimg = FD.end - FD.beg
            slice_num = Nimg // bins
            self.end = slice_num
            self.time_edge = np.array(create_time_slice(N=Nimg, slice_num=slice_num, slice_width=bins)) + FD.beg
            self.get_bin_frame()

    def get_bin_frame(self):
        FD = self.FD
        self.frames = np.zeros([FD.md["ncols"], FD.md["nrows"], len(self.time_edge)])
        for n in tqdm(range(len(self.time_edge))):
            # print (n)
            t1, t2 = self.time_edge[n]
            # print( t1, t2)
            self.frames[:, :, n] = get_avg_imgc(FD, beg=t1, end=t2, sampling=1, plot_=False, show_progress=False)

    def rdframe(self, n):
        return self.frames[:, :, n]

    def rdrawframe(self, n):
        x_ = np.ravel(self.rdframe(n))
        p = np.where(x_)[0]
        v = np.array(x_[p])
        return (np.array(p, dtype=np.int32), v)


class MultifileBNL:
    """
    Re-write multifile from scratch.
    """

    HEADER_SIZE = 1024

    def __init__(self, filename, mode="rb"):
        """
        Prepare a file for reading or writing.
        mode : either 'rb' or 'wb'
        """
        if mode == "wb":
            raise ValueError("Write mode 'wb' not supported yet")
        if mode != "rb" and mode != "wb":
            raise ValueError("Error, mode must be 'rb' or 'wb'" "got : {}".format(mode))
        self._filename = filename
        self._mode = mode
        # open the file descriptor
        # create a memmap
        if mode == "rb":
            self._fd = np.memmap(filename, dtype="c")
        elif mode == "wb":
            self._fd = open(filename, "wb")
        # these are only necessary for writing
        self.md = self._read_main_header()
        self._cols = int(self.md["nrows"])
        self._rows = int(self.md["ncols"])
        # some initialization stuff
        self.nbytes = self.md["bytes"]
        if self.nbytes == 2:
            self.valtype = "<i2"  # np.uint16
        elif self.nbytes == 4:
            self.valtype = "<i4"  # np.uint32
        elif self.nbytes == 8:
            self.valtype = "<i8"  # np.float64
        # frame number currently on
        self.index()

    def index(self):
        """Index the file by reading all frame_indexes.
        For faster later access.
        """
        print("Indexing file...")
        t1 = time.time()
        cur = self.HEADER_SIZE
        file_bytes = len(self._fd)
        self.frame_indexes = list()
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes
            dlen = np.frombuffer(self._fd[cur : cur + 4], dtype="<u4")[0]
            # print("found {} bytes".format(dlen))
            # self.nbytes is number of bytes per val
            cur += 4 + dlen * (4 + self.nbytes)
            # break
        self.Nframes = len(self.frame_indexes)
        t2 = time.time()
        print("Done. Took {} secs for {} frames".format(t2 - t1, self.Nframes))

    def _read_main_header(self):
        """Read header from current seek position.
        Extracting the header was written by Yugang Zhang. This is BNL's
        format.
        1024 byte header +
        4 byte dlen + (4 + nbytes)*dlen bytes
        etc...
        Format:
            unsigned int beam_center_x;
            unsigned int beam_center_y;
        """
        # read in bytes
        # header is always from zero
        cur = 0
        header_raw = self._fd[cur : cur + self.HEADER_SIZE]
        ms_keys = [
            "beam_center_x",
            "beam_center_y",
            "count_time",
            "detector_distance",
            "frame_time",
            "incident_wavelength",
            "x_pixel_size",
            "y_pixel_size",
            "bytes",
            "nrows",
            "ncols",
            "rows_begin",
            "rows_end",
            "cols_begin",
            "cols_end",
        ]
        _ = struct.unpack("@16s", header_raw[:16])
        md_temp = struct.unpack("@8d7I916x", header_raw[16:])
        self.md = dict(zip(ms_keys, md_temp))
        return self.md

    def _read_raw(self, n):
        """Read from raw.
        Reads from current cursor in file.
        """
        if n > self.Nframes:
            raise KeyError("Error, only {} frames, asked for {}".format(self.Nframes, n))
        # dlen is 4 bytes
        cur = self.frame_indexes[n]
        dlen = np.frombuffer(self._fd[cur : cur + 4], dtype="<u4")[0]
        cur += 4
        pos = self._fd[cur : cur + dlen * 4]
        cur += dlen * 4
        pos = np.frombuffer(pos, dtype="<u4")
        # TODO: 2-> nbytes
        vals = self._fd[cur : cur + dlen * self.nbytes]
        vals = np.frombuffer(vals, dtype=self.valtype)
        return pos, vals

    def rdframe(self, n):
        # read header then image
        pos, vals = self._read_raw(n)
        img = np.zeros((self._rows * self._cols,))
        img[pos] = vals
        return img.reshape((self._rows, self._cols))

    def rdrawframe(self, n):
        # read header then image
        return self._read_raw(n)


class MultifileBNLCustom(MultifileBNL):
    def __init__(self, filename, beg=0, end=None, **kwargs):
        super().__init__(filename, **kwargs)
        self.beg = beg
        if end is None:
            end = self.Nframes - 1
        self.end = end

    def rdframe(self, n):
        if n > self.end or n < self.beg:
            raise IndexError("Index out of range")
        # return super().rdframe(n - self.beg)
        return super().rdframe(n)

    def rdrawframe(self, n):
        # return super().rdrawframe(n - self.beg)
        if n > self.end or n < self.beg:
            raise IndexError("Index out of range")
        return super().rdrawframe(n)


def get_avg_imgc(
    FD, beg=None, end=None, sampling=100, plot_=False, bad_frame_list=None, show_progress=True, *argv, **kwargs
):
    """Get average imagef from a data_series by every sampling number to save time"""
    if beg is None:
        beg = FD.beg
    if end is None:
        end = FD.end
    if sampling < 1:
        raise ValueError("sampling must be at least one")

    bad_frames = set() if bad_frame_list is None else set(np.atleast_1d(bad_frame_list).tolist())
    sample_indices = [index for index in range(beg, end, sampling) if index not in bad_frames]
    avg_img = np.zeros((FD.md["ncols"], FD.md["nrows"]), dtype=np.float64)
    indices = (
        tqdm(sample_indices, desc="Averaging %s images" % len(sample_indices)) if show_progress else sample_indices
    )
    for index in indices:
        if hasattr(FD, "_raw_frame_view"):
            p, v = FD._raw_frame_view(index)
        else:
            p, v = FD.rdrawframe(index)
        sparse_add_image(p, v, np.ravel(avg_img))

    if sample_indices:
        avg_img /= len(sample_indices)
    else:
        avg_img.fill(np.nan)
    if plot_:
        if RUN_GUI:
            fig = Figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots()
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        im = ax.imshow(avg_img, cmap="viridis", origin="lower", norm=LogNorm(vmin=0.001, vmax=1e2))
        # ax.set_title("Masked Averaged Image")
        ax.set_title("uid= %s--Masked-Averaged-Image-" % uid)
        fig.colorbar(im)
        if kwargs.get("save", False):
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = kwargs["path"]
            if "uid" in kwargs:
                uid = kwargs["uid"]
            else:
                uid = "uid"
            # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
            fp = path + "uid=%s--avg-img-" % uid + ".png"
            plt.savefig(fp, dpi=fig.dpi)
        # plt.show()
    return avg_img


def mean_intensityc(FD, labeled_array, sampling=1, index=None, multi_cor=False):
    """Compute the mean intensity for each ROI in the compressed file (FD), support parallel computation

    Parameters
    ----------
    FD: Multifile class
        compressed file
    labeled_array : array
        labeled array; 0 is background.
        Each ROI is represented by a nonzero integer. It is not required that
        the ROI labels are contiguous
    index : int, list, optional
        The ROI's to use. If None, this function will extract averages for all
        ROIs

    Returns
    -------
    mean_intensity : array
        The mean intensity of each ROI for all `images`
        Shape is ``(number_of_sampled_frames, len(index))``.
    index : list
        The labels for each element of the `mean_intensity` list
    """

    qind, pixelist = roi.extract_label_indices(labeled_array)
    sx, sy = FD.md["ncols"], FD.md["nrows"]
    if labeled_array.shape != (sx, sy):
        raise ValueError(
            " `image` shape (%d, %d) in FD is not equal to the labeled_array shape (%d, %d)"
            % (sx, sy, labeled_array.shape[0], labeled_array.shape[1])
        )
    # Remap the selected labels to a dense, one-based range for
    # ``np.bincount``. ROI labels need not be contiguous in the input mask.
    available_labels = np.unique(qind)
    if index is None:
        index = list(available_labels)
    else:
        index = np.atleast_1d(index)
        missing_labels = np.setdiff1d(index, available_labels)
        if missing_labels.size:
            raise ValueError(f"ROI labels not present in labeled_array: {missing_labels.tolist()}")

    if len(index) == 0:
        raise ValueError("labeled_array contains no positive ROI labels")

    remapped_qind = np.zeros_like(qind, dtype=np.int64)
    for output_label, input_label in enumerate(index, start=1):
        remapped_qind[qind == input_label] = output_label
    selected = remapped_qind > 0
    qind = remapped_qind[selected]
    pixelist = pixelist[selected]

    # pre-allocate an array for performance
    # might be able to use list comprehension to make this faster

    sample_indices = range(FD.beg, FD.end, sampling)
    mean_intensity = np.zeros([len(sample_indices), len(index)])
    roi_lookup = np.full(FD.md["ncols"] * FD.md["nrows"], -1, dtype=np.int64)
    roi_lookup[pixelist] = qind - 1
    norm = np.bincount(qind, minlength=len(index) + 1)[1:]
    for output_row, frame_index in enumerate(tqdm(sample_indices, desc="Get ROI intensity of each frame")):
        # This is a strictly forward, one-pass scan.  Buffered reads are much
        # more predictable than page-faulting an mmap on network filesystems
        # such as Lustre, while still populating the page cache for later
        # mmap-based correlation work.
        positions, values = FD.rdrawframe(frame_index)
        sparse_roi_sums(positions, values, roi_lookup, mean_intensity[output_row])

    mean_intensity /= norm
    return mean_intensity, index


def _get_mean_intensity_one_q(FD, sampling, labels):
    mi = np.zeros(len(range(FD.beg, FD.end, sampling)))
    n = 0
    qind, pixelist = roi.extract_label_indices(labels)
    # iterate over the images to compute multi-tau correlation
    _ = np.zeros_like(pixelist, dtype=np.float64)
    timg = np.zeros(FD.md["ncols"] * FD.md["nrows"], dtype=np.int32)
    timg[pixelist] = np.arange(1, len(pixelist) + 1)
    for i in range(FD.beg, FD.end, sampling):
        p, v = FD.rdrawframe(i)
        w = np.where(timg[p])[0]
        pxlist = timg[p[w]] - 1
        mi[n] = np.bincount(qind[pxlist], weights=v[w], minlength=2)[1]
        n += 1
    return mi


def get_each_frame_intensityc(
    FD,
    sampling=1,
    bad_pixel_threshold=1e10,
    bad_pixel_low_threshold=0,
    hot_pixel_threshold=2**30,
    plot_=False,
    bad_frame_list=None,
    save=False,
    *argv,
    **kwargs,
):
    """Get the total intensity of each frame by sampling every N frames
    Also get bad_frame_list by check whether above  bad_pixel_threshold

    Usuage:
    imgsum, bad_frame_list = get_each_frame_intensity(good_series ,sampling = 1000,
                             bad_pixel_threshold=1e10,  plot_ = True)
    """

    # print ( argv, kwargs )
    # mask &= img < hot_pixel_threshold
    sample_indices = np.arange(FD.beg, FD.end, sampling)
    imgsum = np.zeros(len(sample_indices))
    n = 0
    for i in tqdm(sample_indices, desc="Get each frame intensity"):
        if hasattr(FD, "_raw_frame_view"):
            _, v = FD._raw_frame_view(i)
        else:
            _, v = FD.rdrawframe(i)
        if len(v) > 0:
            imgsum[n] = sparse_frame_sum(v)
        n += 1

    if plot_:
        uid = "uid"
        if "uid" in kwargs.keys():
            uid = kwargs["uid"]
        fig, ax = plt.subplots()
        ax.plot(imgsum, "bo")
        ax.set_title("uid= %s--imgsum" % uid)
        ax.set_xlabel("Frame_bin_%s" % sampling)
        ax.set_ylabel("Total_Intensity")

        if save:
            # dt =datetime.now()
            # CurTime = '%s%02d%02d-%02d%02d-' % (dt.year, dt.month, dt.day,dt.hour,dt.minute)
            path = kwargs["path"]
            if "uid" in kwargs:
                uid = kwargs["uid"]
            else:
                uid = "uid"
            # fp = path + "uid= %s--Waterfall-"%uid + CurTime + '.png'
            fp = path + "uid=%s--imgsum-" % uid + ".png"
            fig.savefig(fp, dpi=fig.dpi)

        plt.show()

    bad_frame_list_ = sample_indices[(imgsum > bad_pixel_threshold) | (imgsum <= bad_pixel_low_threshold)]

    if bad_frame_list is not None:
        bad_frame_list = np.unique(np.concatenate([bad_frame_list, bad_frame_list_]))
    else:
        bad_frame_list = bad_frame_list_

    if len(bad_frame_list):
        print("Bad frame list length is: %s" % len(bad_frame_list))
    else:
        print("No bad frames are involved.")
    return imgsum, bad_frame_list
