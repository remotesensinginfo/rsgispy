"""
The tools.checkdatasets module contains functions for validating raster,
vector, and HDF5 datasets using ``rasterio``, ``pyogrio``, and optionally ``h5py``
without requiring GDAL's ``osgeo`` bindings.

Functions requiring ``h5py`` are only defined and exported when ``h5py`` is
installed.
"""

import glob
import multiprocessing
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy
import pyproj
import rasterio
from rasterio.windows import Window
import tqdm

import rsgispy.tools.files
import rsgispy.tools.jsonio

try:
    import h5py  # pylint: disable=import-error

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

__all__ = [
    "check_gdal_image_file",
    "run_check_gdal_image_file",
    "run_check_gdal_image_files",
    "check_gdal_vector_file",
    "run_check_gdal_vector_file",
    "run_check_gdal_vector_files",
    "cmp_to_ref_imgs",
]

if HAVE_H5PY:
    __all__.extend(
        [
            "check_hdf5_file",
            "run_check_hdf5_file",
            "run_check_hdf5_files",
        ]
    )


def _validate_file_exists_and_size(
    file_path: str,
    max_file_size: Optional[int],
) -> Tuple[bool, str]:
    """Verify that ``file_path`` exists, is non-empty, and is within size bounds."""
    if not os.path.exists(file_path):
        return False, "File does not exist."

    file_size_bytes = rsgispy.tools.files.get_file_size(file_path)
    if file_size_bytes == 0:
        return False, "File size is zero."
    if max_file_size is not None and file_size_bytes > max_file_size:
        return (
            False,
            f"File size is greater than max. threshold ({file_size_bytes}).",
        )
    return True, ""


def _resolve_epsg(crs_input: Any) -> Optional[int]:
    """Resolve an EPSG code from a ``rasterio`` CRS or WKT/PROJ string."""
    if crs_input is None:
        return None
    if hasattr(crs_input, "to_epsg"):
        epsg = crs_input.to_epsg()
        if epsg is not None:
            return int(epsg)
    try:
        crs_obj = pyproj.CRS.from_user_input(crs_input)
        return crs_obj.to_epsg()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _handle_failed_file(
    file_path: str,
    err_str: str,
    rm_err: bool,
    print_err: bool,
    multi_file: bool = False,
) -> None:
    """Print diagnostics and optionally remove an invalid file."""
    if print_err and err_str:
        print(f"Error: '{err_str}'")
    if rm_err:
        if multi_file:
            rsgispy.tools.files.delete_file_with_basename(file_path)
        else:
            rsgispy.tools.files.delete_file_silent(file_path)
            print(f"Removed {file_path}")
    else:
        print(f"rm {file_path}")


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _run_checks_in_pool(
    items: Sequence[str],
    worker_fn: Callable[[Tuple[Any, ...]], bool],
    build_params_fn: Callable[[str], Tuple[Any, ...]],
    timeout: int,
    rm_err: bool,
    print_file_names: bool = False,
    multi_file: bool = False,
) -> bool:
    """
    Execute per-file validation in an isolated subprocess pool so native crashes
    or hung reads do not bring down the caller process.
    """
    files_ok = True
    pool = multiprocessing.Pool(1)  # pylint: disable=consider-using-with
    try:
        for file_path in items:
            if print_file_names:
                print(file_path)
            params = build_params_fn(file_path)
            try:
                result = pool.apply_async(worker_fn, args=(params,))
                file_ok = result.get(timeout=timeout)
                if not file_ok:
                    files_ok = False
            except Exception as err:  # pylint: disable=broad-exception-caught
                files_ok = False
                _handle_failed_file(
                    file_path,
                    str(err),
                    rm_err=rm_err,
                    print_err=False,
                    multi_file=multi_file,
                )
                # Terminate a hung or crashed worker pool and start fresh for
                # subsequent files in the batch.
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(1)  # pylint: disable=consider-using-with
    finally:
        pool.terminate()
        pool.join()

    return files_ok


def _check_raster_bands_and_pixels(
    src: Any,
    check_bands: bool,
    n_bands: int,
    read_img: bool,
    smpl_n_pxls: int,
    calc_chk_sum: bool,
) -> Tuple[bool, str]:
    """Validate band counts, pixel window reads, and per-band checksums."""
    n_img_bands = src.count
    if n_bands > 0 and n_img_bands != n_bands:
        return (
            False,
            f"Image should have {n_bands} image bands but {n_img_bands} found.",
        )

    if check_bands and n_img_bands < 1:
        return False, "Image says it does not have any image bands."

    if read_img:
        if src.width < 1 or src.height < 1 or n_img_bands < 1:
            return False, "Image has zero width, height, or band count."
        rng = numpy.random.default_rng()
        band = 1 if n_img_bands == 1 else int(rng.integers(1, n_img_bands + 1))
        x_pxls = rng.integers(0, src.width, size=smpl_n_pxls)
        y_pxls = rng.integers(0, src.height, size=smpl_n_pxls)
        for x_off, y_off in zip(x_pxls, y_pxls):
            src.read(
                band,
                window=Window(int(x_off), int(y_off), 1, 1),
                out_dtype="float32",
            )

    if calc_chk_sum:
        for band in range(1, n_img_bands + 1):
            src.checksum(band)

    return True, ""


def _check_crs_epsg(
    crs_val: Any,
    chk_proj: bool,
    epsg_code: int,
    label: str,
) -> Tuple[bool, str]:
    """Validate CRS presence and optional EPSG code match."""
    if not chk_proj and epsg_code <= 0:
        return True, ""

    if crs_val is None or crs_val == "":
        return False, f"{label} projection is None."

    if hasattr(crs_val, "to_wkt") and not crs_val.to_wkt():
        return False, f"{label} projection is empty."

    if epsg_code > 0:
        resolved_epsg = _resolve_epsg(crs_val)
        if resolved_epsg is None:
            return False, f"{label} projection returned a None EPSG code."
        if int(resolved_epsg) != int(epsg_code):
            return (
                False,
                f"{label} EPSG ({resolved_epsg}) does not match "
                f"that specified ({epsg_code})",
            )

    return True, ""


def _check_raster_projection(
    src: Any,
    chk_proj: bool,
    epsg_code: int,
) -> Tuple[bool, str]:
    """Validate CRS presence and optional EPSG match on an open raster dataset."""
    return _check_crs_epsg(src.crs, chk_proj, epsg_code, label="Image")


# pylint: disable=too-many-arguments,too-many-positional-arguments
def check_gdal_image_file(
    input_img: str,
    check_bands: bool = True,
    n_bands: int = 0,
    chk_proj: bool = False,
    epsg_code: int = 0,
    read_img: bool = False,
    smpl_n_pxls: int = 10,
    calc_chk_sum: bool = False,
    max_file_size: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate a raster image file using ``rasterio`` (and ``h5py`` for ``.kea`` files).

    :param input_img: File path to the raster image.
    :param check_bands: Check that the image has at least one band (Default: ``True``).
    :param n_bands: Expected band count (ignored if ``<= 0``; Default: ``0``).
    :param chk_proj: Verify that a projection is defined (Default: ``False``).
    :param epsg_code: Expected EPSG code (ignored if ``<= 0``; Default: ``0``).
    :param read_img: Read ``smpl_n_pxls`` random single-pixel windows from a
                     randomly selected band to verify pixel decoding.
    :param smpl_n_pxls: Number of random pixels to read when ``read_img=True``.
    :param calc_chk_sum: Compute per-band checksums to verify raster block integrity.
    :param max_file_size: Maximum allowed file size in bytes (optional).
    :return: Tuple ``(file_ok, err_str)``.

    .. code:: python

        import rsgispy.tools.checkdatasets

        file_ok, err_str = rsgispy.tools.checkdatasets.check_gdal_image_file(
            "img.tif", chk_proj=True, epsg_code=32630
        )
    """
    file_ok, err_str = _validate_file_exists_and_size(input_img, max_file_size)
    if not file_ok:
        return False, err_str

    if HAVE_H5PY and os.path.splitext(input_img)[1].lower() == ".kea":
        # pylint: disable-next=possibly-used-before-assignment
        file_ok, err_str = check_hdf5_file(input_img)
        if not file_ok:
            return False, err_str or "Error with KEA/HDF5 file."

    try:
        with rasterio.open(input_img, "r") as src:
            file_ok, err_str = _check_raster_bands_and_pixels(
                src,
                check_bands=check_bands,
                n_bands=n_bands,
                read_img=read_img,
                smpl_n_pxls=smpl_n_pxls,
                calc_chk_sum=calc_chk_sum,
            )
            if file_ok:
                file_ok, err_str = _check_raster_projection(src, chk_proj, epsg_code)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return False, str(err)

    return file_ok, err_str


def _run_check_gdal_image_file(img_params: Sequence[Any]) -> bool:
    """Worker entrypoint for multiprocessing raster validation."""
    (
        input_img,
        check_bands,
        n_bands,
        chk_proj,
        epsg_code,
        read_img,
        smpl_n_pxls,
        calc_chk_sum,
        max_file_size,
        rm_err,
        print_err,
    ) = img_params

    try:
        file_ok, err_str = check_gdal_image_file(
            input_img,
            check_bands=check_bands,
            n_bands=n_bands,
            chk_proj=chk_proj,
            epsg_code=epsg_code,
            read_img=read_img,
            smpl_n_pxls=smpl_n_pxls,
            calc_chk_sum=calc_chk_sum,
            max_file_size=max_file_size,
        )
        if not file_ok:
            _handle_failed_file(input_img, err_str, rm_err=rm_err, print_err=print_err)
    except Exception as err:  # pylint: disable=broad-exception-caught
        _handle_failed_file(input_img, str(err), rm_err=rm_err, print_err=print_err)
        file_ok = False

    return file_ok


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_check_gdal_image_file(
    input_img: str,
    check_bands: bool = True,
    n_bands: int = 0,
    chk_proj: bool = False,
    epsg_code: int = 0,
    read_img: bool = False,
    smpl_n_pxls: int = 10,
    calc_chk_sum: bool = False,
    max_file_size: Optional[int] = None,
    rm_err: bool = False,
    print_err: bool = True,
    timeout: int = 4,
) -> bool:
    """
    Validate a single raster file inside a guarded subprocess with timeout.

    See :func:`check_gdal_image_file` for parameter descriptions.

    :param rm_err: Delete the file if validation fails or times out.
    :param print_err: Print failure details to stdout.
    :param timeout: Per-file timeout in seconds (Default: ``4``).
    :return: ``True`` if the file passed all checks, ``False`` otherwise.

    .. code:: python

        import rsgispy.tools.checkdatasets

        file_ok = rsgispy.tools.checkdatasets.run_check_gdal_image_file(
            "img.tif", chk_proj=True, epsg_code=32630, timeout=10
        )
    """
    return _run_checks_in_pool(
        [input_img],
        _run_check_gdal_image_file,
        lambda path: (
            path,
            check_bands,
            n_bands,
            chk_proj,
            epsg_code,
            read_img,
            smpl_n_pxls,
            calc_chk_sum,
            max_file_size,
            rm_err,
            print_err,
        ),
        timeout=timeout,
        rm_err=rm_err,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_check_gdal_image_files(
    input_imgs: List[str],
    check_bands: bool = True,
    n_bands: int = 0,
    chk_proj: bool = False,
    epsg_code: int = 0,
    read_img: bool = False,
    smpl_n_pxls: int = 10,
    calc_chk_sum: bool = False,
    max_file_size: Optional[int] = None,
    rm_err: bool = False,
    print_err: bool = True,
    print_file_names: bool = False,
    timeout: int = 4,
) -> bool:
    """
    Validate a list of raster files inside a guarded subprocess pool with timeout.

    See :func:`check_gdal_image_file` for parameter descriptions.

    :param input_imgs: List of input raster file paths.
    :param rm_err: Delete any invalid file.
    :param print_err: Print error messages for invalid files.
    :param print_file_names: Print each file path before checking it.
    :param timeout: Per-file timeout in seconds (Default: ``4``).
    :return: ``True`` if all files passed checks, ``False`` otherwise.

    .. code:: python

        import rsgispy.tools.checkdatasets

        all_ok = rsgispy.tools.checkdatasets.run_check_gdal_image_files(
            ["img1.tif", "img2.tif"], chk_proj=True, epsg_code=32630
        )
    """
    return _run_checks_in_pool(
        input_imgs,
        _run_check_gdal_image_file,
        lambda path: (
            path,
            check_bands,
            n_bands,
            chk_proj,
            epsg_code,
            read_img,
            smpl_n_pxls,
            calc_chk_sum,
            max_file_size,
            rm_err,
            print_err,
        ),
        timeout=timeout,
        rm_err=rm_err,
        print_file_names=print_file_names,
    )


def check_gdal_vector_file(
    vec_file: str,
    chk_proj: bool = True,
    epsg_code: int = 0,
    max_file_size: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate a vector file using ``pyogrio``.

    :param vec_file: Path to the vector file.
    :param chk_proj: Verify that a CRS is defined (Default: ``True``).
    :param epsg_code: Expected EPSG code (ignored if ``<= 0``; Default: ``0``).
    :param max_file_size: Maximum allowed file size in bytes (optional).
    :return: Tuple ``(file_ok, err_str)``.

    .. code:: python

        import rsgispy.tools.checkdatasets

        file_ok, err_str = rsgispy.tools.checkdatasets.check_gdal_vector_file(
            "polys.gpkg", chk_proj=True, epsg_code=4326
        )
    """
    # pylint: disable=import-outside-toplevel
    import pyogrio

    file_ok, err_str = _validate_file_exists_and_size(vec_file, max_file_size)
    if not file_ok:
        return False, err_str

    try:
        layers = pyogrio.list_layers(vec_file)
        if len(layers) == 0:
            return False, "Vector data source contains no layers."

        first_info = None
        for layer_name, _ in layers:
            info = pyogrio.read_info(vec_file, layer=layer_name)
            if first_info is None:
                first_info = info

        if chk_proj and first_info is not None:
            return _check_crs_epsg(
                first_info.get("crs"), chk_proj, epsg_code, label="Vector"
            )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return False, str(err)

    return True, ""


def _run_check_gdal_vector_file(file_params: Sequence[Any]) -> bool:
    """Worker entrypoint for multiprocessing vector validation."""
    vec_file, chk_proj, epsg_code, max_file_size, rm_err, print_err, multi_file = (
        file_params
    )

    try:
        file_ok, err_str = check_gdal_vector_file(
            vec_file,
            chk_proj=chk_proj,
            epsg_code=epsg_code,
            max_file_size=max_file_size,
        )
        if not file_ok:
            _handle_failed_file(
                vec_file,
                err_str,
                rm_err=rm_err,
                print_err=print_err,
                multi_file=multi_file,
            )
    except Exception as err:  # pylint: disable=broad-exception-caught
        _handle_failed_file(
            vec_file,
            str(err),
            rm_err=rm_err,
            print_err=print_err,
            multi_file=multi_file,
        )
        file_ok = False

    return file_ok


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_check_gdal_vector_file(
    vec_file: str,
    chk_proj: bool = True,
    epsg_code: int = 0,
    max_file_size: Optional[int] = None,
    rm_err: bool = False,
    print_err: bool = True,
    multi_file: bool = False,
    timeout: int = 4,
) -> bool:
    """
    Validate a single vector file inside a guarded subprocess with timeout.

    See :func:`check_gdal_vector_file` for parameter descriptions.

    :param rm_err: Delete the file if validation fails.
    :param print_err: Print failure details to stdout.
    :param multi_file: Delete all sibling files sharing the same basename (e.g., Shapefiles).
    :param timeout: Timeout in seconds (Default: ``4``).
    :return: ``True`` if the vector file is valid, ``False`` otherwise.

    .. code:: python

        import rsgispy.tools.checkdatasets

        file_ok = rsgispy.tools.checkdatasets.run_check_gdal_vector_file(
            "polys.gpkg", chk_proj=True, epsg_code=4326, timeout=10
        )
    """
    return _run_checks_in_pool(
        [vec_file],
        _run_check_gdal_vector_file,
        lambda path: (
            path,
            chk_proj,
            epsg_code,
            max_file_size,
            rm_err,
            print_err,
            multi_file,
        ),
        timeout=timeout,
        rm_err=rm_err,
        multi_file=multi_file,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_check_gdal_vector_files(
    vec_files: List[str],
    chk_proj: bool = True,
    epsg_code: int = 0,
    max_file_size: Optional[int] = None,
    rm_err: bool = False,
    print_err: bool = True,
    multi_file: bool = False,
    print_file_names: bool = False,
    timeout: int = 4,
) -> bool:
    """
    Validate a list of vector files inside a guarded subprocess pool with timeout.

    :param vec_files: List of input vector file paths.
    :return: ``True`` if all files are valid, ``False`` otherwise.

    .. code:: python

        import rsgispy.tools.checkdatasets

        all_ok = rsgispy.tools.checkdatasets.run_check_gdal_vector_files(
            ["polys1.gpkg", "polys2.gpkg"], chk_proj=True, epsg_code=4326
        )
    """
    return _run_checks_in_pool(
        vec_files,
        _run_check_gdal_vector_file,
        lambda path: (
            path,
            chk_proj,
            epsg_code,
            max_file_size,
            rm_err,
            print_err,
            multi_file,
        ),
        timeout=timeout,
        rm_err=rm_err,
        print_file_names=print_file_names,
        multi_file=multi_file,
    )


if HAVE_H5PY:

    def check_hdf5_file(input_file: str) -> Tuple[bool, str]:
        """
        Check whether an HDF5 file (including ``.kea`` rasters) is readable and valid.

        :param input_file: Path to the HDF5 file.
        :return: Tuple ``(is_valid, err_str)``.

        .. code:: python

            import rsgispy.tools.checkdatasets

            file_ok, err_str = rsgispy.tools.checkdatasets.check_hdf5_file("img.kea")
        """

        def _check_h5_var(h5_obj: Any) -> Tuple[bool, str]:
            try:
                if isinstance(h5_obj, h5py.Dataset):
                    _ = h5_obj.shape
                    _ = h5_obj.dtype
                    return True, ""
                if isinstance(h5_obj, h5py.Group):
                    for child_obj in h5_obj.values():
                        lcl_ok, lcl_err = _check_h5_var(child_obj)
                        if not lcl_ok:
                            return False, lcl_err
            except Exception as err:  # pylint: disable=broad-exception-caught
                return False, str(err)
            return True, ""

        if not os.path.exists(input_file):
            return False, "File does not exist."

        try:
            with h5py.File(input_file, "r") as f_h5:
                for child_obj in f_h5.values():
                    glb_ok, err_str = _check_h5_var(child_obj)
                    if not glb_ok:
                        return False, err_str
        except Exception as err:  # pylint: disable=broad-exception-caught
            return False, str(err)

        return True, ""

    def _run_check_hdf5_file(file_params: Sequence[Any]) -> bool:
        """Worker entrypoint for multiprocessing HDF5 validation."""
        input_file, rm_err, print_err = file_params

        try:
            file_ok, err_str = check_hdf5_file(input_file)
            if not file_ok:
                _handle_failed_file(
                    input_file, err_str, rm_err=rm_err, print_err=print_err
                )
        except Exception as err:  # pylint: disable=broad-exception-caught
            _handle_failed_file(input_file, str(err), rm_err=rm_err, print_err=print_err)
            file_ok = False

        return file_ok

    def run_check_hdf5_file(
        input_file: str,
        rm_err: bool = False,
        print_err: bool = True,
        timeout: int = 4,
    ) -> bool:
        """
        Validate a single HDF5 file inside a guarded subprocess with timeout.

        :param input_file: File path to the HDF5 file.
        :param rm_err: Delete the file if validation fails.
        :param print_err: Print failure details to stdout.
        :param timeout: Timeout in seconds (Default: ``4``).
        :return: ``True`` if the file passed checks, ``False`` otherwise.

        .. code:: python

            import rsgispy.tools.checkdatasets

            file_ok = rsgispy.tools.checkdatasets.run_check_hdf5_file(
                "img.kea", timeout=10
            )
        """
        return _run_checks_in_pool(
            [input_file],
            _run_check_hdf5_file,
            lambda path: (path, rm_err, print_err),
            timeout=timeout,
            rm_err=rm_err,
        )

    def run_check_hdf5_files(
        input_files: List[str],
        rm_err: bool = False,
        print_err: bool = True,
        print_file_names: bool = False,
        timeout: int = 4,
    ) -> bool:
        """
        Validate a list of HDF5 files inside a guarded subprocess pool with timeout.

        :param input_files: List of HDF5 file paths.
        :param rm_err: Delete invalid files.
        :param print_err: Print failure details to stdout.
        :param print_file_names: Print each file path before checking it.
        :param timeout: Per-file timeout in seconds (Default: ``4``).
        :return: ``True`` if all files passed checks, ``False`` otherwise.

        .. code:: python

            import rsgispy.tools.checkdatasets

            all_ok = rsgispy.tools.checkdatasets.run_check_hdf5_files(
                ["img1.kea", "img2.kea"], timeout=10
            )
        """
        return _run_checks_in_pool(
            input_files,
            _run_check_hdf5_file,
            lambda path: (path, rm_err, print_err),
            timeout=timeout,
            rm_err=rm_err,
            print_file_names=print_file_names,
        )


def _read_raster_geom_signature(
    img_path: str,
) -> Tuple[Tuple[int, int], Tuple[float, float, float, float], Optional[int]]:
    """Open a raster once and return ``((width, height), bbox, epsg)``."""
    with rasterio.open(img_path, "r") as src:
        bounds = src.bounds
        bbox = (float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top))
        epsg = _resolve_epsg(src.crs)
        return (int(src.width), int(src.height)), bbox, epsg


def cmp_to_ref_imgs(
    ref_imgs: List[str],
    input_img_dir: str,
    input_img_ext: str,
    rm_errs: bool = False,
    output_file: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Compare images in ``input_img_dir`` against reference images for matching
    dimensions, bounding box, and EPSG code.

    An input image is matched to a reference image when the reference image's
    basename appears within the input image's basename.

    :param ref_imgs: List of reference image paths.
    :param input_img_dir: Directory containing images to check.
    :param input_img_ext: File extension (without leading dot, e.g., ``"tif"``).
    :param rm_errs: Delete input images that fail comparison when ``True``.
    :param output_file: Optional JSON report output path.
    :return: Report dictionary with keys ``checked``, ``not_checked``,
             ``errors``, and ``ref_no_match``.

    .. code:: python

        import rsgispy.tools.checkdatasets

        report = rsgispy.tools.checkdatasets.cmp_to_ref_imgs(
            ref_imgs=["ref/tile_01.tif"],
            input_img_dir="outputs",
            input_img_ext="tif",
        )
    """
    input_imgs = glob.glob(os.path.join(input_img_dir, f"*.{input_img_ext}"))
    input_imgs_lut: Dict[str, List[Any]] = {
        rsgispy.tools.files.get_file_basename(img): [img, False]
        for img in input_imgs
    }

    out_report: Dict[str, List[str]] = {
        "checked": [],
        "not_checked": [],
        "errors": [],
        "ref_no_match": [],
    }
    n_errs = 0

    for ref_img in tqdm.tqdm(ref_imgs):
        ref_base = rsgispy.tools.files.get_file_basename(ref_img)
        matched_entries = [
            (chk_base, entry[0])
            for chk_base, entry in input_imgs_lut.items()
            if ref_base in chk_base
        ]
        if matched_entries:
            ref_size, ref_bbox, ref_epsg = _read_raster_geom_signature(ref_img)
            for chk_base, chk_img in matched_entries:
                chk_size, chk_bbox, chk_epsg = _read_raster_geom_signature(chk_img)
                error = (
                    (ref_size != chk_size)
                    or (ref_bbox != chk_bbox)
                    or (ref_epsg != chk_epsg)
                )
                if error:
                    out_report["errors"].append(chk_img)
                    n_errs += 1

                out_report["checked"].append(chk_img)
                input_imgs_lut[chk_base][1] = True

                if error and rm_errs:
                    rsgispy.tools.files.delete_file_silent(chk_img)
        else:
            out_report["ref_no_match"].append(ref_img)

    out_report["not_checked"] = [
        img_path for img_path, checked in input_imgs_lut.values() if not checked
    ]

    if output_file is not None:
        rsgispy.tools.jsonio.write_dict_to_json(out_report, output_file)

    print(f"Found {n_errs} images with errors...")
    return out_report
