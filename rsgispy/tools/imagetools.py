"""
The tools.imagetools module contains functions for non-spatial image, EXIF,
PDF image, and QR code utilities.
"""

import os
import shutil
from typing import Any, Dict, List, Optional

import rsgispy
import rsgispy.tools.filetools
import rsgispy.tools.utils

__all__ = [
    "animate_img_set",
    "list_exif_info_pil",
    "list_exif_info",
    "get_exif_info",
    "extract_images_from_pdf",
    "add_pdf_blank_pages",
    "create_qr_code",
]


def animate_img_set(
    input_imgs: List[str],
    output_img: str,
    fps: int = 1,
    loop: int = 0,
    resize: bool = False,
    out_x_size: int = 720,
    out_y_size: Optional[int] = None,
    tmp_dir: str = "./tmp",
) -> None:
    """
    Animate a list of input image files into a GIF using ``imageio``.

    :param input_imgs: List of input image file paths.
    :param output_img: Output animated GIF file path.
    :param fps: Frames per second (Default = ``1``).
    :param loop: Loop count (``0`` loops forever; Default = ``0``).
    :param resize: Whether to resize images with PIL before animating (Default = ``False``).
    :param out_x_size: Target width when ``resize=True`` (Default = ``720``).
    :param out_y_size: Target height when ``resize=True`` (computed proportionally if ``None``).
    :param tmp_dir: Directory for temporary resized images when ``resize=True``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,import-outside-toplevel
    import imageio.v2

    tqdm_cls = rsgispy.get_tqdm()
    uid_str = rsgispy.tools.utils.uid_generator()
    lcl_tmp_dir = os.path.join(tmp_dir, f"tmp_{uid_str}")

    if resize:
        from PIL import Image

        os.makedirs(lcl_tmp_dir, exist_ok=True)
        ani_input_imgs: List[str] = []
        for in_img in tqdm_cls(input_imgs, desc="Resizing"):
            basename = rsgispy.tools.filetools.get_file_basename(in_img)
            out_resize_img = os.path.join(lcl_tmp_dir, f"{basename}_resize.png")
            with Image.open(in_img) as lcl_img_obj:
                target_y = out_y_size
                if target_y is None:
                    in_x_size, in_y_size = lcl_img_obj.size
                    target_y = int(out_x_size * (in_y_size / in_x_size))
                lcl_img_resize_obj = lcl_img_obj.resize((out_x_size, target_y))
                lcl_img_resize_obj.save(out_resize_img)
            ani_input_imgs.append(out_resize_img)
    else:
        ani_input_imgs = list(input_imgs)

    try:
        with imageio.v2.get_writer(
            output_img, mode="i", fps=fps, loop=loop
        ) as writer:
            for in_img in tqdm_cls(ani_input_imgs, desc="Animating"):
                image = imageio.v2.imread(in_img)
                writer.append_data(image)
    finally:
        if resize and os.path.exists(lcl_tmp_dir):
            shutil.rmtree(lcl_tmp_dir)


def list_exif_info_pil(input_img: str) -> None:
    """
    Print EXIF metadata from ``input_img`` using ``PIL``.

    :param input_img: Path to the input image file.
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, IFD, TAGS

    with Image.open(input_img) as pil_image:
        pil_exif_data = pil_image.getexif()
        for ifd_id in IFD:
            print(">>>>>>>>>", ifd_id.name, "<<<<<<<<<<")
            try:
                ifd = pil_exif_data.get_ifd(ifd_id)
                resolve = GPSTAGS if ifd_id == IFD.GPSInfo else TAGS
                for key, val in ifd.items():
                    tag = resolve.get(key, key)
                    if tag not in ("UserComment", "PrintImageMatching", "MakerNote"):
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        print(f"{tag}:\t {val}")
            except KeyError:
                pass


def list_exif_info(input_img: str) -> None:
    """
    Print EXIF metadata from ``input_img`` using ``exiftool``.

    :param input_img: Path to the input image file.
    """
    exif_dict = get_exif_info(input_img)
    for exif_tag, exif_val in exif_dict.items():
        print(f"{exif_tag}:\t{exif_val}")


def get_exif_info(input_img: str) -> Dict[str, Any]:
    """
    Return a dictionary of EXIF metadata from ``input_img`` using ``exiftool``.

    :param input_img: Path to the input image file.
    :return: Dictionary mapping EXIF tag names to metadata values.
    """
    # pylint: disable=import-outside-toplevel
    import exiftool

    out_exif_data: Dict[str, Any] = {}
    with exiftool.ExifToolHelper() as et_helper:
        exif_data = et_helper.get_metadata(input_img)
        for exif_item in exif_data:
            for exif_tag, exif_val in exif_item.items():
                out_exif_data[exif_tag] = exif_val
    return out_exif_data


def _open_pdf_doc(input_pdf: str):
    """Open a PDF document using PyMuPDF (``pymupdf`` or ``fitz``)."""
    # pylint: disable=import-outside-toplevel
    try:
        import pymupdf as fitz_mod
    except ImportError:
        import fitz as fitz_mod
    return fitz_mod, fitz_mod.Document(input_pdf)


def extract_images_from_pdf(input_pdf: str, output_dir: str) -> None:
    """
    Extract embedded images from a PDF file and save them as PNG files.

    :param input_pdf: Path to the input PDF file.
    :param output_dir: Output directory where PNG images will be saved.
    """
    tqdm_cls = rsgispy.get_tqdm()
    os.makedirs(output_dir, exist_ok=True)
    fitz_mod, doc = _open_pdf_doc(input_pdf)
    try:
        for page_idx in tqdm_cls(range(len(doc)), desc="pages"):
            for img_info in tqdm_cls(
                doc.get_page_images(page_idx), desc="Page", leave=False
            ):
                xref = img_info[0]
                pix = fitz_mod.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz_mod.Pixmap(fitz_mod.csRGB, pix)
                pix.save(os.path.join(output_dir, f"p{page_idx}_{xref}.png"))
    finally:
        doc.close()


def add_pdf_blank_pages(input_pdf: str, output_pdf: str) -> None:
    """
    Insert a blank page after each page in an existing PDF document.

    :param input_pdf: Path to the input PDF file.
    :param output_pdf: Path to the output PDF file.
    """
    _, pdf_doc = _open_pdf_doc(input_pdf)
    try:
        n_pages = pdf_doc.page_count
        n_add_pages = 1
        for page_idx in range(n_pages):
            pdf_doc.new_page(pno=page_idx + n_add_pages)
            n_add_pages += 1
        pdf_doc.save(output_pdf)
    finally:
        pdf_doc.close()


def create_qr_code(
    output_img: str,
    qr_data: str,
    qr_code_size: int = 10,
    qr_code_border: int = 4,
) -> None:
    """
    Create a QR code image file encoding ``qr_data``.

    :param output_img: Output image file path.
    :param qr_data: String payload to encode in the QR code.
    :param qr_code_size: Pixel box size for each QR module (Default = ``10``).
    :param qr_code_border: Quiet-zone border size in boxes (Default = ``4``).
    """
    # pylint: disable=import-outside-toplevel
    import qrcode

    qr_obj = qrcode.QRCode(
        version=5,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=qr_code_size,
        border=qr_code_border,
    )
    qr_obj.add_data(qr_data)
    qr_obj.make(fit=True)

    img = qr_obj.make_image(fill_color="black", back_color="white")
    img.save(output_img)
