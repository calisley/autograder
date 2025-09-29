#!/usr/bin/env python3
"""
Multithreaded PDF to Image Converter Using PyMuPDF (No Poppler Required)

Convert PDF files to PNG or JPEG images with specified dimensions.
This script uses PyMuPDF (fitz) which doesn't require external tools like Poppler.

Usage:
    pdf2img.py [-h] [--format {png,jpg,jpeg,tiff}] [--dpi DPI]
               [--width WIDTH] [--height HEIGHT] [--max-width MAX_WIDTH] [--max-height MAX_HEIGHT]
               [--quality QUALITY] [--output OUTPUT] [--pages PAGES]
               [--preserve-aspect-ratio] pdf_file

Examples:
    # Process a single file
    pdf2img.py document.pdf --format jpg --dpi 300 --output images/
"""

import argparse
import os
import resource
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import fitz  # PyMuPDF


def get_memory_usage():
    """Return current memory usage in MB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def process_pdf(args):
    """Process a single PDF file"""
    # Check if the PDF file exists
    pdf_file = args.pdf_file
    if not os.path.isfile(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found.", file=sys.stderr)
        return False

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base name for output files
    base_name = Path(pdf_file).stem

    # Determine which pages to convert and handle file existence check
    try:
        pages_to_convert = []
        # Use a context manager for the PDF document to ensure it gets closed
        with fitz.open(pdf_file) as doc:

            pages_to_convert = list(range(doc.page_count))

            if len(pages_to_convert) < doc.page_count:
                print(f"Converting {len(pages_to_convert)} of {doc.page_count} pages")

    except Exception as e:
        print(f"Error examining PDF '{pdf_file}': {str(e)}", file=sys.stderr)
        return False

    # Now process the PDF pages one by one
    # We'll open the PDF again to ensure we're not keeping the whole document in memory
    success = True
    fmt = args.format.lower()
    if fmt == "jpg":
        fmt = "jpeg"

    # Add safety limits for very large documents
    # Define maximum allowed dimensions
    MAX_DIMENSION = 15000  # Limit to 15,000 pixels in any dimension
    
    try:
        # Process each page individually to minimize memory usage
        for page_idx in pages_to_convert:
            # Only open the document for one page at a time
            with fitz.open(pdf_file) as doc:
               
                # Load the specific page
                page = doc.load_page(page_idx)
                
                # Determine sizing approach
                if args.max_width or args.max_height:
                    # Use max dimensions (preserves aspect ratio)
                    max_w = min(args.max_width if args.max_width else float("inf"), MAX_DIMENSION)
                    max_h = min(args.max_height if args.max_height else float("inf"), MAX_DIMENSION)

                    # Calculate zoom factors
                    zoom_x = (
                        max_w / page.rect.width if max_w < float("inf") else float("inf")
                    )
                    zoom_y = (
                        max_h / page.rect.height if max_h < float("inf") else float("inf")
                    )

                    # Use the smaller zoom to ensure the image fits within max dimensions
                    zoom = min(zoom_x, zoom_y)
                    if zoom == float("inf"):  # If neither max_width nor max_height was set
                        zoom = 1.0

                    matrix = fitz.Matrix(zoom, zoom)
                    
                    # Check estimated dimensions before creating pixmap
                    est_width = page.rect.width * zoom
                    est_height = page.rect.height * zoom
                    
                    if est_width > MAX_DIMENSION or est_height > MAX_DIMENSION:
                        # Recalculate zoom to fit within MAX_DIMENSION
                        zoom_x = MAX_DIMENSION / page.rect.width
                        zoom_y = MAX_DIMENSION / page.rect.height
                        zoom = min(zoom_x, zoom_y)
                        matrix = fitz.Matrix(zoom, zoom)
                    

                elif args.width and args.height:
                    # Use exact width and height
                    width = min(args.width, MAX_DIMENSION)
                    height = min(args.height, MAX_DIMENSION)
                    
                    if args.preserve_aspect_ratio:
                        # Calculate zoom factors
                        zoom_x = width / page.rect.width
                        zoom_y = height / page.rect.height

                        # Use the smaller zoom to preserve aspect ratio
                        zoom = min(zoom_x, zoom_y)
                        matrix = fitz.Matrix(zoom, zoom)

                    else:
                        # Use exact dimensions (may distort image)
                        # Calculate matrix for exact sizing
                        scale_x = width / page.rect.width
                        scale_y = height / page.rect.height
                        matrix = fitz.Matrix(scale_x, scale_y)

                    # Use DPI (but limit the resulting dimensions)
                    # First, check what dimensions would result at the requested DPI
                    dpi = min(args.dpi, 600)  # Cap DPI at 600 to prevent memory issues
                    zoom = dpi / 72  # 72 DPI is the base for PDF
                    est_width = page.rect.width * zoom
                    est_height = page.rect.height * zoom
                    
                    if est_width > MAX_DIMENSION or est_height > MAX_DIMENSION:
                        # Calculate safe DPI
                        zoom_x = MAX_DIMENSION / page.rect.width
                        zoom_y = MAX_DIMENSION / page.rect.height
                        zoom = min(zoom_x, zoom_y)
                        safe_dpi = zoom * 72
                        print(f"Warning: Page {page_idx+1} at {dpi} DPI would result in dimensions "
                              f"{est_width:.0f}x{est_height:.0f}, which exceeds the safety limit. "
                              f"Reducing to {safe_dpi:.0f} DPI.", file=sys.stderr)
                        dpi = safe_dpi
                    
                    matrix = fitz.Matrix(dpi/72, dpi/72)

                
                try:
                    # Get the pixel map
                    pix = page.get_pixmap(matrix=matrix)
                    
                    # Save the image
                    output_file = output_dir / f"{base_name}_page_{page_idx + 1}.{args.format}"

                    if fmt == "png":
                        pix.save(str(output_file), output=fmt)
                    elif fmt == "jpeg":
                        pix.save(str(output_file), output=fmt, quality=args.quality)
                    else:  # tiff
                        pix.save(str(output_file), output=fmt)
                    
                    # Explicitly delete the pixmap to free memory
                    del pix
                except Exception as e:
                    print(f"Error processing page {page_idx+1} of '{pdf_file}': {str(e)}", file=sys.stderr)
                    success = False
                    continue
            
        return success

    except Exception as e:
        print(f"Error processing '{pdf_file}': {str(e)}", file=sys.stderr)
        return False


def create_images(pdf_folder, output_dir):
    """
    Convert all PDFs in pdf_folder to images in output_dir using parallel threads.
    Images will be PNG, 544x704.
    """
    # Find all PDF files in the folder
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return

    # Prepare arguments for each PDF
    class Args:
        def __init__(self, pdf_file, output):
            self.pdf_file = pdf_file
            self.format = "png"
            self.dpi = 200
            self.width = 544
            self.height = 704
            self.max_width = None
            self.max_height = None
            self.quality = 90
            self.output = output
            self.pages = None
            self.preserve_aspect_ratio = True

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        for pdf_file in pdf_files:
            args = Args(pdf_file, output_dir)
            futures.append(executor.submit(process_pdf, args))

        # Use tqdm to track progress
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting PDFs to images"):
            pass

    print(f"Finished converting {len(pdf_files)} PDFs to images in {output_dir}")


def pdf2img(pdf_file:str,
            width:int,
            height:int,
            max_width:int,
            max_height:int,
            pages:str,
            preserve_aspect_ratio:bool=True,
            format:str= "png", 
            dpi:int=200,
            quality:int=90,
            output:str="."):
    # Start time
    start_time = time.time()

    # Create output directory if it doesn't exist
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process PDFs in parallel using ThreadPoolExecutor
    success_count = 0
    failure_count = 0
    
    try:
        process_pdf(args)
        return 0
    except Exception as exc:
        print(
            f"Generated an exception: {exc}", file=sys.stderr
        )
        return 1
    finally:
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds:")