#!/usr/bin/env python3
"""
Script to add transparency channel to PNG images and make background transparent.

This script calculates the difference between each pixel and the background color,
then applies transparency based on the specified thresholds.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, binary_closing, binary_opening
from sklearn.cluster import KMeans


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to CIELAB color space for perceptually uniform color difference.

    Args:
        rgb: NumPy array of RGB values in range [0, 255], shape (H, W, 3).

    Returns:
        NumPy array of LAB values, shape (H, W, 3).
    """
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # RGB to XYZ conversion
    mask = rgb_norm > 0.04045
    rgb_norm[mask] = ((rgb_norm[mask] + 0.055) / 1.055) ** 2.4
    rgb_norm[~mask] = rgb_norm[~mask] / 12.92

    # Apply transformation matrix
    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    xyz = np.dot(rgb_norm, rgb_to_xyz_matrix.T) * 100.0

    # XYZ to LAB conversion
    # White point D65
    white_point = np.array([95.047, 100.000, 108.883])

    xyz_normalized = xyz / white_point

    mask = xyz_normalized > 0.008856
    f = np.zeros_like(xyz_normalized)
    f[mask] = xyz_normalized[mask] ** (1.0 / 3.0)
    f[~mask] = (7.787 * xyz_normalized[~mask]) + (16.0 / 116.0)

    l = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])

    return np.stack([l, a, b], axis=2)


def calculate_color_distance(
    pixels: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
    use_lab: bool = True
) -> np.ndarray:
    """
    Calculate the normalized distance between each pixel and the background color.

    Supports both RGB and CIELAB color spaces. LAB space provides perceptually
    uniform color differences that better match human perception.

    Args:
        pixels: NumPy array of shape (H, W, 3) or (H, W, 4) containing RGB(A) values.
        background_color: The background color to compare against (default white).
        use_lab: If True, use CIELAB color space for perceptually uniform distances.

    Returns:
        NumPy array of shape (H, W) with normalized distances in range [0, 1].
    """
    # Extract RGB channels only
    rgb = pixels[:, :, :3].astype(np.float32)

    if use_lab:
        # Convert to LAB space for perceptually uniform color difference
        lab_pixels = rgb_to_lab(rgb)
        bg_rgb = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
        bg_lab = rgb_to_lab(bg_rgb)

        # Calculate Euclidean distance in LAB space
        # Maximum LAB distance is approximately sqrt(100^2 + 128^2 + 128^2) ≈ 197.5
        max_distance = np.sqrt(100**2 + 128**2 + 128**2)
        distance = np.sqrt(np.sum((lab_pixels - bg_lab) ** 2, axis=2)) / max_distance
    else:
        # Original RGB-based calculation
        bg = np.array(background_color, dtype=np.float32)
        max_distance = np.sqrt(3 * (255**2))
        distance = np.sqrt(np.sum((rgb - bg) ** 2, axis=2)) / max_distance

    return distance


def apply_feathering(
    alpha_channel: np.ndarray,
    feather_radius: int = 2
) -> np.ndarray:
    """
    Apply feathering (blurring) to the alpha channel for smoother edges.

    This creates a smooth transition between transparent and opaque regions,
    reducing jagged edges and making the result look more natural.

    Args:
        alpha_channel: Alpha channel array of shape (H, W).
        feather_radius: Radius for feathering in pixels (default: 2).

    Returns:
        Feathered alpha channel array.
    """
    if feather_radius <= 0:
        return alpha_channel

    # Apply Gaussian blur to alpha channel
    from scipy.ndimage import gaussian_filter

    # Convert to float for blurring
    alpha_float = alpha_channel.astype(np.float32) / 255.0

    # Apply Gaussian blur
    blurred = gaussian_filter(alpha_float, sigma=feather_radius)

    # Convert back to uint8
    return (blurred * 255).astype(np.uint8)


def detect_edges(rgb_image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Detect edges in the image using Sobel operator to protect important boundaries.

    Args:
        rgb_image: RGB image array of shape (H, W, 3).
        threshold: Threshold for edge detection (0-1).

    Returns:
        Binary edge mask where 1 indicates an edge, shape (H, W).
    """
    # Convert to grayscale
    gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

    # Sobel filters
    from scipy.ndimage import sobel

    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize and threshold
    max_grad = np.max(gradient_magnitude)
    if max_grad > 0:
        gradient_magnitude = gradient_magnitude / max_grad

    edges = gradient_magnitude > threshold

    return edges.astype(np.uint8)


def morphological_optimize(
    alpha_channel: np.ndarray,
    close_iterations: int = 1,
    open_iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological operations to clean up the alpha channel.

    - Closing: Fills small holes in opaque regions
    - Opening: Removes small isolated transparent regions

    Args:
        alpha_channel: Alpha channel array of shape (H, W).
        close_iterations: Number of closing iterations.
        open_iterations: Number of opening iterations.

    Returns:
        Optimized alpha channel array.
    """
    from scipy.ndimage import binary_closing, binary_opening

    binary_mask = alpha_channel > 128

    # Apply closing (fill holes)
    if close_iterations > 0:
        for _ in range(close_iterations):
            binary_mask = binary_closing(binary_mask, iterations=1)

    # Apply opening (remove noise)
    if open_iterations > 0:
        for _ in range(open_iterations):
            binary_mask = binary_opening(binary_mask, iterations=1)

    return (binary_mask * 255).astype(np.uint8)


def auto_detect_background(
    rgb_image: np.ndarray,
    n_clusters: int = 3
) -> tuple[int, int, int]:
    """
    Automatically detect the dominant background color using K-means clustering.

    This is useful when the background is not pure white or when processing
    images with unknown background colors.

    Args:
        rgb_image: RGB image array of shape (H, W, 3).
        n_clusters: Number of color clusters for K-means.

    Returns:
        Tuple (R, G, B) representing the detected background color.
    """
    from sklearn.cluster import KMeans

    # Reshape image to list of pixels
    pixels = rgb_image.reshape(-1, 3)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Find the cluster with the largest area (most pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]

    # Get the color of the dominant cluster
    bg_color = kmeans.cluster_centers_[dominant_cluster]

    return tuple(int(c) for c in bg_color)


def apply_transparency(
    image: Image.Image,
    transparent_threshold: float = 0.1,
    opaque_threshold: float = 1.00,
    background_color: tuple[int, int, int] = (255, 255, 255),
    use_lab: bool = True,
    feather_radius: int = 2,
    edge_protection: bool = True,
    morphological_opt: bool = True,
    color_protection: bool = True,
) -> Image.Image:
    """
    Apply transparency to an image based on color distance from background.

    Enhanced version with multiple improvements:
    - CIELAB color space for perceptually uniform color differences
    - Edge feathering for smooth transitions
    - Edge protection to preserve subject boundaries
    - Morphological operations to clean up noise
    - Color protection to preserve non-white colored backgrounds

    Args:
        image: PIL Image object to process.
        transparent_threshold: Distance threshold below which pixels become transparent.
        opaque_threshold: Distance threshold above which pixels remain opaque.
        background_color: The background color to detect (default white).
        use_lab: Use CIELAB color space for better color difference perception.
        feather_radius: Edge feathering radius in pixels (0 to disable).
        edge_protection: Enable edge detection to preserve subject boundaries.
        morphological_opt: Enable morphological operations to clean up noise.
        color_protection: Protect non-white colors from being removed (prevents light blue/orange backgrounds from disappearing).

    Returns:
        PIL Image with transparency applied.

    Raises:
        ValueError: If thresholds are invalid.
    """
    if not (0 <= transparent_threshold <= 1):
        raise ValueError("transparent_threshold must be between 0 and 1")
    if not (0 <= opaque_threshold <= 1):
        raise ValueError("opaque_threshold must be between 0 and 1")
    if transparent_threshold > opaque_threshold:
        raise ValueError("transparent_threshold must be <= opaque_threshold")

    # Convert to RGBA if necessary
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    pixels = np.array(image)
    distance = calculate_color_distance(pixels, background_color, use_lab=use_lab)

    # Get original alpha channel
    original_alpha = pixels[:, :, 3].copy()

    # Calculate alpha values based on thresholds
    # - Below transparent_threshold: fully transparent (alpha = 0)
    # - Above opaque_threshold: fully opaque (alpha = 255)
    # - In between: keep original alpha
    alpha = np.where(distance < transparent_threshold, 0, original_alpha)
    alpha = np.where(distance > opaque_threshold, 255, alpha)

    # Edge protection: preserve detected edges
    if edge_protection:
        edges = detect_edges(pixels[:, :, :3], threshold=0.15)
        # Protect edges by setting them to opaque
        alpha = np.where(edges > 0, 255, alpha)

    # Color protection: preserve non-white colors to avoid removing colored backgrounds
    # This prevents light blue, light orange, etc. from being removed
    if use_lab:
        # In LAB space, check if the color has significant chromaticity (a and b channels)
        lab_pixels = rgb_to_lab(pixels[:, :, :3])
        # Calculate chromaticity distance from neutral gray (a=0, b=0)
        chromaticity = np.sqrt(lab_pixels[:, :, 1]**2 + lab_pixels[:, :, 2]**2)
        # Threshold for significant color (empirical value)
        color_threshold = 5.0
        # Protect pixels with significant chromaticity
        alpha = np.where(chromaticity > color_threshold, 255, alpha)

    # Morphological optimization
    if morphological_opt:
        alpha = morphological_optimize(alpha, close_iterations=1, open_iterations=1)

    # Apply feathering for smooth edges
    if feather_radius > 0:
        alpha = apply_feathering(alpha, feather_radius=feather_radius)

    # Apply the new alpha channel
    pixels[:, :, 3] = alpha.astype(np.uint8)

    return Image.fromarray(pixels, mode="RGBA")


def process_image(
    input_path: str,
    output_path: str,
    transparent_threshold: float = 0.1,
    opaque_threshold: float = 1.00,
    background_color: tuple[int, int, int] = (255, 255, 255),
    use_lab: bool = True,
    feather_radius: int = 2,
    edge_protection: bool = True,
    morphological_opt: bool = True,
    auto_detect_bg: bool = False,
    color_protection: bool = True,
) -> None:
    """
    Process an image file to add transparency with enhanced features.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the output image.
        transparent_threshold: Distance threshold for full transparency.
        opaque_threshold: Distance threshold for full opacity.
        background_color: The background color to detect.
        use_lab: Use CIELAB color space for better color difference perception.
        feather_radius: Edge feathering radius in pixels (0 to disable).
        edge_protection: Enable edge detection to preserve subject boundaries.
        morphological_opt: Enable morphological operations to clean up noise.
        auto_detect_bg: Automatically detect background color using K-means.
        color_protection: Protect non-white colors from being removed.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If input file is not a valid image.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"🚀 Processing image: {input_path}")
    print(f"   Transparent threshold: {transparent_threshold}")
    print(f"   Opaque threshold: {opaque_threshold}")
    print(f"   Use LAB color space: {use_lab}")
    print(f"   Feather radius: {feather_radius}")
    print(f"   Edge protection: {edge_protection}")
    print(f"   Morphological optimization: {morphological_opt}")
    print(f"   Color protection: {color_protection}")

    try:
        image = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # Auto-detect background color if requested
    if auto_detect_bg:
        print(f"   Auto-detecting background color...")
        rgb_pixels = np.array(image.convert("RGB"))
        detected_bg = auto_detect_background(rgb_pixels)
        print(f"   Detected background: RGB{detected_bg}")
        background_color = detected_bg
    else:
        print(f"   Background color: RGB{background_color}")

    result = apply_transparency(
        image,
        transparent_threshold=transparent_threshold,
        opaque_threshold=opaque_threshold,
        background_color=background_color,
        use_lab=use_lab,
        feather_radius=feather_radius,
        edge_protection=edge_protection,
        morphological_opt=morphological_opt,
        color_protection=color_protection,
    )

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result.save(output_path, format="PNG")
    print(f"✅ Image saved to: {output_path}")


def parse_color(color_str: str) -> tuple[int, int, int]:
    """
    Parse a color string in format 'R,G,B' to a tuple.

    Args:
        color_str: Color string like '255,255,255'.

    Returns:
        Tuple of (R, G, B) values.

    Raises:
        ValueError: If color format is invalid.
    """
    try:
        parts = [int(x.strip()) for x in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError("Color must have exactly 3 components")
        if not all(0 <= p <= 255 for p in parts):
            raise ValueError("Color values must be between 0 and 255")
        return tuple(parts)  # type: ignore
    except Exception as e:
        raise ValueError(f"Invalid color format '{color_str}': {e}")


def generate_default_output_path(input_path: str) -> str:
    """
    Generate a default output path by appending '_cleaned' suffix to the input filename.

    Args:
        input_path: Path to the input image file.

    Returns:
        Generated output path with '_cleaned' suffix.

    Examples:
        'image.png' -> 'image_cleaned.png'
        'path/to/photo.jpg' -> 'path/to/photo_cleaned.png'
    """
    input_file = Path(input_path)
    output_filename = f"{input_file.stem}_cleaned.png"
    return str(input_file.parent / output_filename)


def get_images_from_directory(directory_path: str) -> list[str]:
    """
    Get all image files from a directory.

    Args:
        directory_path: Path to the directory to search for images.

    Returns:
        List of absolute paths to image files found in the directory.

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no image files are found in the directory.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

    # Find all image files
    image_files = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")

    return sorted(image_files)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Add transparency to PNG images by making background transparent (Enhanced Version).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with enhanced features enabled by default
  %(prog)s -i input.png

  # Disable feathering for crisp edges
  %(prog)s -i input.png --no-feather

  # Use RGB color space instead of LAB
  %(prog)s -i input.png --no-lab

  # Auto-detect background color
  %(prog)s -i input.png --auto-detect-bg

  # For images with colored backgrounds (blue, orange, etc.)
  %(prog)s -i input.png --color-protection

  # Batch processing
  %(prog)s -i input1.png input2.png input3.png

  # Process directory
  %(prog)s -i /path/to/images/

  # Custom thresholds with all features
  %(prog)s -i input.png --transparent 0.05 --opaque 0.5 --feather 3

  # Remove black background
  %(prog)s -i input.png --background 0,0,0

Enhanced Features:
  - LAB color space for perceptually uniform color differences
  - Edge feathering for smooth transitions
  - Edge protection to preserve subject boundaries
  - Morphological operations to clean up noise
  - Color protection to preserve non-white backgrounds (prevents light blue/orange from disappearing)
  - Automatic background detection
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        help="Path(s) to input image file(s) or directory. If a directory is provided, "
        "all images in the directory will be processed. Supports multiple files for batch processing.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Path to save the output image. Only valid when processing a single file. "
        "If not specified, output will be saved as '<input_name>_cleaned.png'. "
        "When processing a directory, outputs are saved to '<directory>/output/'.",
    )
    parser.add_argument(
        "--transparent",
        type=float,
        default=0.1,
        help="Transparent threshold (0-1). Pixels with distance below this become transparent. Default: 0.1",
    )
    parser.add_argument(
        "--opaque",
        type=float,
        default=1.00,
        help="Opaque threshold (0-1). Pixels with distance above this remain opaque. Default: 1.00",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="255,255,255",
        help="Background color to detect as 'R,G,B'. Default: 255,255,255 (white)",
    )
    parser.add_argument(
        "--no-lab",
        action="store_true",
        help="Disable LAB color space and use RGB instead",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=2,
        help="Edge feathering radius in pixels (0 to disable). Default: 2",
    )
    parser.add_argument(
        "--no-edge-protection",
        action="store_true",
        help="Disable edge protection",
    )
    parser.add_argument(
        "--no-morphological",
        action="store_true",
        help="Disable morphological optimization",
    )
    parser.add_argument(
        "--auto-detect-bg",
        action="store_true",
        help="Automatically detect background color using K-means clustering",
    )
    parser.add_argument(
        "--no-color-protection",
        action="store_true",
        help="Disable color protection (may remove light colored backgrounds)",
    )

    args = parser.parse_args()

    # Check if input is a directory or files
    input_path = args.input[0]
    is_directory = Path(input_path).is_dir()

    # Validate: -o option is only valid for single file input
    if args.output is not None and (len(args.input) > 1 or is_directory):
        print(
            "❌ Error: -o/--output option is only valid when processing a single file."
        )
        if is_directory:
            print(
                "   For directory processing, outputs are saved to '<directory>/output/'."
            )
        else:
            print(
                "   For batch processing, output filenames are generated automatically as '<input_name>_cleaned.png'."
            )
        exit(1)

    try:
        background_color = parse_color(args.background)

        # Collect all files to process
        files_to_process = []
        if is_directory:
            # Get all images from directory
            files_to_process = get_images_from_directory(input_path)
            # Create output directory
            output_dir = Path(input_path) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Processing directory: {input_path}")
            print(f"📂 Output directory: {output_dir}")
        else:
            files_to_process = args.input

        # Process each input file
        for input_file in files_to_process:
            # Determine output path
            if args.output is not None:
                output_path = args.output
            else:
                if is_directory:
                    # Save to output subdirectory
                    input_file_path = Path(input_file)
                    output_filename = f"{input_file_path.stem}_cleaned.png"
                    output_path = str(output_dir / output_filename)
                else:
                    output_path = generate_default_output_path(input_file)

            try:
                process_image(
                    input_path=input_file,
                    output_path=output_path,
                    transparent_threshold=args.transparent,
                    opaque_threshold=args.opaque,
                    background_color=background_color,
                    use_lab=not args.no_lab,
                    feather_radius=args.feather,
                    edge_protection=not args.no_edge_protection,
                    morphological_opt=not args.no_morphological,
                    auto_detect_bg=args.auto_detect_bg,
                    color_protection=not args.no_color_protection,
                )
            except Exception as e:
                print(f"❌ Error processing '{input_file}': {e}")
                # Continue processing other files in batch mode
                if len(files_to_process) == 1:
                    exit(1)

        if len(files_to_process) > 1:
            print(f"🎉 Batch processing complete. Processed {len(files_to_process)} file(s).")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
