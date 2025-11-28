import os
import sys
import traceback
import numpy as np

from modules.io_utils import load_image, save_image, create_session_output
from modules.align import align_images
from modules.deltae import compute_delta_e
from modules.analysis import filter_noise_defects, analyze_defect
from modules.heatmap import generate_heatmap_in_memory
from threshold_config import get_config

def process_tshirt(golden, test, cfg):
    """
    Process images in-memory without saving to disk.
    
    Args:
        golden: numpy array of golden sample image
        test: numpy array of test sample image
        cfg: configuration dictionary
        
    Returns:
        dict containing results and processed images
    """
    # Align images
    aligned = align_images(
        template=golden,
        image=test,
        orb_max_features=cfg["ORB_MAX_FEATURES"],
        orb_keep_percent=cfg["ORB_KEEP_PERCENT"]
    )

    # Compute Delta-E
    delta_e = compute_delta_e(golden, aligned)
    delta_e_uint8 = delta_e.astype("uint8")
    delta_e_ptp = np.ptp(delta_e)
    delta_e_normalized = (255 * (delta_e - delta_e.min()) / (delta_e_ptp if delta_e_ptp else 1)).astype("uint8")

    # Defect analysis
    thresholds = {
        "mean_diff": cfg["MEAN_DIFF"],
        "max_diff": cfg["MAX_DIFF"],
        "area_percent": cfg["AREA_PERCENT"],
        "delta_e_pixel_threshold": cfg["DELTA_E_PIXEL_THRESHOLD"]
    }

    is_defect, mean_diff, max_diff, area_percent = analyze_defect(delta_e, thresholds)

    # Create defect mask
    defect_mask = (delta_e > cfg["DELTA_E_PIXEL_THRESHOLD"]).astype("uint8") * 255

    # Filter noise
    filtered_mask = filter_noise_defects(
        defect_mask,
        min_size=cfg["MIN_DEFECT_SIZE"],
        min_circularity=cfg["MIN_CIRCULARITY"],
        morph_open_kernel_size=cfg["MORPH_OPEN_KERNEL_SIZE"],
        morph_open_iterations=cfg["MORPH_OPEN_ITERATIONS"],
        morph_close_kernel_size=cfg["MORPH_CLOSE_KERNEL_SIZE"],
        morph_close_iterations=cfg["MORPH_CLOSE_ITERATIONS"]
    )

    # Create overlay
    overlay = aligned.copy()
    overlay[filtered_mask > 0] = [0, 0, 255]

    # Calculate filtered percentage
    filtered_pixels = int((filtered_mask > 0).sum())
    total_pixels = int(defect_mask.size)
    filtered_percent = (filtered_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    # Generate heatmap in-memory
    heatmap = generate_heatmap_in_memory(delta_e_normalized, golden)

    return {
        "is_defect": is_defect,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "area_percent": area_percent,
        "filtered_percent": filtered_percent,
        "aligned": aligned,
        "delta_e_map": delta_e_uint8,
        "delta_e_normalized": delta_e_normalized,
        "defect_mask_unfiltered": defect_mask,
        "defect_mask_filtered": filtered_mask,
        "overlay": overlay,
        "heatmap": heatmap
    }

def process_tshirt_disk(golden_path, test_path, cfg, output_base="output"):
    session_dir = create_session_output(output_base)
    try:
        golden = load_image(golden_path)
        test = load_image(test_path)

        save_image(os.path.join(session_dir, "01_test_image.jpg"), test)
        save_image(os.path.join(session_dir, "02_golden_sample.jpg"), golden)

        
        aligned = align_images(
            template=golden,
            image=test,
            orb_max_features=cfg["ORB_MAX_FEATURES"],
            orb_keep_percent=cfg["ORB_KEEP_PERCENT"]
        )

        save_image(os.path.join(session_dir, "03_aligned_test.jpg"), aligned)

        # Delta-E
        delta_e = compute_delta_e(golden, aligned)
        save_image(os.path.join(session_dir, "04_delta_e_map.jpg"), delta_e.astype("uint8"))
        delta_e_normalized = (255 * (delta_e - delta_e.min()) / (delta_e.ptp() if delta_e.ptp() else 1)).astype("uint8")
        save_image(os.path.join(session_dir, "04a_delta_e_normalized_map.jpg"), delta_e_normalized)

        # Defect analysis
        thresholds = {
            "mean_diff": cfg["MEAN_DIFF"],
            "max_diff": cfg["MAX_DIFF"],
            "area_percent": cfg["AREA_PERCENT"],
            "delta_e_pixel_threshold": cfg["DELTA_E_PIXEL_THRESHOLD"]
        }

        is_defect, mean_diff, max_diff, area_percent = analyze_defect(delta_e, thresholds)

        defect_mask = (delta_e > cfg["DELTA_E_PIXEL_THRESHOLD"]).astype("uint8") * 255
        save_image(os.path.join(session_dir, "05_defects_unfiltered.jpg"), defect_mask)

        filtered_mask = filter_noise_defects(
            defect_mask,
            min_size=cfg["MIN_DEFECT_SIZE"],
            min_circularity=cfg["MIN_CIRCULARITY"],
            morph_open_kernel_size=cfg["MORPH_OPEN_KERNEL_SIZE"],
            morph_open_iterations=cfg["MORPH_OPEN_ITERATIONS"],
            morph_close_kernel_size=cfg["MORPH_CLOSE_KERNEL_SIZE"],
            morph_close_iterations=cfg["MORPH_CLOSE_ITERATIONS"]
        )
        save_image(os.path.join(session_dir, "06_defects_filtered.jpg"), filtered_mask)

        # Overlay filtered defects on aligned image
        overlay = aligned.copy()
        overlay[filtered_mask > 0] = [0, 0, 255]#bright red for defects
        save_image(os.path.join(session_dir, "07_defect_overlay.jpg"), overlay)

        # Recompute filtered percent
        filtered_pixels = int((filtered_mask > 0).sum())
        total_pixels = int(defect_mask.size)
        filtered_percent = (filtered_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    
        heatmap_path = os.path.join(session_dir, "08_final_heatmap.jpg")
        generate_heatmap_in_memory(delta_e_normalized, golden)

        # Print summary (no GUI)
        print("===== DEFECT DETECTION SUMMARY =====")
        print(f"Is defect (pre-filter): {is_defect}")
        print(f"Mean ΔE: {mean_diff:.2f}, Max ΔE: {max_diff:.2f}, Area %: {area_percent:.2f}")
        print(f"Defect area after filtering: {filtered_percent:.2f}%")
        print(f"All outputs saved to: {session_dir}")
        print("===================================")

        return {
            "session_dir": session_dir,
            "is_defect": is_defect,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "area_percent": area_percent,
            "filtered_percent": filtered_percent
        }

    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":
    cfg = get_config()

    # Use paths from threshold_config.py by default; allow override via CLI args:
    golden_path = cfg.get("Golden_sample") or ""
    test_path = cfg.get("Test_sample") or ""

    if len(sys.argv) >= 3:
        golden_path = sys.argv[1]
        test_path = sys.argv[2]
    elif not golden_path or not test_path:
        print("Usage: python main.py <golden_path> <test_path>")
        print("Or set Golden_sample/Test_sample in threshold_config.py")
        sys.exit(1)

    if not os.path.exists(golden_path) or not os.path.exists(test_path):
        print("ERROR: golden or test path not found.")
        print(f"Golden: {golden_path}")
        print(f"Test:   {test_path}")
        sys.exit(1)

    # Run pipeline
    result = process_tshirt_disk(golden_path, test_path, cfg, output_base="output")
