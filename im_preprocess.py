import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import tifffile
    import numpy as np
    import re
    import cv2
    import shutil
    from pathlib import Path
    from scipy.ndimage import gaussian_filter
    import random

    return Path, cv2, gaussian_filter, mo, np, random, re, shutil, tifffile


@app.cell
def _(mo):
    browser = mo.ui.file_browser(initial_path='C:/Users/sovie/code/pix2pix/')
    browser
    return (browser,)


@app.cell
def _(mo):
    # Add a dropdown to select the preprocessing task
    task_selector = mo.ui.dropdown(
        options=["Spectral Classification", "DOE Removal"],
        value="Spectral Classification",
        label="Select Pre-processing Task:"
    )
    task_selector
    return (task_selector,)


@app.cell
def _(cv2, gaussian_filter, np, re):
    def is_poorly_exposed(image):
        if image.dtype == np.uint8:
            max_val, bins = 255, 256
        elif image.dtype == np.uint16:
            max_val, bins = 65535, 65536
        else:
            max_val = image.max()
            bins = 256

        hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, max_val))
        total_pixels = image.size

        saturated_pixels = hist[-1] 
        saturation_ratio = saturated_pixels / total_pixels

        black_threshold_bin = int(bins * 0.05) 
        black_pixels = np.sum(hist[:black_threshold_bin])
        black_ratio = black_pixels / total_pixels

        maxval = np.max(image)
        #is_over = maxval > 0.85 * 65535
        # is_over = saturation_ratio > 0.05
        is_under = black_ratio > 0.95
        is_over = 0
        # is_under = 0

        return is_over, is_under

    def get_dataset_id(path_parts):
        for p in path_parts:
            match = re.search(r'_(\d+)$', p.lower())
            if match:
                return int(match.group(1))
        return 0

    def background_subtract(image):
        img_float = image.astype(np.float32)
        blur = gaussian_filter(img_float, 5)
        subtracted = cv2.subtract(img_float, blur)
        subtracted = np.clip(subtracted, 0, None)
        return image

    def get_global_min_max(dataset):
        global_min = float('inf')
        global_max = float('-inf')

        global_min = 0
        global_max = 65535
        # for image in dataset:
        #     over_g, under_g = is_poorly_exposed(image)
        #     if over_g or under_g:
        #         continue
        #     current_min = image.min()
        #     current_max = image.max()

        #     if current_min < global_min:
        #         global_min = current_min
        #     if current_max > global_max:
        #         global_max = current_max



        return global_min, global_max

    def normalize_image_global(image, global_min, global_max):
        img_float = image.astype(np.float32)
        dynamic_range = global_max - global_min

        if dynamic_range == 0:
            return np.zeros(image.shape, dtype=np.uint8)

        norm = (img_float - global_min) / dynamic_range
        norm = np.clip(norm, 0.0, 1.0)

        return (norm * 255).astype(np.uint8)

    def parse_path_for_task(path, task_name):
        import re
        stem = path.stem
        stem_lower = stem.lower()
        parts = [p.lower() for p in path.parts]

        if task_name == "Spectral Classification":
            sample_type = ""
            doe_status = ""
            dataset_id = ""
            category = None

            for p in parts:
                # 1. Identify the color category
                if p == "all" or p.startswith("all_"): category = "all"
                elif p == "red" or p.startswith("red_"): category = "red"
                elif p == "green" or p.startswith("green_"): category = "green"
                elif p == "blue" or p.startswith("blue_"): category = "blue"

                # 2. Extract the sample type to prevent bead/bio cross-over
                if "bio" in p: sample_type = "bio_"
                elif "beads" in p: sample_type = "beads_"

                # 3. Extract DOE status to prevent collisions if multiple folders are selected
                if "withdoe" in p: doe_status = "withdoe_"
                elif "nodoe" in p: doe_status = "nodoe_"

                # 4. Extract the trailing number from the folder name if it exists
                match = re.search(r'_(\d+)$', p)
                if match:
                    dataset_id = f"_{match.group(1)}"

            if category:
                # Build a unified key that drops the color but keeps all other metadata
                full_prefix = f"{sample_type}{doe_status}"
                return category, f"{full_prefix}{stem}{dataset_id}".strip("_-")

        elif task_name == "DOE Removal":
            color_prefix = ""
            sample_type = ""
            dataset_id = ""

            # 1. Identify color, sample type, AND dataset suffix (e.g. "_1")
            for p in parts:
                if "all" in p: color_prefix = "all_"
                elif "red" in p: color_prefix = "red_"
                elif "green" in p: color_prefix = "green_"
                elif "blue" in p: color_prefix = "blue_"

                # Extract the sample type to prevent bead/bio cross-over
                if "bio" in p: sample_type = "bio_"
                elif "beads" in p: sample_type = "beads_"

                # Extract the trailing number from the folder name if it exists
                if "withdoe" in p or "nodoe" in p:
                    match = re.search(r'_(\d+)$', p)
                    if match:
                        dataset_id = f"_{match.group(1)}"

            # Combine prefixes
            full_prefix = f"{color_prefix}{sample_type}"

            # 2. Check if the keyword is in the FILENAME itself
            if "withdoe" in stem_lower:
                base = re.sub(r'(?i)[_-]?withdoe[_-]?(?:\d+)?', '', stem)
                return "doe", f"{full_prefix}{base}{dataset_id}".strip("_-")

            elif "nodoe" in stem_lower:
                base = re.sub(r'(?i)[_-]?nodoe[_-]?(?:\d+)?', '', stem)
                return "clean", f"{full_prefix}{base}{dataset_id}".strip("_-")

            # 3. Check if the keyword is in the FOLDER name
            for p in parts:
                if "withdoe" in p or p == "doe" or p == "traina":
                    return "doe", f"{full_prefix}{stem}{dataset_id}".strip("_-")
                elif "nodoe" in p or p == "clean" or p == "trainb":
                    return "clean", f"{full_prefix}{stem}{dataset_id}".strip("_-")

        return None, None

    return (
        background_subtract,
        is_poorly_exposed,
        normalize_image_global,
        parse_path_for_task,
    )


@app.cell
def _(
    Path,
    background_subtract,
    browser,
    cv2,
    is_poorly_exposed,
    mo,
    normalize_image_global,
    np,
    parse_path_for_task,
    random,
    shutil,
    task_selector,
    tifffile,
):
    def generate_training_dataset(_=None):
        if not browser.value:
            print("Please select a folder or files in the browser first.")
            return

        task = task_selector.value
        base_dir = Path("./dataset_aligned")

        domain_A_all = {}  
        domain_B_red = {}
        domain_B_green = {}
        domain_B_blue = {}

        domain_A_doe = {}
        domain_B_clean = {}

        selected_items = list(browser.value)
        total_scanned = 0
        unmatched_samples = []

        # OS-Agnostic File Discovery (Fixes Linux Case-Sensitivity Trap)
        for item in selected_items:
            path = Path(item.path)
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                        total_scanned += 1
                        category, unique_key = parse_path_for_task(f, task)

                        if category == "all": domain_A_all[unique_key] = f
                        elif category == "red": domain_B_red[unique_key] = f
                        elif category == "green": domain_B_green[unique_key] = f
                        elif category == "blue": domain_B_blue[unique_key] = f
                        elif category == "doe": domain_A_doe[unique_key] = f
                        elif category == "clean": domain_B_clean[unique_key] = f
                        else:
                            if len(unmatched_samples) < 5: unmatched_samples.append(f.name)
            else:
                if path.is_file() and path.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                    total_scanned += 1
                    category, unique_key = parse_path_for_task(path, task)
                    if category == "all": domain_A_all[unique_key] = path
                    elif category == "red": domain_B_red[unique_key] = path
                    elif category == "green": domain_B_green[unique_key] = path
                    elif category == "blue": domain_B_blue[unique_key] = path
                    elif category == "doe": domain_A_doe[unique_key] = path
                    elif category == "clean": domain_B_clean[unique_key] = path
                    else:
                        if len(unmatched_samples) < 5: unmatched_samples.append(path.name)

        def read_image(fpath):
            if fpath.suffix.lower() in ['.tif', '.tiff']: return tifffile.imread(fpath)
            else: return cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)

        print(f"\n--- Diagnostic: File Parsing for {task} ---")
        print(f"Total files discovered: {total_scanned}")
        if unmatched_samples:
            print(f"Unmatched file examples: {unmatched_samples}")

        if task == "Spectral Classification":
            print(f"Found 'all' (Domain A) images:   {len(domain_A_all)}")
            print(f"Found 'red' (Domain B) images:   {len(domain_B_red)}")
            print(f"Found 'green' (Domain B) images: {len(domain_B_green)}")
            print(f"Found 'blue' (Domain B) images:  {len(domain_B_blue)}")
            if len(domain_A_all) == 0: return print("\n❌ ERROR: No Domain A files found.")
        else:
            print(f"Found 'doe' (Domain A) images:   {len(domain_A_doe)}")
            print(f"Found 'clean' (Domain B) images: {len(domain_B_clean)}")
            if len(domain_A_doe) == 0: return print("\n❌ ERROR: No Domain A files found.")

        # =====================================================================
        # PHASE 1: Find Global Min/Max across the dataset
        # =====================================================================
        print("\n--- Phase 1: Scanning dataset for Global Min/Max ---")
        global_min = float('inf')
        global_max = float('-inf')
        global_min = 0.0*65535
        global_max = 65535*1.0

        # def yield_all_valid_paths():
        #     if task == "Spectral Classification":
        #         for f in domain_A_all.values(): yield f
        #         for f in domain_B_red.values(): yield f
        #         for f in domain_B_green.values(): yield f
        #         for f in domain_B_blue.values(): yield f
        #     else:
        #         for f in domain_A_doe.values(): yield f
        #         for f in domain_B_clean.values(): yield f

        # for fpath in yield_all_valid_paths():
        #         try:
        #             img = read_image(fpath)

        #             # ✅ Check exposure and drop bad images BEFORE doing math
        #             over, under = is_poorly_exposed(img)
        #             if over or under:
        #                 continue

        #             bg_sub = background_subtract(img)
        #             c_min, c_max = bg_sub.min(), bg_sub.max()
        #             if c_min < global_min: global_min = c_min
        #             if c_max > global_max: global_max = c_max
        #         except Exception:
        #             continue

        # if global_min == float('inf'):
        #     return print("❌ ERROR: Could not read images to calculate min/max.")

        print(f"  -> Calculated Global Min: {global_min}")
        print(f"  -> Calculated Global Max: {global_max}")

        # =====================================================================
        # PHASE 2: Process Images Using Global Limits
        # =====================================================================
        pairs_master = [] 
        print(f"\n--- Phase 2: Processing images for {task}... ---")

        if task == "Spectral Classification":
            for unique_key, all_file in domain_A_all.items():
                try:
                    img_all = read_image(all_file)
                    over_a, under_a = is_poorly_exposed(img_all)
                    if over_a or under_a: 
                        continue

                    bg_sub_all = background_subtract(img_all)
                    processed_all = normalize_image_global(bg_sub_all, global_min, global_max)
                    if not np.any(processed_all): continue
                except Exception as e:
                    continue

                red_file = domain_B_red.get(unique_key)
                green_file = domain_B_green.get(unique_key)
                blue_file = domain_B_blue.get(unique_key)
                active_channels = []

                if red_file:
                    try:
                        img_r = read_image(red_file)
                        bg_sub_r = background_subtract(img_r)
                        processed_r = normalize_image_global(bg_sub_r, global_min, global_max)
                        if np.any(processed_r): active_channels.append(('r', processed_r, 2)) 
                    except Exception: pass

                if green_file:
                    try:
                        img_g = read_image(green_file)
                        bg_sub_g = background_subtract(img_g)
                        processed_g = normalize_image_global(bg_sub_g, global_min, global_max)
                        if np.any(processed_g): active_channels.append(('g', processed_g, 1)) 
                    except Exception: pass

                if blue_file:
                    try:
                        img_b = read_image(blue_file)
                        bg_sub_b = background_subtract(img_b)
                        processed_b = normalize_image_global(bg_sub_b, global_min, global_max)
                        if np.any(processed_b): active_channels.append(('b', processed_b, 0)) 
                    except Exception: pass

                if active_channels:
                    h = min([ch[1].shape[0] for ch in active_channels] + [processed_all.shape[0]])
                    w = min([ch[1].shape[1] for ch in active_channels] + [processed_all.shape[1]])

                    domain_a_final = processed_all[:h, :w]
                    domain_b_color = np.zeros((h, w, 3), dtype=np.uint8)
                    type_str = ""

                    for name, img, ch_idx in active_channels:
                        domain_b_color[:, :, ch_idx] = img[:h, :w]
                        type_str += name

                    if np.any(domain_b_color):
                        pairs_master.append({
                            'stem': unique_key, 
                            'input_A': domain_a_final, 
                            'target_B': domain_b_color, 
                            'type': type_str
                        })

        elif task == "DOE Removal":
            for unique_key, doe_file in domain_A_doe.items():
                clean_file = domain_B_clean.get(unique_key)
                if not clean_file: continue

                try:
                    # Process Domain A
                    img_doe = read_image(doe_file)
                    over_a, under_a = is_poorly_exposed(img_doe)
                    if over_a or under_a: 
                        continue
                    bg_sub_doe = background_subtract(img_doe)
                    processed_doe = normalize_image_global(bg_sub_doe, global_min, global_max)

                    # Process Domain B
                    img_clean = read_image(clean_file)
                    bg_sub_clean = background_subtract(img_clean)
                    processed_clean = normalize_image_global(bg_sub_clean, global_min, global_max)

                    # Crop to ensure matching dimensions
                    h = min(processed_doe.shape[0], processed_clean.shape[0])
                    w = min(processed_doe.shape[1], processed_clean.shape[1])

                    domain_a_final = processed_doe[:h, :w]
                    domain_b_final = processed_clean[:h, :w]

                    if np.any(domain_a_final) and np.any(domain_b_final):
                        pairs_master.append({
                            'stem': unique_key,
                            'input_A': domain_a_final,
                            'target_B': domain_b_final,
                            'type': "doe_removed"
                        })
                except Exception as e:
                    print(f"❌ Error processing pair {unique_key}: {e}")
                    continue

        if len(pairs_master) == 0:
            return print("\n❌ ERROR: Files were found, but no valid pairs were generated.")

        random.seed(42) 
        random.shuffle(pairs_master)

        total_pairs = len(pairs_master)
        train_idx = int(total_pairs * 0.76)
        test_idx = train_idx + int(total_pairs * 0.19)

        splits = {
            'train': pairs_master[:train_idx],
            'test': pairs_master[train_idx:test_idx],
            'val': pairs_master[test_idx:]
        }

        def save_aligned_pairs(split_data, split_name):
            print(f"Saving {len(split_data)} pairs to '{split_name}'...")

            if base_dir.exists() and split_name == 'train':
                shutil.rmtree(base_dir, ignore_errors=True)
            (base_dir / split_name).mkdir(parents=True, exist_ok=True)

            for i, pair in enumerate(split_data):
                img_a = pair['input_A']
                img_b = pair['target_B']

                # ONLY convert to RGB if the task requires it
                if task == "Spectral Classification":
                    if len(img_a.shape) == 2:
                        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
                    elif len(img_a.shape) == 3 and img_a.shape[2] == 1:
                        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)

                    if len(img_b.shape) == 2:
                        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
                    elif len(img_b.shape) == 3 and img_b.shape[2] == 1:
                        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)

                aligned_img = np.concatenate((img_a, img_b), axis=1)
            
                # Format the index as a 4-digit zero-padded number, starting at 1
                save_path = base_dir / split_name / f"custom_{i+1:04d}.png"
                cv2.imwrite(str(save_path), aligned_img)

        print("\n--- Starting File Write ---")
        save_aligned_pairs(splits['train'], "train")
        save_aligned_pairs(splits['test'], "test")
        save_aligned_pairs(splits['val'], "val")

        print(f"\n--- {task} Pre-processing Complete ---")
        print(f"Total Paired Images: {total_pairs}")
        print(f"Saved Aligned Dataset to: {base_dir.absolute()}")

    mo.ui.button(label='Run Dataset Generation', on_click=generate_training_dataset)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
