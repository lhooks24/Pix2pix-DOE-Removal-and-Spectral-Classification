import cv2
import numpy as np
from pathlib import Path
import csv

def calculate_psnr(img1, img2):
    """Calculates PSNR (20log10(MAX/RMSE)) between two arrays (single channel or RGB)."""
    # np.mean calculates the MSE across all elements in the array
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf') # Handles perfectly identical images or completely empty channels
    # MAX is 255.0, RMSE is sqrt(mse)
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_colorized_channel(ch_data, color_name, save_path):
    """Saves a 2D single-channel image as a colorized BGR image for OpenCV."""
    zeros = np.zeros_like(ch_data)
    if color_name == 'R':
        colorized = cv2.merge([zeros, zeros, ch_data]) 
    elif color_name == 'G':
        colorized = cv2.merge([zeros, ch_data, zeros])
    elif color_name == 'B':
        colorized = cv2.merge([ch_data, zeros, zeros])
    cv2.imwrite(str(save_path), colorized)

# Point this to the root of your results directory where your 'images' folder lives
base_dir = Path("C:/Users/sovie/OneDrive - University of Utah/grad school/research/Super-Res/ML Training (Pix or it didn't happen)/v3") 
images_dir = base_dir / "images"

fake_b_files = sorted(images_dir.glob("*_fake_B.png"))

# Set up the HTML structure
html_content = [
    "<!DOCTYPE html>",
    "<html>",
    "<head><title>Spectral Channel Analysis</title>",
    "<style>",
    "body { font-family: sans-serif; background-color: #1e1e1e; color: #ddd; }",
    "table { border-collapse: collapse; width: 100%; text-align: center; }",
    "th, td { border: 1px solid #444; padding: 5px; }",
    "img { width: 256px; image-rendering: pixelated; }", 
    ".header-row { background-color: #333; }",
    ".divider { border-left: 3px solid #666; }",
    ".score { font-family: monospace; color: #ffeb3b; margin-top: 5px; display: block; }",
    "</style>",
    "</head>",
    "<body>",
    "<h2>Spectral Classification: Raw Channel Split & PSNR</h2>",
    "<table>",
    "<tr class='header-row'>",
    "<th>Input (Real A)</th>",
    "<th class='divider'>Fake B (Red)</th><th>Fake B (Green)</th><th>Fake B (Blue)</th>",
    "<th class='divider'>Real B (Red)</th><th>Real B (Green)</th><th>Real B (Blue)</th>",
    "</tr>"
]

# --- CSV DATA SETUP ---
csv_data = [["Image ID", "RED_PSNR", "Green_PSNR", "BLUE_PSNR", "RGB_PSNR"]]

# Trackers for dataset-wide statistics
stats_tracker = {'R': [], 'G': [], 'B': [], 'RGB': []}

for fake_b_path in fake_b_files:
    stem = fake_b_path.stem
    prefix = stem.replace("_fake_B", "")
    
    real_a_path = images_dir / f"{prefix}_real_A.png"
    real_b_path = images_dir / f"{prefix}_real_B.png"
    
    if not real_b_path.exists():
        continue
        
    # Load images
    fake_b_img = cv2.imread(str(fake_b_path))
    real_b_img = cv2.imread(str(real_b_path))
    
    # Split channels (OpenCV loads as B, G, R)
    fb_B, fb_G, fb_R = cv2.split(fake_b_img)
    rb_B, rb_G, rb_R = cv2.split(real_b_img)
    
    # --- CALCULATE RAW PSNR ---
    psnr_r = calculate_psnr(fb_R, rb_R)
    psnr_g = calculate_psnr(fb_G, rb_G)
    psnr_b = calculate_psnr(fb_B, rb_B)
    psnr_rgb = calculate_psnr(fake_b_img, real_b_img) # Calculate across the full 3D array
    
    # Add to trackers if the score is a valid number (not infinity)
    if psnr_r != float('inf'): stats_tracker['R'].append(psnr_r)
    if psnr_g != float('inf'): stats_tracker['G'].append(psnr_g)
    if psnr_b != float('inf'): stats_tracker['B'].append(psnr_b)
    if psnr_rgb != float('inf'): stats_tracker['RGB'].append(psnr_rgb)

    # Formatting for HTML display
    str_r = f"{psnr_r:.2f} dB" if psnr_r != float('inf') else "Empty/Perfect"
    str_g = f"{psnr_g:.2f} dB" if psnr_g != float('inf') else "Empty/Perfect"
    str_b = f"{psnr_b:.2f} dB" if psnr_b != float('inf') else "Empty/Perfect"
    
    # Formatting for CSV storage
    csv_data.append([
        prefix,
        round(psnr_r, 2) if psnr_r != float('inf') else "N/A",
        round(psnr_g, 2) if psnr_g != float('inf') else "N/A",
        round(psnr_b, 2) if psnr_b != float('inf') else "N/A",
        round(psnr_rgb, 2) if psnr_rgb != float('inf') else "N/A"
    ])
    
    # --- SAVE RAW CHANNELS FOR VISUALIZATION ---
    save_colorized_channel(fb_R, 'R', images_dir / f"{prefix}_fake_B_R.png")
    save_colorized_channel(fb_G, 'G', images_dir / f"{prefix}_fake_B_G.png")
    save_colorized_channel(fb_B, 'B', images_dir / f"{prefix}_fake_B_B.png")
    
    save_colorized_channel(rb_R, 'R', images_dir / f"{prefix}_real_B_R.png")
    save_colorized_channel(rb_G, 'G', images_dir / f"{prefix}_real_B_G.png")
    save_colorized_channel(rb_B, 'B', images_dir / f"{prefix}_real_B_B.png")
    
    # Append the row with PSNR scores under the Fake B images
    html_content.append("<tr>")
    html_content.append(f"<td><img src='images/{prefix}_real_A.png'><br>{prefix}</td>")
    html_content.append(f"<td class='divider'><img src='images/{prefix}_fake_B_R.png'><span class='score'>PSNR: {str_r}</span></td>")
    html_content.append(f"<td><img src='images/{prefix}_fake_B_G.png'><span class='score'>PSNR: {str_g}</span></td>")
    html_content.append(f"<td><img src='images/{prefix}_fake_B_B.png'><span class='score'>PSNR: {str_b}</span></td>")
    html_content.append(f"<td class='divider'><img src='images/{prefix}_real_B_R.png'></td>")
    html_content.append(f"<td><img src='images/{prefix}_real_B_G.png'></td>")
    html_content.append(f"<td><img src='images/{prefix}_real_B_B.png'></td>")
    html_content.append("</tr>")

html_content.append("</table></body></html>")

# Write the final HTML file
html_path = base_dir / "spectral_comparison.html"
with open(html_path, "w") as f:
    f.write("\n".join(html_content))


# --- CALCULATE AND APPEND DATASET SCORES TO CSV ---
def safe_stat(data_list, stat_func):
    """Safely calculates a statistic, returning 'N/A' if the list is empty."""
    if not data_list: return "N/A"
    return round(stat_func(data_list), 2)

# Add spacing rows before the summary
csv_data.append([])
csv_data.append(["Dataset Scores", "Value"])

# Calculate and append stats for each category
categories = [
    ("Red", stats_tracker['R']),
    ("Green", stats_tracker['G']),
    ("Blue", stats_tracker['B']),
    ("RGB", stats_tracker['RGB'])
]

for name, data in categories:
    csv_data.append([f"{name}_AVG", safe_stat(data, np.mean)])
    csv_data.append([f"{name}_Median", safe_stat(data, np.median)])
    csv_data.append([f"{name}_StdDev", safe_stat(data, np.std)])

# Write the CSV file
csv_path = base_dir / "spectral_psnr_scores.csv"
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"Done! {len(fake_b_files)} sets processed.")
print(f"Visualizations saved to: {html_path}")
print(f"Metrics saved to: {csv_path}")