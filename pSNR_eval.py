import cv2
import numpy as np
from pathlib import Path
import csv

def calculate_psnr(img1, img2):
    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf') # Perfect match
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Set the results directory using a raw string to handle Windows backslashes safely
target_path = r"C:\Users\sovie\OneDrive - University of Utah\grad school\research\Super-Res\ML Training (Pix or it didn't happen)\v3\images"
results_dir = Path(target_path)

# Prepare the CSV file path
csv_file_path = results_dir / "psnr_evaluation_scores.csv"

psnr_scores = []
csv_rows = [] # List to hold data for the CSV

print("Calculating PSNR scores...")

# Find all the generated images
for fake_path in results_dir.glob("*_fake_B.png"):
    # Reconstruct the path to the matching ground truth image
    real_path = fake_path.parent / fake_path.name.replace("_fake_B.png", "_real_B.png")
    
    if real_path.exists():
        fake_img = cv2.imread(str(fake_path))
        real_img = cv2.imread(str(real_path))
        
        score = calculate_psnr(fake_img, real_img)
        psnr_scores.append(score)
        
        # Clean up the name for the CSV (e.g., 'image_001' instead of 'image_001_fake_B.png')
        base_name = fake_path.name.replace("_fake_B.png", "")
        csv_rows.append([base_name, round(score, 4)])

if psnr_scores:
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    
    # Write the data to a CSV file
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write headers
            writer.writerow(['Image_Name', 'PSNR_dB'])
            # Write individual scores
            writer.writerows(csv_rows)
            # Add a blank row and then the average at the bottom
            writer.writerow([])
            writer.writerow(['AVERAGE_PSNR', round(avg_psnr, 4)])
            
    except Exception as e:
        print(f"Error writing CSV file: {e}")

    print(f"--- Evaluation Complete ---")
    print(f"Images Evaluated: {len(psnr_scores)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Individual scores saved to: {csv_file_path}")
else:
    print("No matching real and fake images found. Check your paths!")