#!/usr/bin/env python3
"""
DIAGNOSTIC TOOL - Find exactly what's wrong with your detection
Run this with a test image: python diagnose.py your_image.jpg
"""

import sys
import os
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import traceback

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_ok(text):
    print(f"{text}")

def print_error(text):
    print(f"{text}")

def print_warning(text):
    print(f"{text}")

def print_info(text):
    print(f"{text}")

# ============================================================================
# TEST 1: CHECK IF FILE EXISTS AND CAN BE OPENED
# ============================================================================
print_section("TEST 1: FILE VALIDATION")

if len(sys.argv) < 2:
    print_error("Usage: python diagnose.py your_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print_error(f"File not found: {image_path}")
    sys.exit(1)

print_ok(f"File exists: {image_path}")
print_info(f"File size: {os.path.getsize(image_path) / 1024:.2f} KB")

try:
    img = Image.open(image_path)
    print_ok(f"Image format: {img.format}")
    print_ok(f"Image size: {img.size}")
    print_ok(f"Image mode: {img.mode}")
except Exception as e:
    print_error(f"Cannot open image: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: CHECK DEPENDENCIES
# ============================================================================
print_section("TEST 2: DEPENDENCY CHECK")

dependencies = {
    'cv2': 'OpenCV (cv2)',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'PIL': 'Pillow',
}

missing = []
for module_name, readable_name in dependencies.items():
    try:
        __import__(module_name)
        print_ok(f"{readable_name} installed")
    except ImportError:
        print_error(f"{readable_name} NOT installed")
        missing.append(module_name)

if missing:
    print("\n" + "!"*70)
    print(f"INSTALL MISSING PACKAGES:")
    print(f"pip install " + " ".join(missing))
    print("!"*70)
    sys.exit(1)

# ============================================================================
# TEST 3: ELA DETECTION
# ============================================================================
print_section("TEST 3: ELA DETECTION")

try:
    import cv2
    
    # Load and prepare image
    original = Image.open(image_path).convert("RGB")
    print_ok(f"Original image loaded: {original.size}")
    
    # Save at quality 60 for comparison
    temp_path = "diagnostic_temp_resaved.jpg"
    original.save(temp_path, "JPEG", quality=60)
    print_ok(f"Resaved at quality 60: {temp_path}")
    
    resaved = Image.open(temp_path)
    print_ok(f"Resaved image loaded: {resaved.size}")
    
    # Calculate ELA
    ela_image = ImageChops.difference(original, resaved)
    ela_array = np.array(ela_image)
    
    print_ok(f"ELA array shape: {ela_array.shape}")
    print_ok(f"ELA min value: {ela_array.min()}")
    print_ok(f"ELA max value: {ela_array.max()}")
    print_ok(f"ELA mean value: {ela_array.mean():.2f}")
    
    # Check if ELA has meaningful data
    if ela_array.max() < 5:
        print_warning("ELA max value is very low (<5) - images are too similar!")
        print_info("This means quality=60 might not be low enough, or image is already highly compressed")
    
    # Normalize
    max_val = ela_array.max() if ela_array.max() > 0 else 1
    ela_scaled = (ela_array / max_val * 255).astype(np.uint8)
    
    # Convert for processing
    orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    gray_ela = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
    
    # Threshold
    threshold_value = 30
    _, mask = cv2.threshold(gray_ela, threshold_value, 255, cv2.THRESH_BINARY)
    
    print_ok(f"Threshold value: {threshold_value}")
    print_ok(f"Suspicious pixels: {np.sum(mask > 0)}")
    print_ok(f"Percentage of image: {(np.sum(mask > 0) / mask.size * 100):.2f}%")
    
    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print_ok(f"Total contours found: {len(contours)}")
    
    suspicious_area = 0
    significant_regions = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            suspicious_area += area
            significant_regions += 1
    
    print_ok(f"Significant regions (>300px): {significant_regions}")
    print_ok(f"Total suspicious area: {suspicious_area}")
    
    image_area = gray_ela.shape[0] * gray_ela.shape[1]
    anomaly_ratio = suspicious_area / image_area if image_area > 0 else 0
    
    print_ok(f"Image area: {image_area}")
    print_ok(f"Anomaly ratio: {anomaly_ratio:.6f}")
    
    # Calculate ELA score
    ela_score = round(anomaly_ratio * 100 * 3, 2)
    ela_score = min(ela_score, 100)
    
    print_ok(f"ELA Score (raw): {ela_score}")
    
    # Normalize
    ela_score_norm = min(max(ela_score * 3, 0), 100)
    print_ok(f"ELA Score (normalized ×3): {ela_score_norm}")
    
    if ela_score < 1:
        print_warning("ELA Score is very low - check if image is heavily compressed already")
    
    # Save diagnostic images
    Image.fromarray(ela_scaled).save("diagnostic_ela_result.jpg")
    print_ok("Saved: diagnostic_ela_result.jpg")
    
    os.remove(temp_path)
    
except Exception as e:
    print_error(f"ELA Detection failed: {e}")
    traceback.print_exc()
    ela_score_norm = 0

# ============================================================================
# TEST 4: NOISE DETECTION
# ============================================================================
print_section("TEST 4: NOISE DETECTION")

try:
    import cv2
    
    image = Image.open(image_path).convert('RGB')
    gray = image.convert("L")
    gray_array = np.array(gray).astype(np.float32)
    
    print_ok(f"Gray image shape: {gray_array.shape}")
    
    # Test Laplacian
    laplacian = cv2.Laplacian(gray_array.astype(np.uint8), cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    print_ok(f"Laplacian variance: {laplacian_var:.2f}")
    print_info(f"Expected range for real images: 50-500")
    print_info(f"Expected range for AI images: 300-800+")
    
    laplacian_score = min((laplacian_var / 500) * 100, 100)
    print_ok(f"Laplacian score: {laplacian_score:.2f}")
    
    # Test Gaussian blur difference
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_array = np.array(blurred).astype(np.float32)
    
    diff = np.abs(gray_array - blurred_array)
    diff_normalized = (diff / (diff.max() + 1)) * 255
    
    mean_noise = np.mean(diff_normalized)
    std_noise = np.std(diff_normalized)
    
    print_ok(f"Mean noise: {mean_noise:.2f}")
    print_ok(f"Std noise: {std_noise:.2f}")
    
    # Color channel analysis
    r, g, b = image.split()
    r_std = np.std(np.array(r).astype(np.float32))
    g_std = np.std(np.array(g).astype(np.float32))
    b_std = np.std(np.array(b).astype(np.float32))
    
    color_imbalance = abs(r_std - g_std) + abs(g_std - b_std) + abs(r_std - b_std)
    color_imbalance_score = min((color_imbalance / 50) * 100, 100)
    
    print_ok(f"Color imbalance: {color_imbalance:.2f}")
    print_ok(f"Color imbalance score: {color_imbalance_score:.2f}")
    
    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges).astype(np.float32)
    edge_density = np.mean(edge_array) / 255
    edge_density_score = edge_density * 100
    
    print_ok(f"Edge density: {edge_density:.4f}")
    print_ok(f"Edge density score: {edge_density_score:.2f}")
    
    # Uniformity
    uniformity_score = 1 / (std_noise + 1)
    uniformity_score = min(uniformity_score * 100, 100)
    
    print_ok(f"Uniformity score: {uniformity_score:.2f}")
    
    # Combined noise score
    noise_score = (
        (laplacian_score * 0.25) +
        ((mean_noise / 255) * 100 * 0.15) +
        ((std_noise / 255) * 100 * 0.2) +
        (color_imbalance_score * 0.15) +
        (edge_density_score * 0.1) +
        (0 * 0.15)  # Simplified, skipping patch analysis
    )
    
    noise_score = min(max(noise_score, 0), 100)
    print_ok(f"Noise Score (raw): {noise_score:.2f}")
    
    noise_score_norm = min(max(noise_score * 2, 0), 100)
    print_ok(f"Noise Score (normalized ×2): {noise_score_norm:.2f}")
    
    Image.fromarray(diff_normalized.astype(np.uint8)).save("diagnostic_noise_result.jpg")
    print_ok("Saved: diagnostic_noise_result.jpg")
    
except Exception as e:
    print_error(f"Noise Detection failed: {e}")
    traceback.print_exc()
    noise_score_norm = 0
    uniformity_score = 0

# ============================================================================
# TEST 5: FINAL SCORING
# ============================================================================
print_section("TEST 5: FINAL SCORING")

print_info(f"ELA Score (normalized): {ela_score_norm:.2f}")
print_info(f"Noise Score (normalized): {noise_score_norm:.2f}")
print_info(f"Uniformity Score: {uniformity_score:.2f}")

final_score = round(
    (ela_score_norm * 0.5) +
    (noise_score_norm * 0.4) +
    ((100 - uniformity_score) * 0.1),
    2
)

print_ok(f"Final Score: {final_score:.2f}")

print("\nCalculation breakdown:")
print(f"  ELA contribution (×0.5):       {ela_score_norm * 0.5:.2f}")
print(f"  Noise contribution (×0.4):     {noise_score_norm * 0.4:.2f}")
print(f"  Uniformity contribution (×0.1): {(100 - uniformity_score) * 0.1:.2f}")
print(f"  {'─'*40}")
print(f"  FINAL SCORE:                   {final_score:.2f}")

print(f"\nThreshold: >= 20 = FAKE, < 20 = REAL")

if final_score >= 20:
    print_ok(f"VERDICT: FAKE (Confidence: {final_score}%)")
else:
    print_ok(f"VERDICT: REAL (Confidence: {round(100 - final_score, 2)}%)")

# ============================================================================
# TEST 6: ANALYSIS & RECOMMENDATIONS
# ============================================================================
print_section("TEST 6: ANALYSIS & RECOMMENDATIONS")

if ela_score_norm < 5 and noise_score_norm < 20:
    print_warning("Both ELA and Noise scores are very low")
    print_info("This image is probably:")
    print_info("  1. A real, unedited photo")
    print_info("  2. Already highly compressed")
    print_info("  3. A screenshot or camera image")
    
elif ela_score_norm > 40 or noise_score_norm > 60:
    print_ok("Detection looks good - image shows signs of manipulation or AI generation")
    
else:
    print_warning("Scores are borderline - this image might be hard to classify")

print("\nIf detection is not working as expected:")
print("1. Test with a KNOWN fake image (DALL-E, Midjourney, etc)")
print("2. Test with a KNOWN real image (phone camera photo)")
print("3. Compare the scores between them")
print("4. Check that the difference is clear")

print_section("DIAGNOSTIC COMPLETE")
print(f"\nDiagnostic images saved:")
print(f"  - diagnostic_ela_result.jpg")
print(f"  - diagnostic_noise_result.jpg")
print(f"\nReview these images to understand what the detection is analyzing")
