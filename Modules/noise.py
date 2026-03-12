from PIL import Image, ImageFilter, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def detect_noise(image_path):
    """
    FIXED Noise detection - proper scaling without over-scoring
    
    Key fix:
    - Proper normalization of metrics
    - Laplacian scoring that matches real vs fake
    - No over-aggressive weighting
    """
    
    image = Image.open(image_path).convert('RGB')
    os.makedirs("static/uploads", exist_ok=True)

    print(f"[NOISE] Analyzing image: {image_path}")

    gray = image.convert("L")
    gray_array = np.array(gray).astype(np.float32)

    # ===============================
    # 1. LAPLACIAN VARIANCE
    # ===============================
    laplacian = cv2.Laplacian(gray_array.astype(np.uint8), cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    # FIXED: Proper threshold
    # Real images: 50-300
    # AI images: 300-800+
    # Don't over-score - just detect the difference
    if laplacian_var < 150:
        laplacian_score = 20  # Low variance = likely real
    elif laplacian_var < 300:
        laplacian_score = 40  # Medium variance = borderline
    else:
        laplacian_score = 70  # High variance = likely fake
    
    print(f"[NOISE] Laplacian variance: {laplacian_var:.2f} → Score: {laplacian_score:.2f}")

    # ===============================
    # 2. GAUSSIAN BLUR DIFFERENCE
    # ===============================
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_array = np.array(blurred).astype(np.float32)

    diff = np.abs(gray_array - blurred_array)
    diff_normalized = (diff / (diff.max() + 1)) * 255
    diff_normalized = diff_normalized.astype(np.uint8)

    noise_image = Image.fromarray(diff_normalized)
    noise_path = "static/uploads/noise_result.jpg"
    noise_image.save(noise_path)

    # ===============================
    # 3. HISTOGRAM
    # ===============================
    plt.figure(figsize=(8, 4))
    plt.hist(diff_normalized.flatten(), bins=256, alpha=0.7, color='blue')
    plt.title("Noise Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    hist_path = "static/uploads/noise_histogram.png"
    plt.savefig(hist_path, dpi=100, bbox_inches='tight')
    plt.close()

    # ===============================
    # 4. EDGE DETECTION
    # ===============================
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges).astype(np.float32)
    edge_path = "static/uploads/edge_map.jpg"
    edges.save(edge_path)

    # ===============================
    # 5. METRICS (FIXED)
    # ===============================
    
    mean_noise = np.mean(diff_normalized)
    std_noise = np.std(diff_normalized)
    
    # FIXED: Don't over-score mean/std
    # Real images typically have low mean noise
    mean_noise_score = min((mean_noise / 255) * 100, 30)  # Cap at 30
    std_noise_score = min((std_noise / 255) * 100, 30)    # Cap at 30
    
    print(f"[NOISE] Mean noise: {mean_noise:.2f}, Std: {std_noise:.2f}")
    print(f"[NOISE] Mean score: {mean_noise_score:.2f}, Std score: {std_noise_score:.2f}")

    # Color Channel Analysis (FIXED)
    r, g, b = image.split()
    r_array = np.array(r).astype(np.float32)
    g_array = np.array(g).astype(np.float32)
    b_array = np.array(b).astype(np.float32)
    
    r_std = np.std(r_array)
    g_std = np.std(g_array)
    b_std = np.std(b_array)

    # FIXED: Better color imbalance calculation
    color_imbalance = abs(r_std - g_std) + abs(g_std - b_std) + abs(r_std - b_std)
    # Real images: typically 5-20
    # AI images: typically 25-50
    color_imbalance_score = min((color_imbalance / 100) * 100, 50)  # Cap at 50
    
    print(f"[NOISE] Color imbalance: {color_imbalance:.2f} → Score: {color_imbalance_score:.2f}")

    # Edge Density (FIXED)
    edge_density = np.mean(edge_array) / 255
    edge_density_score = edge_density * 100 * 0.5  # Scale down by 50%
    print(f"[NOISE] Edge density: {edge_density:.4f} → Score: {edge_density_score:.2f}")

    # Patch Variance Inconsistency (FIXED)
    patch_size = 16
    h, w = gray_array.shape
    patch_variances = []
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = gray_array[i:i+patch_size, j:j+patch_size]
            patch_variances.append(np.var(patch))
    
    if patch_variances:
        patch_variance_std = np.std(patch_variances)
        patch_variance_mean = np.mean(patch_variances)
        patch_variance_cv = (patch_variance_std / (patch_variance_mean + 1)) * 100
        inconsistency_score = min((patch_variance_cv / 100) * 100, 40)  # Cap at 40
        print(f"[NOISE] Patch variance CV: {patch_variance_cv:.2f} → Score: {inconsistency_score:.2f}")
    else:
        inconsistency_score = 0

    # Uniformity Score (FIXED)
    uniformity_score = 1 / (std_noise + 1)
    uniformity_score = min(uniformity_score * 100, 100)
    print(f"[NOISE] Uniformity score: {uniformity_score:.2f}")

    # ===============================
    # 6. COMBINED NOISE SCORE (FIXED)
    # ===============================
    # CRITICAL FIX: Much simpler and balanced
    # Don't over-weight any single metric
    noise_score = (
        (laplacian_score * 0.4) +           # Laplacian: Main indicator (40%)
        (mean_noise_score * 0.1) +          # Mean noise: Supplementary (10%)
        (std_noise_score * 0.1) +           # Std noise: Supplementary (10%)
        (color_imbalance_score * 0.15) +    # Color: Indicator (15%)
        (edge_density_score * 0.1) +        # Edges: Supplementary (10%)
        (inconsistency_score * 0.15)        # Variance: Indicator (15%)
    )

    # CRITICAL: Cap the score properly
    noise_score = min(max(noise_score, 0), 100)
    noise_score = round(noise_score, 2)
    uniformity_score = round(uniformity_score, 2)

    print(f"[NOISE] Final noise score: {noise_score}")
    print(f"[NOISE] Uniformity score: {uniformity_score}")

    return noise_path, hist_path, edge_path, noise_score, uniformity_score