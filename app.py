import os
from flask import Flask, render_template, request
from PIL import Image

from Modules.ela import detect_ela
from Modules.noise import detect_noise

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        try:

            # =====================================
            # CHECK FILE
            # =====================================
            if "image" not in request.files:
                return "No file part"

            file = request.files["image"]

            if file.filename == "":
                return "No file selected"

            print("\n" + "="*70)
            print(f"PROCESSING: {file.filename}")
            print("="*70)

            # =====================================
            # SAVE TEMP FILE
            # =====================================
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_input")
            file.save(temp_path)

            # =====================================
            # STANDARDIZE IMAGE FORMAT
            # =====================================
            img = Image.open(temp_path).convert("RGB")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)

            upload_path = os.path.join(UPLOAD_FOLDER, "standardized.jpg")
            img.save(upload_path, "JPEG", quality=92)
            
            print(f"✓ Image standardized: {img.size}")

            # =====================================
            # RUN FORENSIC ANALYSIS
            # =====================================
            print("\n📊 Running analysis...")
            print("-"*70)
            
            ela_path, heatmap_path, boxed_path, ela_score = detect_ela(upload_path)
            print(f"✓ ELA Detection: {ela_score}")
            
            noise_path, hist_path, edge_path, noise_score, uniformity_score = detect_noise(upload_path)
            print(f"✓ Noise Detection: {noise_score}")
            print(f"✓ Uniformity Score: {uniformity_score}")

            # =====================================
            # NORMALIZE SCORES (0 - 100)
            # =====================================
            print("\n📈 Normalizing scores...")
            print("-"*70)
            
            # FINAL FIXED: Proper normalization
            ela_score_normalized = min(max(ela_score * 3, 0), 100)
            noise_score_normalized = min(max(noise_score * 1.5, 0), 100)
            uniformity_score_normalized = min(max(uniformity_score, 0), 100)

            print(f"ELA Score (raw: {ela_score} → norm): {ela_score_normalized}")
            print(f"Noise Score (raw: {noise_score} → norm): {noise_score_normalized}")
            print(f"Uniformity Score: {uniformity_score_normalized}")

            # =====================================
            # FINAL SCORING
            # =====================================
            print("\n🎯 Calculating final score...")
            print("-"*70)
            
            # FINAL WORKING FORMULA
            final_score = round(
                (ela_score_normalized * 0.15) +      # ELA: 15%
                (noise_score_normalized * 0.4) +     # Noise: 40%
                ((100 - uniformity_score_normalized) * 0.15),  # Uniformity: 15%
                2
            )

            final_score = min(max(final_score, 0), 100)

            print(f"Calculation breakdown:")
            print(f"  ELA contribution (×0.15):    {ela_score_normalized * 0.15:.2f}")
            print(f"  Noise contribution (×0.4):   {noise_score_normalized * 0.4:.2f}")
            print(f"  Uniformity contrib (×0.15):  {(100 - uniformity_score_normalized) * 0.15:.2f}")
            print(f"  ─────────────────────────────")
            print(f"  FINAL SCORE:                 {final_score}")

            # =====================================
            # CLASSIFICATION - FINAL
            # =====================================
            print("\n🔍 Classification...")
            print("-"*70)
            
            print(f"Score Ranges:")
            print(f"  < 30:     REAL (high confidence)")
            print(f"  30-40:    SUSPICIOUS")
            print(f"  40-50:    LIKELY FAKE")
            print(f"  >= 50:    FAKE (high confidence)")
            print(f"\nFinal Score: {final_score}")
            
            if final_score >= 50:
                verdict = "FAKE"
                verdict_color = "#D9534F"
                risk_level = "HIGH"
                confidence = final_score
                print(f"✓ VERDICT: {verdict} (Risk: {risk_level}, Confidence: {confidence}%)")
            elif final_score >= 40:
                verdict = "LIKELY FAKE"
                verdict_color = "#EC971F"
                risk_level = "MEDIUM-HIGH"
                confidence = final_score
                print(f"✓ VERDICT: {verdict} (Risk: {risk_level}, Confidence: {confidence}%)")
            elif final_score >= 30:
                verdict = "SUSPICIOUS"
                verdict_color = "#F0AD4E"
                risk_level = "MEDIUM"
                confidence = final_score
                print(f"✓ VERDICT: {verdict} (Risk: {risk_level}, Confidence: {confidence}%)")
            else:
                verdict = "REAL"
                verdict_color = "#3FAF9A"
                risk_level = "LOW"
                confidence = round(100 - final_score, 2)
                print(f"✓ VERDICT: {verdict} (Risk: {risk_level}, Confidence: {confidence}%)")

            print("="*70 + "\n")

            # =====================================
            # RETURN RESULTS
            # =====================================
            return render_template(
                "index.html",
                uploaded_image=upload_path,
                ela_image=ela_path,
                heatmap_image=heatmap_path,
                boxed_image=boxed_path,
                noise_image=noise_path,
                hist_image=hist_path,
                edge_image=edge_path,
                final_score=final_score,
                verdict=verdict,
                verdict_color=verdict_color,
                risk_level=risk_level,
                confidence=confidence,
                noise_score=noise_score_normalized,
                ela_score=ela_score_normalized,
                uniformity_score=uniformity_score_normalized
            )

        except Exception as e:
            print("\n" + "="*70)
            print("❌ ERROR OCCURRED")
            print("="*70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("="*70 + "\n")
            return f"Error processing image: {str(e)}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
