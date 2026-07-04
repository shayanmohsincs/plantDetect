from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import base64
import json
import re
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")

# Configure Gemini API from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set it in Railway dashboard or .env file")
    GEMINI_API_KEY = "placeholder"  # Will fail at runtime with proper error

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


def parse_gemini_response(response_text):
    """
    Parse Gemini's plant analysis and transform to expected JSON structure.
    Returns response in format matching Plant.id v2 health_assessment structure.
    """
    analysis = {
        "health_assessment": {
            "is_healthy": True,
            "is_healthy_probability": 1.0,
            "diseases": []
        }
    }

    # Normalize response text for analysis
    response_lower = response_text.lower()
    
    # Check if plant is healthy
    is_healthy = any(keyword in response_lower for keyword in [
        "healthy", "no disease", "no sign", "no visible disease", "looks good",
        "appears healthy", "well", "good condition", "no abnormality"
    ])
    
    # Extract disease information if detected
    if not is_healthy and any(word in response_lower for word in [
        "disease", "blight", "spot", "mold", "mildew", "rust", "infection",
        "infected", "diseased", "damage", "fungal", "bacterial", "viral"
    ]):
        analysis["health_assessment"]["is_healthy"] = False
        analysis["health_assessment"]["is_healthy_probability"] = 0.2
        
        # Parse disease details from response
        diseases = []
        lines = response_text.split('\n')
        
        # Extract disease information
        disease_keywords = ["disease", "blight", "spot", "mold", "mildew", "rust", "leaf", "infection"]
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in disease_keywords) and len(line) > 5:
                disease_name = extract_disease_name(line)
                
                # Look for treatment info in surrounding lines
                treatment_info = extract_treatment_info(lines, i)
                
                if disease_name and disease_name != "Unknown":
                    disease_obj = {
                        "name": disease_name,
                        "probability": 0.75,
                        "treatment": {
                            "chemical": treatment_info.get("chemical", ["Consult local agricultural expert"]),
                            "biological": treatment_info.get("biological", ["Remove affected parts"]),
                            "prevention": treatment_info.get("prevention", ["Monitor plant health"])
                        }
                    }
                    
                    # Check if we haven't already added this disease
                    if not any(d["name"].lower() == disease_name.lower() for d in diseases):
                        diseases.append(disease_obj)
        
        if diseases:
            analysis["health_assessment"]["diseases"] = diseases
        else:
            # Generic disease if parsing didn't work but disease detected
            analysis["health_assessment"]["diseases"] = [{
                "name": "Plant Disease Detected",
                "probability": 0.65,
                "treatment": {
                    "chemical": ["Apply appropriate fungicide or bactericide"],
                    "biological": ["Remove infected leaves and isolate plant"],
                    "prevention": ["Maintain proper plant care and hygiene"]
                }
            }]
    else:
        # Plant is healthy or analysis unclear
        analysis["health_assessment"]["is_healthy"] = True
        analysis["health_assessment"]["is_healthy_probability"] = 0.9
        analysis["health_assessment"]["diseases"] = []
    
    return analysis


def extract_disease_name(text):
    """Extract clean disease name from text"""
    # Remove common patterns
    text = re.sub(r'^[^:]*:\s*', '', text)  # Remove prefix before colon
    text = re.sub(r'[*#-]', '', text)  # Remove markdown
    text = text.strip()
    
    # Get first few words (disease names typically 1-4 words)
    words = text.split()[:4]
    name = ' '.join(words)
    
    # Capitalize properly
    name = ' '.join(word.capitalize() for word in name.split())
    
    return name if len(name) > 2 else None


def extract_treatment_info(lines, disease_line_idx):
    """Extract treatment information from surrounding lines"""
    info = {
        "chemical": [],
        "biological": [],
        "prevention": []
    }
    
    # Look in surrounding lines for treatment keywords
    search_range = min(10, len(lines) - disease_line_idx)
    
    for i in range(disease_line_idx, min(disease_line_idx + search_range, len(lines))):
        line = lines[i].lower()
        
        if "treatment" in line or "spray" in line or "fungicide" in line or "apply" in line:
            text = lines[i].strip()
            if text and len(text) > 5:
                info["chemical"].append(text.replace("*", "").replace("#", ""))
        
        if "prevention" in line or "prevent" in line or "avoid" in line:
            text = lines[i].strip()
            if text and len(text) > 5:
                info["prevention"].append(text.replace("*", "").replace("#", ""))
        
        if "remove" in line or "prune" in line or "isolate" in line:
            text = lines[i].strip()
            if text and len(text) > 5:
                info["biological"].append(text.replace("*", "").replace("#", ""))
    
    return info


@app.route("/detect", methods=["POST"])
def detect():
    """
    Detect plant disease using Gemini Vision API.
    Expects image in multipart form data and returns response matching
    Plant.id v2 health_assessment structure.
    """
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:
        # Read and validate image
        image_file.seek(0)
        image_data = image_file.read()
        
        if not image_data:
            return jsonify({"error": "Image file is empty"}), 400
        
        # Verify it's a valid image
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        # Check API key is set
        if not GEMINI_API_KEY or GEMINI_API_KEY == "placeholder":
            print("[ERROR] GEMINI_API_KEY not configured")
            return jsonify({
                "error": "Gemini API key not configured",
                "details": "Set GEMINI_API_KEY environment variable"
            }), 500
        
        print(f"[DEBUG] API Key set: {GEMINI_API_KEY[:10]}...")  # Log first 10 chars

        # Initialize Gemini model
        model = genai.GenerativeModel(MODEL_NAME)
        
        prompt = """You are an expert plant pathologist. Analyze this plant leaf image:

1. Is the plant healthy or diseased?
2. If diseased, what is the disease name?
3. Confidence level?
4. What treatments work?
5. How to prevent it?

Be specific and clear."""

        # Determine MIME type from filename
        filename = image_file.filename.lower()
        if filename.endswith('.png'):
            mime_type = "image/png"
        elif filename.endswith('.webp'):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        # Encode image to base64
        image_base64 = base64.standard_b64encode(image_data).decode("utf-8")
        
        print(f"[DEBUG] Image size: {len(image_data)} bytes, MIME: {mime_type}")

        # Call Gemini API with image
        try:
            response = model.generate_content([
                prompt,
                {
                    "mime_type": mime_type,
                    "data": image_base64
                }
            ])
            
            if not response or not response.text:
                print("[ERROR] Gemini returned empty response")
                return jsonify({
                    "error": "Gemini API returned empty response"
                }), 500
            
            response_text = response.text
            print(f"[DEBUG] Gemini Response: {response_text[:300]}")

        except Exception as gemini_error:
            print(f"[ERROR] Gemini API call failed: {type(gemini_error).__name__}: {str(gemini_error)}")
            return jsonify({
                "error": f"Gemini API error: {str(gemini_error)}"
            }), 500

        # Parse response
        diagnosis = parse_gemini_response(response_text)
        return jsonify(diagnosis)

    except Exception as e:
        print(f"[ERROR] Unexpected error in detect: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to process image",
            "details": str(e)
        }), 500


# Enable CORS so the frontend can communicate with the backend
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000)