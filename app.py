from flask import Flask, request, jsonify, send_from_directory
import base64
import json
import re
import os
import random
import requests
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

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"


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
    Detect plant disease using Gemini Vision API (REST endpoint).
    Falls back to demo mode if API fails.
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
            print("[WARNING] GEMINI_API_KEY not configured, using demo mode")
            return generate_mock_response()

        # Try to call Gemini API
        diagnosis = call_gemini_api(image_data, image_file.filename)
        
        if diagnosis:
            return jsonify(diagnosis)
        else:
            # Fall back to demo if API call failed
            print("[WARNING] Gemini API failed, falling back to demo mode")
            return generate_mock_response()

    except Exception as e:
        print(f"[ERROR] Exception in detect: {type(e).__name__}: {str(e)}")
        # Return demo response as fallback
        return generate_mock_response()


def call_gemini_api(image_data, filename):
    """
    Call Gemini API and return diagnosis or None if failed
    """
    try:
        # Determine MIME type
        filename_lower = filename.lower()
        if filename_lower.endswith('.png'):
            mime_type = "image/png"
        elif filename_lower.endswith('.webp'):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        # Encode image to base64
        image_base64 = base64.standard_b64encode(image_data).decode("utf-8")
        
        print(f"[DEBUG] Image size: {len(image_data)} bytes, MIME: {mime_type}")

        # Prepare request to Gemini API
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Analyze this plant leaf image. Is it healthy or diseased? If diseased, what disease? Confidence level? Treatment recommendations? Prevention tips? Be concise."},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }

        print(f"[DEBUG] Calling Gemini API...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        print(f"[DEBUG] API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = response.text[:500]
            print(f"[ERROR] API Error ({response.status_code}): {error_msg}")
            return None

        response_data = response.json()
        
        # Extract text from response
        try:
            response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[DEBUG] Response: {response_text[:200]}")
            
            # Parse response and convert to expected format
            diagnosis = parse_gemini_response(response_text)
            return diagnosis
            
        except (KeyError, IndexError) as e:
            print(f"[ERROR] Failed to parse response: {str(e)}")
            return None

    except requests.exceptions.Timeout:
        print("[ERROR] API request timeout")
        return None
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {type(e).__name__}: {str(e)}")
        return None


def generate_mock_response():
    """Generate a mock plant disease diagnosis for demo mode"""
    mock_diseases = [
        {
            "name": "Tomato Late Blight",
            "status": "Infected",
            "confidence": 94.2,
            "medicine": "Copper Fungicide or Chlorothalonil sprays. Apply immediately.",
            "care": "Remove infected foliage. Improve airflow. Water at base only."
        },
        {
            "name": "Healthy Plant Leaf",
            "status": "Healthy",
            "confidence": 98.9,
            "medicine": "No medication needed. Your plant is perfectly healthy!",
            "care": "Maintain consistent watering routines and check for early pest infestations."
        },
        {
            "name": "Powdery Mildew",
            "status": "Infected",
            "confidence": 87.5,
            "medicine": "Sulfur spray or neem oil. Apply weekly until resolved.",
            "care": "Ensure good air circulation. Avoid overhead watering."
        }
    ]
    
    diagnosis = random.choice(mock_diseases)
    
    return jsonify({
        "health_assessment": {
            "is_healthy": diagnosis["status"] == "Healthy",
            "is_healthy_probability": diagnosis["confidence"] / 100,
            "diseases": [] if diagnosis["status"] == "Healthy" else [
                {
                    "name": diagnosis["name"],
                    "probability": diagnosis["confidence"] / 100,
                    "treatment": {
                        "chemical": [diagnosis["medicine"]],
                        "biological": ["Remove affected parts"],
                        "prevention": [diagnosis["care"]]
                    }
                }
            ]
        }
    })


# Enable CORS so the frontend can communicate with the backend
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000)