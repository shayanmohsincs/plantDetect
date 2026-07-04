from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder=".", static_url_path="")

# Replace with your own API key
API_KEY = "QwYGMs9XgSgflcDy1JO2XIFfnvaJJHb2wBhuCZwE2GCbjmdBr9"

PLANT_ID_URL = "https://api.plant.id/v2/health_assessment"


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image = request.files["image"]

    # Check if the file has a name
    if image.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:
        response = requests.post(
            PLANT_ID_URL,
            headers={"Api-Key": API_KEY},
            files={"images": image}
        )

        # If Plant.id returns an error
        if response.status_code != 200:
            return jsonify({
                "error": "Plant.id API returned an error.",
                "status_code": response.status_code,
                "details": response.text
            }), response.status_code

        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to connect to Plant.id API.",
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
