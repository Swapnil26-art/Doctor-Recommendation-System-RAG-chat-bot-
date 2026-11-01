"""MedBot — Flask web app for medical Q&A plus nearby doctor recommendations.

Run locally:

    python app.py

Run in production (Linux container):

    gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 4 app:app
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

import places
import rag
from config import get_settings, require_settings

logger = logging.getLogger(__name__)

# Guards against trivially large prompts (cost / abuse).
MAX_MESSAGE_LENGTH = 1000

LOCATION_REQUIRED_MESSAGE = (
    "I can find doctors near you, but I need your location first. "
    "Allow location access in your browser and ask again."
)
LOCATION_INVALID_MESSAGE = (
    "I couldn't read your location. Please refresh the page, allow location "
    "access, and try again."
)
PLACES_ERROR_MESSAGE = (
    "Sorry, the doctor search service isn't responding right now. Please try again in a moment."
)
RAG_ERROR_MESSAGE = (
    "Sorry, I couldn't reach the medical knowledge base right now. Please try again in a moment."
)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def parse_location(raw: object) -> tuple[float, float] | None:
    """Parse the ``"lat,lon"`` string sent by the browser."""
    if not isinstance(raw, str) or "," not in raw:
        return None
    latitude_text, _, longitude_text = raw.partition(",")
    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
    except ValueError:
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return latitude, longitude


def create_app() -> Flask:
    """Application factory (no heavy models are loaded here)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        """Liveness probe that also reports which keys are configured."""
        missing = set(get_settings().missing_keys)
        return jsonify(
            {
                "status": "ok",
                "configured": {
                    "google": "GOOGLE_API_KEY" not in missing,
                    "maps": "GOOGLE_MAPS_API_KEY" not in missing,
                    "pinecone": "PINECONE_API_KEY" not in missing,
                },
            }
        )

    @app.post("/get")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = {}
        message = str(payload.get("msg") or "").strip()

        if not message:
            return jsonify({"type": "text", "response": "Please enter a question."}), 400
        if len(message) > MAX_MESSAGE_LENGTH:
            return (
                jsonify(
                    {
                        "type": "text",
                        "response": f"That question is too long. Please keep it under {MAX_MESSAGE_LENGTH} characters.",
                    }
                ),
                400,
            )

        if places.is_doctor_request(message):
            return _doctor_response(payload, message)
        return _rag_response(message)

    return app


def _doctor_response(payload: dict, message: str):
    specialty = places.specialty_for(message)
    coordinates = parse_location(payload.get("location"))

    if coordinates is None:
        response = LOCATION_INVALID_MESSAGE if payload.get("location") else LOCATION_REQUIRED_MESSAGE
        return jsonify({"type": "text", "response": response})

    try:
        settings = require_settings()
        results = places.find_nearby_places(
            coordinates[0],
            coordinates[1],
            specialty,
            api_key=settings.google_maps_api_key,
            radius_km=settings.places_radius_km,
            limit=settings.places_max_results,
        )
    except places.PlacesError:
        logger.exception("Doctor search failed")
        return jsonify({"type": "text", "response": PLACES_ERROR_MESSAGE}), 502
    except RuntimeError:
        logger.exception("Doctor search is not configured")
        return jsonify({"type": "text", "response": PLACES_ERROR_MESSAGE}), 500

    if not results:
        return jsonify(
            {
                "type": "text",
                "response": (
                    f"I couldn't find any {specialty}s within "
                    f"{get_settings().places_radius_km} km of you. "
                    "You may be outside the search radius."
                ),
            }
        )

    logger.info("Found %d places for specialty=%s", len(results), specialty)
    return jsonify(
        {
            "type": "places",
            "specialty": specialty,
            "intro": f"Here are the closest {specialty}s and hospitals I found near you:",
            "places": [asdict(place) for place in results],
        }
    )


def _rag_response(message: str):
    try:
        answer = rag.answer_question(message)
    except Exception:
        # Log the real error server-side; never echo internals to the browser.
        logger.exception("RAG request failed")
        return jsonify({"type": "text", "response": RAG_ERROR_MESSAGE}), 502
    return jsonify({"type": "text", "response": answer})


app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    logger.info("Starting MedBot on http://%s:%s", host, port)
    app.run(host=host, port=port, debug=False)
