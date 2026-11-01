"""Doctor / hospital lookup via the Google Places API.

All network access and result shaping lives here so it can be unit tested
without Flask, and so the API key never ends up in a log line.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import requests
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

PLACES_ENDPOINT = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
REQUEST_TIMEOUT_SECONDS = 10

# Words that mean "this is a request for a real-world medical service".
DOCTOR_INTENT_KEYWORDS = (
    "doctor",
    "doctors",
    "hospital",
    "hospitals",
    "clinic",
    "clinics",
    "nearby",
    "near me",
    "specialist",
    "find me",
    "recommend a",
    "recommend me",
)

# Symptom / condition -> medical specialty used as the Places search keyword.
# Longer keys are matched first, so "chest pain" wins over "pain".
SPECIALTY_BY_SYMPTOM = {
    # Dermatology
    "skin": "dermatologist",
    "acne": "dermatologist",
    "rash": "dermatologist",
    "eczema": "dermatologist",
    "psoriasis": "dermatologist",
    "hair loss": "dermatologist",
    # Orthopedics
    "bone": "orthopedic surgeon",
    "fracture": "orthopedic surgeon",
    "back pain": "orthopedic surgeon",
    "knee pain": "orthopedic surgeon",
    "joint pain": "orthopedic surgeon",
    # Cardiology
    "heart": "cardiologist",
    "chest pain": "cardiologist",
    "blood pressure": "cardiologist",
    # Ophthalmology
    "eye": "ophthalmologist",
    "vision": "ophthalmologist",
    # Mental health
    "mental": "psychiatrist",
    "anxiety": "psychiatrist",
    "depression": "psychiatrist",
    # Ear / nose / throat
    "ear": "otolaryngologist",
    "throat": "otolaryngologist",
    "sinus": "otolaryngologist",
    # Dentistry
    "tooth": "dentist",
    "teeth": "dentist",
    "dental": "dentist",
    # Pediatrics
    "child": "pediatrician",
    "baby": "pediatrician",
    # Gastroenterology
    "stomach": "gastroenterologist",
    "digestion": "gastroenterologist",
    # Pulmonology
    "breathing": "pulmonologist",
    "asthma": "pulmonologist",
    # Neurology
    "headache": "neurologist",
    "migraine": "neurologist",
    "seizure": "neurologist",
    # Gynecology
    "pregnan": "gynecologist",
    "gynecolog": "gynecologist",
    # Urology
    "kidney": "urologist",
    "urine": "urologist",
    # Endocrinology
    "diabetes": "endocrinologist",
    "thyroid": "endocrinologist",
    # General
    "fever": "general physician",
    "cold": "general physician",
    "flu": "general physician",
    "cough": "general physician",
}

# Direct requests for a named professional ("find me a dermatologist").
PROFESSION_BY_TERM = {
    "dermatologist": "dermatologist",
    "cardiologist": "cardiologist",
    "orthopedic": "orthopedic surgeon",
    "ophthalmologist": "ophthalmologist",
    "eye doctor": "ophthalmologist",
    "psychiatrist": "psychiatrist",
    "psychologist": "psychiatrist",
    "otolaryngologist": "otolaryngologist",
    "dentist": "dentist",
    "pediatrician": "pediatrician",
    "child specialist": "pediatrician",
    "gastroenterologist": "gastroenterologist",
    "pulmonologist": "pulmonologist",
    "neurologist": "neurologist",
    "gynecologist": "gynecologist",
    "urologist": "urologist",
    "endocrinologist": "endocrinologist",
    "physician": "general physician",
}

# "find a dentist" / "recommend a cardiologist" — a request verb plus a
# profession name is a doctor search even without "near me".
REQUEST_PHRASES = (
    "find",
    "recommend",
    "suggest",
    "show me",
    "looking for",
    "need a",
    "need an",
    "where can i",
    "which doctor",
)


class PlacesError(RuntimeError):
    """Raised when the Places API cannot be reached or returns an error."""


@dataclass(frozen=True)
class Place:
    """A single doctor / hospital result, already distance-sorted-safe."""

    name: str
    address: str
    distance_km: float
    map_url: str
    rating: float | None = None
    user_ratings_total: int | None = None


def _mentions(keyword: str, text: str) -> bool:
    """Word-boundary match so "ear" does not fire inside "near"."""
    return re.search(rf"\b{re.escape(keyword)}", text) is not None


def is_doctor_request(text: str) -> bool:
    """True when the message is asking us to find a real medical service."""
    lowered = (text or "").lower()
    if any(keyword in lowered for keyword in DOCTOR_INTENT_KEYWORDS):
        return True
    mentioned_profession = any(_mentions(term, lowered) for term in PROFESSION_BY_TERM)
    asked_to_find = any(phrase in lowered for phrase in REQUEST_PHRASES)
    return mentioned_profession and asked_to_find


def specialty_for(text: str) -> str:
    """Map a free-text question to a Places search keyword."""
    lowered = (text or "").lower()
    # Explicit place types win: "nearest hospital" is about hospitals.
    if _mentions("hospital", lowered):
        return "hospital"
    if _mentions("clinic", lowered):
        return "clinic"
    for term in sorted(PROFESSION_BY_TERM, key=len, reverse=True):
        if _mentions(term, lowered):
            return PROFESSION_BY_TERM[term]
    for symptom in sorted(SPECIALTY_BY_SYMPTOM, key=len, reverse=True):
        if _mentions(symptom, lowered):
            return SPECIALTY_BY_SYMPTOM[symptom]
    return "doctor"


def _place_type_and_keyword(specialty: str) -> tuple[str, str]:
    lowered = specialty.lower()
    if "hospital" in lowered:
        return "hospital", "hospital"
    if "clinic" in lowered:
        return "clinic", "clinic"
    return "doctor", specialty


def find_nearby_places(
    latitude: float,
    longitude: float,
    specialty: str,
    *,
    api_key: str,
    radius_km: int = 50,
    limit: int = 5,
    session: requests.Session | None = None,
) -> list[Place]:
    """Return up to ``limit`` open places near a coordinate, nearest first.

    Raises:
        PlacesError: if the upstream request fails.
    """
    place_type, keyword = _place_type_and_keyword(specialty)
    params = {
        "location": f"{latitude},{longitude}",
        "radius": int(radius_km * 1000),
        "type": place_type,
        "keyword": keyword,
        "key": api_key,
    }

    http = session or requests
    try:
        response = http.get(PLACES_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Never log the URL: it contains the API key.
        logger.exception("Google Places request failed")
        raise PlacesError("Doctor search is temporarily unavailable.") from exc

    status = payload.get("status")
    if status == "ZERO_RESULTS":
        return []
    if status != "OK":
        logger.warning(
            "Google Places returned status=%s message=%s",
            status,
            payload.get("error_message", "n/a"),
        )
        return []

    origin = (latitude, longitude)
    places: list[Place] = []
    for result in payload.get("results", []):
        if result.get("business_status") == "CLOSED_PERMANENTLY":
            continue

        location = result.get("geometry", {}).get("location", {})
        try:
            dest = (float(location["lat"]), float(location["lng"]))
        except (KeyError, TypeError, ValueError):
            continue

        places.append(
            Place(
                name=result.get("name", "Unknown place"),
                address=result.get("vicinity") or result.get("formatted_address") or "Address not available",
                distance_km=round(geodesic(origin, dest).km, 2),
                map_url=f"https://www.google.com/maps?q={dest[0]},{dest[1]}",
                rating=result.get("rating"),
                user_ratings_total=result.get("user_ratings_total"),
            )
        )

    places.sort(key=lambda place: place.distance_km)
    return places[:limit]
