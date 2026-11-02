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

PLACES_ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
REQUEST_TIMEOUT_SECONDS = 10
# The Text Search location bias only accepts circles up to 50 km.
MAX_RADIUS_METERS = 50_000
MAX_RESULTS = 20
PLACES_FIELD_MASK = ",".join(
    [
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.businessStatus",
    ]
)

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


def _included_type(specialty: str) -> str:
    """Map a specialty to a Places API (New) ``includedType``."""
    lowered = specialty.lower()
    if "hospital" in lowered:
        return "hospital"
    if "clinic" in lowered:
        return "medical_clinic"
    if "dentist" in lowered:
        return "dentist"
    # Cardiologist / dermatologist / general physician ... all search doctors.
    return "doctor"


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

    Uses the Places API (New) Text Search endpoint, which supports specialty
    keywords ("cardiologist") that Nearby Search's fixed types cannot.

    Raises:
        PlacesError: if the upstream request fails.
    """
    radius_meters = min(max(int(radius_km * 1000), 1), MAX_RADIUS_METERS)
    payload = {
        "textQuery": specialty,
        "includedType": _included_type(specialty),
        "maxResultCount": min(max(int(limit), 1), MAX_RESULTS),
        "rankPreference": "DISTANCE",
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": float(radius_meters),
            }
        },
    }

    http = session or requests
    try:
        response = http.post(
            PLACES_ENDPOINT,
            json=payload,
            headers={
                # The key travels in a header, never in the URL or a log line.
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": PLACES_FIELD_MASK,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Google Places request failed")
        raise PlacesError("Doctor search is temporarily unavailable.") from exc

    # An empty result set is an HTTP 200 with no "places" key.
    origin = (latitude, longitude)
    places: list[Place] = []
    for result in data.get("places") or []:
        if result.get("businessStatus") == "CLOSED_PERMANENTLY":
            continue

        location = result.get("location") or {}
        try:
            dest = (float(location["latitude"]), float(location["longitude"]))
        except (KeyError, TypeError, ValueError):
            continue

        places.append(
            Place(
                name=(result.get("displayName") or {}).get("text") or "Unknown place",
                address=result.get("formattedAddress") or "Address not available",
                distance_km=round(geodesic(origin, dest).km, 2),
                map_url=f"https://www.google.com/maps?q={dest[0]},{dest[1]}",
                rating=result.get("rating"),
                user_ratings_total=result.get("userRatingCount"),
            )
        )

    places.sort(key=lambda place: place.distance_km)
    return places[:limit]
