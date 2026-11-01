"""Unit tests for the Google Places integration (no network access)."""

from __future__ import annotations

import pytest
import requests

import places
from places import Place, find_nearby_places, is_doctor_request, specialty_for


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


def result(name, lat, lon, *, status=None, rating=None, reviews=None):
    item = {"name": name, "vicinity": f"{name} street", "geometry": {"location": {"lat": lat, "lng": lon}}}
    if status:
        item["business_status"] = status
    if rating is not None:
        item["rating"] = rating
    if reviews is not None:
        item["user_ratings_total"] = reviews
    return item


@pytest.mark.parametrize(
    "message",
    [
        "find me a doctor",
        "any hospitals nearby?",
        "clinic near me",
        "recommend a specialist",
        "find a dentist",
        "I need a dermatologist",
        "which doctor should I see",
    ],
)
def test_detects_doctor_requests(message):
    assert is_doctor_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "what is diabetes",
        "how much paracetamol is safe",
        "explain insulin resistance",
        "find information about fever",
        "tell me about asthma",
    ],
)
def test_medical_questions_are_not_doctor_requests(message):
    assert not is_doctor_request(message)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I have chest pain", "cardiologist"),
        ("itchy skin rash", "dermatologist"),
        ("my tooth hurts", "dentist"),
        ("headache every day", "neurologist"),
        ("nearest hospital", "hospital"),
        ("a clinic please", "clinic"),
        ("I feel unwell", "doctor"),
    ],
)
def test_specialty_mapping(message, expected):
    assert specialty_for(message) == expected


def test_longest_symptom_keyword_wins():
    assert specialty_for("chest pain and a skin rash") == "cardiologist"


def test_results_are_sorted_by_distance_and_limited():
    session = FakeSession(
        {
            "status": "OK",
            "results": [
                result("Far Clinic", 1.0, 1.0),
                result("Near Clinic", 0.01, 0.01),
                result("Mid Clinic", 0.5, 0.5),
            ],
        }
    )

    results = find_nearby_places(0, 0, "doctor", api_key="secret", limit=2, session=session)

    assert [place.name for place in results] == ["Near Clinic", "Mid Clinic"]
    assert all(isinstance(place, Place) for place in results)
    assert results[0].distance_km < results[1].distance_km


def test_permanently_closed_places_are_dropped():
    session = FakeSession(
        {
            "status": "OK",
            "results": [
                result("Closed", 0.1, 0.1, status="CLOSED_PERMANENTLY"),
                result("Open", 0.2, 0.2, status="OPERATIONAL"),
            ],
        }
    )

    results = find_nearby_places(0, 0, "doctor", api_key="secret", session=session)

    assert [place.name for place in results] == ["Open"]


def test_api_key_goes_in_params_not_the_url():
    session = FakeSession({"status": "ZERO_RESULTS", "results": []})

    find_nearby_places(1.5, 2.5, "cardiologist", api_key="super-secret", session=session)

    call = session.calls[0]
    assert "super-secret" not in call["url"]
    assert call["params"]["key"] == "super-secret"
    assert call["params"]["keyword"] == "cardiologist"
    assert call["timeout"] == places.REQUEST_TIMEOUT_SECONDS


def test_zero_results_returns_empty_list():
    session = FakeSession({"status": "ZERO_RESULTS"})
    assert find_nearby_places(0, 0, "doctor", api_key="k", session=session) == []


def test_request_denied_raises_places_error():
    session = FakeSession({"status": "REQUEST_DENIED", "error_message": "bad key"})

    with pytest.raises(places.PlacesError):
        find_nearby_places(0, 0, "doctor", api_key="k", session=session)


def test_malformed_geometry_is_skipped():
    session = FakeSession(
        {
            "status": "OK",
            "results": [
                {"name": "Broken", "geometry": {}},
                result("Good", 0.1, 0.1),
            ],
        }
    )

    results = find_nearby_places(0, 0, "doctor", api_key="k", session=session)

    assert [place.name for place in results] == ["Good"]


def test_network_error_raises_places_error():
    session = FakeSession(error=requests.ConnectionError("network down"))

    with pytest.raises(places.PlacesError):
        find_nearby_places(0, 0, "doctor", api_key="k", session=session)


def test_hospital_specialty_uses_hospital_place_type():
    session = FakeSession({"status": "ZERO_RESULTS"})

    find_nearby_places(0, 0, "hospital", api_key="k", session=session)

    assert session.calls[0]["params"]["type"] == "hospital"
    assert session.calls[0]["params"]["keyword"] == "hospital"


def test_rating_is_passed_through():
    session = FakeSession(
        {"status": "OK", "results": [result("Rated", 0.1, 0.1, rating=4.6, reviews=120)]}
    )

    place = find_nearby_places(0, 0, "doctor", api_key="k", session=session)[0]

    assert place.rating == 4.6
    assert place.user_ratings_total == 120
