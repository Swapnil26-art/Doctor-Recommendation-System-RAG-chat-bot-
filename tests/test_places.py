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
    def __init__(self, payload=None, error=None, status_code=200):
        self.payload = {} if payload is None else payload
        self.error = error
        self.status_code = status_code
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self.error:
            raise self.error
        return FakeResponse(self.payload, self.status_code)


def result(name, lat, lon, *, status=None, rating=None, reviews=None):
    """Build a place in the Places API (New) response shape."""
    item = {
        "displayName": {"text": name},
        "formattedAddress": f"{name} street",
        "location": {"latitude": lat, "longitude": lon},
    }
    if status:
        item["businessStatus"] = status
    if rating is not None:
        item["rating"] = rating
    if reviews is not None:
        item["userRatingCount"] = reviews
    return item


def places_payload(*items):
    return {"places": list(items)}


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
        places_payload(
            result("Far Clinic", 1.0, 1.0),
            result("Near Clinic", 0.01, 0.01),
            result("Mid Clinic", 0.5, 0.5),
        )
    )

    results = find_nearby_places(0, 0, "doctor", api_key="secret", limit=2, session=session)

    assert [place.name for place in results] == ["Near Clinic", "Mid Clinic"]
    assert all(isinstance(place, Place) for place in results)
    assert results[0].distance_km < results[1].distance_km


def test_permanently_closed_places_are_dropped():
    session = FakeSession(
        places_payload(
            result("Closed", 0.1, 0.1, status="CLOSED_PERMANENTLY"),
            result("Open", 0.2, 0.2, status="OPERATIONAL"),
        )
    )

    results = find_nearby_places(0, 0, "doctor", api_key="secret", session=session)

    assert [place.name for place in results] == ["Open"]


def test_request_shape_uses_the_new_text_search_api():
    session = FakeSession({})

    find_nearby_places(1.5, 2.5, "cardiologist", api_key="super-secret", radius_km=8, session=session)

    call = session.calls[0]
    assert call["url"] == places.PLACES_ENDPOINT
    assert call["url"] == "https://places.googleapis.com/v1/places:searchText"
    assert "super-secret" not in call["url"]
    assert "super-secret" not in str(call["json"])
    assert call["headers"]["X-Goog-Api-Key"] == "super-secret"
    assert call["timeout"] == places.REQUEST_TIMEOUT_SECONDS

    body = call["json"]
    assert body["textQuery"] == "cardiologist"
    assert body["includedType"] == "doctor"
    assert body["rankPreference"] == "DISTANCE"
    assert body["locationBias"]["circle"]["center"] == {"latitude": 1.5, "longitude": 2.5}
    assert body["locationBias"]["circle"]["radius"] == 8000.0


def test_field_mask_requests_every_field_we_read():
    for field in (
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.businessStatus",
    ):
        assert field in places.PLACES_FIELD_MASK


def test_radius_is_clamped_to_the_api_maximum():
    session = FakeSession({})

    find_nearby_places(0, 0, "doctor", api_key="k", radius_km=500, session=session)

    assert session.calls[0]["json"]["locationBias"]["circle"]["radius"] == 50000.0


def test_result_count_is_clamped():
    session = FakeSession({})

    find_nearby_places(0, 0, "doctor", api_key="k", limit=999, session=session)

    assert session.calls[0]["json"]["maxResultCount"] == places.MAX_RESULTS


def test_zero_results_returns_empty_list():
    session = FakeSession({})
    assert find_nearby_places(0, 0, "doctor", api_key="k", session=session) == []


def test_denied_request_raises_places_error():
    session = FakeSession(
        {"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "bad key"}},
        status_code=403,
    )

    with pytest.raises(places.PlacesError):
        find_nearby_places(0, 0, "doctor", api_key="k", session=session)


def test_malformed_geometry_is_skipped():
    session = FakeSession(
        places_payload(
            {"displayName": {"text": "Broken"}, "formattedAddress": "nowhere"},
            result("Good", 0.1, 0.1),
        )
    )

    results = find_nearby_places(0, 0, "doctor", api_key="k", session=session)

    assert [place.name for place in results] == ["Good"]


def test_network_error_raises_places_error():
    session = FakeSession(error=requests.ConnectionError("network down"))

    with pytest.raises(places.PlacesError):
        find_nearby_places(0, 0, "doctor", api_key="k", session=session)


@pytest.mark.parametrize(
    ("specialty", "expected_type"),
    [
        ("hospital", "hospital"),
        ("clinic", "medical_clinic"),
        ("dentist", "dentist"),
        ("cardiologist", "doctor"),
        ("general physician", "doctor"),
    ],
)
def test_specialty_maps_to_included_type(specialty, expected_type):
    session = FakeSession({})

    find_nearby_places(0, 0, specialty, api_key="k", session=session)

    assert session.calls[0]["json"]["includedType"] == expected_type


def test_rating_is_passed_through():
    session = FakeSession(places_payload(result("Rated", 0.1, 0.1, rating=4.6, reviews=120)))

    place = find_nearby_places(0, 0, "doctor", api_key="k", session=session)[0]

    assert place.rating == 4.6
    assert place.user_ratings_total == 120
