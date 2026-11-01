"""Route-level tests for the Flask application factory."""

from __future__ import annotations

import pytest

import places
import rag
from app import create_app


@pytest.fixture
def client():
    application = create_app()
    application.config.update(TESTING=True)
    with application.test_client() as test_client:
        yield test_client


def test_index_renders_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"MedBot" in response.data


def test_healthz_reports_configured_keys(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["configured"] == {"google": True, "maps": True, "pinecone": True}


def test_healthz_never_leaks_key_values(client):
    response = client.get("/healthz")
    assert b"test-google-key" not in response.data
    assert b"test-maps-key" not in response.data
    assert b"test-pinecone-key" not in response.data


@pytest.mark.parametrize("payload", [{}, {"msg": ""}, {"msg": "   "}, {"msg": None}])
def test_empty_message_is_rejected(client, payload):
    response = client.post("/get", json=payload)
    assert response.status_code == 400
    assert response.get_json()["type"] == "text"


def test_overlong_message_is_rejected(client):
    response = client.post("/get", json={"msg": "a" * 1001})
    assert response.status_code == 400


def test_non_object_json_body_is_handled(client):
    response = client.post("/get", json=["not", "an", "object"])
    assert response.status_code == 400


def test_rag_path_returns_text(client, monkeypatch):
    monkeypatch.setattr(
        rag, "answer_question", lambda question: rag.Answer("Diabetes is a condition.", [])
    )
    response = client.post("/get", json={"msg": "What is diabetes?"})
    assert response.status_code == 200
    assert response.get_json() == {"type": "text", "response": "Diabetes is a condition."}


def test_rag_response_includes_sources(client, monkeypatch):
    answer = rag.Answer(
        "Asthma affects the airways.",
        [rag.Source("Medical_book.pdf", 412, 0.87)],
    )
    monkeypatch.setattr(rag, "answer_question", lambda question: answer)

    body = client.post("/get", json={"msg": "what is asthma"}).get_json()

    assert body["sources"] == [{"source": "Medical_book.pdf", "page": 412, "score": 0.87}]


def test_rag_failure_hides_internal_details(client, monkeypatch):
    def boom(question):
        raise RuntimeError("super-secret-internal-detail")

    monkeypatch.setattr(rag, "answer_question", boom)
    response = client.post("/get", json={"msg": "What is diabetes?"})
    assert response.status_code == 502
    assert b"super-secret-internal-detail" not in response.data


def test_doctor_request_without_location_asks_for_it(client):
    response = client.post("/get", json={"msg": "find me a cardiologist"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["type"] == "text"
    assert "location" in body["response"].lower()


def test_doctor_request_with_invalid_location(client):
    response = client.post("/get", json={"msg": "find a doctor", "location": "not-a-coordinate"})
    assert response.status_code == 200
    assert "invalid" in response.get_json()["response"].lower() or "couldn't read" in response.get_json()["response"].lower()


@pytest.mark.parametrize("location", ["91,0", "0,181", "abc,def", "", "1"])
def test_out_of_range_location_is_rejected(client, location):
    response = client.post("/get", json={"msg": "find a hospital", "location": location})
    assert response.get_json()["type"] == "text"


def test_doctor_results_are_returned_as_structured_data(client, monkeypatch):
    captured = {}

    def fake_find(lat, lon, specialty, *, api_key, radius_km, limit):
        captured.update(lat=lat, lon=lon, specialty=specialty, limit=limit)
        return [
            places.Place(
                name="City Hospital",
                address="1 Main St",
                distance_km=1.23,
                map_url="https://www.google.com/maps?q=1,2",
            )
        ]

    monkeypatch.setattr(places, "find_nearby_places", fake_find)
    response = client.post(
        "/get", json={"msg": "find me a dermatologist near me", "location": "12.9,77.5"}
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["type"] == "places"
    assert body["specialty"] == "dermatologist"
    assert body["places"][0]["distance_km"] == 1.23
    assert captured == {"lat": 12.9, "lon": 77.5, "specialty": "dermatologist", "limit": 5}


def test_places_failure_returns_generic_message(client, monkeypatch):
    def boom(*args, **kwargs):
        raise places.PlacesError("upstream said: quota exceeded for key ABC")

    monkeypatch.setattr(places, "find_nearby_places", boom)
    response = client.post("/get", json={"msg": "find a clinic", "location": "12.9,77.5"})
    assert response.status_code == 502
    assert b"quota exceeded" not in response.data


def test_specialty_is_passed_through_for_hospitals(client, monkeypatch):
    captured = {}

    def fake_find(lat, lon, specialty, **kwargs):
        captured["specialty"] = specialty
        return []

    monkeypatch.setattr(places, "find_nearby_places", fake_find)
    client.post("/get", json={"msg": "nearest hospital", "location": "12.9,77.5"})
    assert captured["specialty"] == "hospital"


def test_no_results_returns_a_helpful_message(client, monkeypatch):
    monkeypatch.setattr(places, "find_nearby_places", lambda *args, **kwargs: [])

    response = client.post("/get", json={"msg": "find a dentist", "location": "12.9,77.5"})

    body = response.get_json()
    assert response.status_code == 200
    assert body["type"] == "text"
    assert "couldn't find" in body["response"].lower()


def test_missing_configuration_returns_generic_error(client, monkeypatch):
    import app as medbot_app

    def not_configured(*args, **kwargs):
        raise RuntimeError("PINECONE_API_KEY is missing")

    monkeypatch.setattr(medbot_app, "require_settings", not_configured)

    response = client.post("/get", json={"msg": "find a doctor", "location": "12.9,77.5"})

    assert response.status_code == 500
    assert b"PINECONE_API_KEY" not in response.data
    assert response.get_json()["type"] == "text"

