
"""Automated tests for the Flask application."""

import re
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from app import app
from model import preprocess_img


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def get_csrf_token(client):
    """Retrieve the CSRF token from the upload form."""
    response = client.get("/")
    match = re.search(
        rb'name="csrf_token"\s+value="([^"]+)"',
        response.data,
    )
    assert match is not None
    return match.group(1).decode("utf-8")


def test_home_page(client):
    """Verify that the home page loads successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_missing_upload(client):
    """Verify that a missing file returns an error message."""
    token = get_csrf_token(client)
    response = client.post(
        "/prediction",
        data={"csrf_token": token},
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_invalid_image(client):
    """Verify that invalid image data is rejected."""
    token = get_csrf_token(client)
    response = client.post(
        "/prediction",
        data={
            "csrf_token": token,
            "file": (BytesIO(b"not a real image"), "invalid.jpg"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_valid_prediction(client):
    """Verify that a valid image produces a prediction."""
    token = get_csrf_token(client)

    with patch("app.preprocess_img") as mock_preprocess, \
         patch("app.predict_result", return_value=5):

        mock_preprocess.return_value = "processed_image"

        response = client.post(
            "/prediction",
            data={
                "csrf_token": token,
                "file": (BytesIO(b"sample image"), "sample.jpg"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"5" in response.data


def test_missing_csrf_token(client):
    """Reject an upload request without a CSRF token."""
    response = client.post("/prediction", data={})

    assert response.status_code == 400


def test_invalid_csrf_token(client):
    """Reject an upload request with an invalid CSRF token."""
    response = client.post(
        "/prediction",
        data={"csrf_token": "invalid-token"},
    )

    assert response.status_code == 400


def test_image_preprocessing():
    """Verify image preprocessing produces the expected dimensions."""
    image = Image.new("RGB", (100, 100), color="white")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    processed_image = preprocess_img(image_buffer)

    assert processed_image.shape == (1, 224, 224, 3)
    assert processed_image.min() >= 0.0
    assert processed_image.max() <= 1.0
