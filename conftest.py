
"""Shared pytest fixtures for Flask application tests."""

import re

import pytest

from app import app


class CSRFTestClient:
    """Wrap the Flask test client to include valid CSRF tokens."""

    def __init__(self, client):
        self.client = client

    def post(self, url, **kwargs):
        """Submit a POST request with a valid CSRF token."""
        response = self.client.get("/")

        match = re.search(
            rb'name="csrf_token"\s+value="([^"]+)"',
            response.data,
        )
        assert match is not None, "CSRF token not found in form"

        token = match.group(1).decode("utf-8")

        data = kwargs.pop("data", None)
        if data is None:
            data = {}

        data = dict(data)
        data["csrf_token"] = token

        return self.client.post(url, data=data, **kwargs)

    def __getattr__(self, name):
        """Delegate other client methods to the Flask test client."""
        return getattr(self.client, name)


@pytest.fixture
def client():
    """Provide a Flask test client for existing acceptance tests."""
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield CSRFTestClient(test_client)
