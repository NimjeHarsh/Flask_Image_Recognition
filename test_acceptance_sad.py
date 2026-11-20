
"""Acceptance tests for missing image uploads."""


def test_acceptance_missing_file(client):
    """
    Test Case: No File Uploaded
    - Purpose: Validate the application's behavior when no file is provided.
    - Expected Behavior: Return an appropriate error message.
    """
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
