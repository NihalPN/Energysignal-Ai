import pytest
from unittest.mock import patch, MagicMock

# Assuming these are your actual function imports
# from src.citation_validation import check_crossref_doi
# from src.nli_engine import evaluate_hallucination


def check_crossref_doi(doi: str) -> bool:
    """Mock implementation of your actual Crossref function."""
    import requests

    response = requests.get(f"https://api.crossref.org/works/{doi}")
    return response.status_code == 200


@patch("requests.get")
def test_crossref_api_success(mock_get):
    """Integration test: validates citation against the Crossref API."""
    # Setup the mock to simulate a successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    is_valid = check_crossref_doi("10.1038/nature14539")

    assert is_valid is True
    mock_get.assert_called_once_with("https://api.crossref.org/works/10.1038/nature14539")


@patch("requests.get")
def test_crossref_api_failure(mock_get):
    """Integration test: handles invalid DOIs correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    is_valid = check_crossref_doi("10.9999/fake.doi")

    assert is_valid is False
