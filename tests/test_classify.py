from unittest.mock import MagicMock, patch

from app.nodes.classify import classify_inquiry


def test_classify_qualified_lead():
    mock_response = MagicMock()
    mock_response.content = "Classification: qualified_lead\nReasoning: Genuine interest expressed."

    with patch("app.nodes.classify.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        state = {"inquiry": "I'd like to learn about your services"}
        result = classify_inquiry(state)

    assert result["classification"] == "qualified_lead"
    assert "Genuine interest" in result["reasoning"]


def test_classify_handles_llm_failure():
    with patch("app.nodes.classify.llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("API down")
        state = {"inquiry": "test"}
        result = classify_inquiry(state)

    assert result["classification"] == "unknown"
