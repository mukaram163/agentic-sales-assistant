from unittest.mock import MagicMock, patch
from app.tools.crm import create_qualified_lead


def test_create_qualified_lead_success():
    mock_result = MagicMock()
    mock_result.data = [{"id": "abc-123"}]

    with patch("app.tools.crm.supabase") as mock_supabase:
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result
        result = create_qualified_lead.invoke({"inquiry": "test inquiry", "reasoning": "test reason"})

    assert "abc-123" in result
    assert "qualifying" in result


def test_create_qualified_lead_db_failure():
    with patch("app.tools.crm.supabase") as mock_supabase:
        mock_supabase.table.side_effect = Exception("DB error")
        result = create_qualified_lead.invoke({"inquiry": "test", "reasoning": "test"})

    assert "failed" in result.lower()