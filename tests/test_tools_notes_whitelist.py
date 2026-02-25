"""Tests for note tool whitelist enforcement."""

from unittest.mock import MagicMock, patch

import pytest


def _get_tool_fn(tool):
    """Get the underlying function from a tool (handles both wrapped and unwrapped)."""
    if hasattr(tool, "fn"):
        return tool.fn
    return tool


# === Fixtures ===


@pytest.fixture
def mock_whitelist_config():
    """Enable whitelist in _module_config for note tools."""
    with patch("joplin_mcp.tools.notes._module_config") as mock_cfg:
        mock_cfg.has_notebook_whitelist = True
        mock_cfg.notebook_whitelist = ["AI", "Projects/*"]
        # Preserve other config attributes that note tools may need
        mock_cfg.should_show_content.return_value = True
        mock_cfg.should_show_full_content.return_value = True
        mock_cfg.get_max_preview_length.return_value = 300
        mock_cfg.is_smart_toc_enabled.return_value = False
        mock_cfg.get_smart_toc_threshold.return_value = 2000
        mock_cfg.tools = {}
        yield mock_cfg


@pytest.fixture
def mock_no_whitelist_config():
    """Explicitly disable whitelist in _module_config for backward compat tests."""
    with patch("joplin_mcp.tools.notes._module_config") as mock_cfg:
        mock_cfg.has_notebook_whitelist = False
        mock_cfg.notebook_whitelist = None
        mock_cfg.should_show_content.return_value = True
        mock_cfg.should_show_full_content.return_value = True
        mock_cfg.get_max_preview_length.return_value = 300
        mock_cfg.is_smart_toc_enabled.return_value = False
        mock_cfg.get_smart_toc_threshold.return_value = 2000
        mock_cfg.tools = {}
        yield mock_cfg


# === Tests for create_note with whitelist ===


class TestCreateNoteWhitelist:
    """Tests for create_note whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_create_note_whitelisted_notebook(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when target notebook is whitelisted."""
        from joplin_mcp.tools.notes import create_note

        mock_get_nb_id.return_value = "whitelisted_nb_id"
        mock_client = MagicMock()
        mock_client.add_note.return_value = "new_note_id"
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(create_note)
        result = await fn(title="Test Note", notebook_name="AI", body="content")

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        mock_client.add_note.assert_called_once()
        assert "SUCCESS" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_create_note_non_whitelisted_notebook(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when target notebook is not whitelisted."""
        from joplin_mcp.tools.notes import create_note

        mock_get_nb_id.return_value = "blocked_nb_id"
        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(create_note)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(title="Bad Note", notebook_name="Secret", body="content")

        mock_client = mock_get_client.return_value
        mock_client.add_note.assert_not_called()


# === Tests for get_note with whitelist ===


class TestGetNoteWhitelist:
    """Tests for get_note whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_get_note_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when note is in a whitelisted notebook."""
        from joplin_mcp.tools.notes import get_note

        mock_note = MagicMock()
        mock_note.parent_id = "whitelisted_nb_id"
        mock_note.title = "Test Note"
        mock_note.body = "content"
        mock_note.id = "12345678901234567890123456789012"
        mock_note.created_time = 1609459200000
        mock_note.updated_time = 1609545600000
        mock_note.is_todo = 0
        mock_note.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(get_note)
        # Should not raise
        result = await fn(note_id="12345678901234567890123456789012")

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        assert result is not None

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_get_note_non_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when note is in a non-whitelisted notebook."""
        from joplin_mcp.tools.notes import get_note

        mock_note = MagicMock()
        mock_note.parent_id = "blocked_nb_id"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(get_note)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(note_id="12345678901234567890123456789012")


# === Tests for update_note with whitelist ===


class TestUpdateNoteWhitelist:
    """Tests for update_note whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_update_note_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when note is in a whitelisted notebook."""
        from joplin_mcp.tools.notes import update_note

        mock_note = MagicMock()
        mock_note.parent_id = "whitelisted_nb_id"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(update_note)
        result = await fn(
            note_id="12345678901234567890123456789012",
            title="Updated Title",
        )

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        mock_client.modify_note.assert_called_once()
        assert "SUCCESS" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_update_note_non_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when note is in a non-whitelisted notebook."""
        from joplin_mcp.tools.notes import update_note

        mock_note = MagicMock()
        mock_note.parent_id = "blocked_nb_id"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(update_note)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(
                note_id="12345678901234567890123456789012",
                title="Should Fail",
            )

        mock_client.modify_note.assert_not_called()


# === Tests for edit_note with whitelist ===


class TestEditNoteWhitelist:
    """Tests for edit_note whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_edit_note_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when note is in a whitelisted notebook."""
        from joplin_mcp.tools.notes import edit_note

        mock_note = MagicMock()
        mock_note.parent_id = "whitelisted_nb_id"
        mock_note.body = "old text here"
        mock_note.title = "Test"
        mock_note.id = "12345678901234567890123456789012"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(edit_note)
        result = await fn(
            note_id="12345678901234567890123456789012",
            old_string="old text",
            new_string="new text",
        )

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        assert "EDIT_NOTE" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_edit_note_non_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when note is in a non-whitelisted notebook."""
        from joplin_mcp.tools.notes import edit_note

        mock_note = MagicMock()
        mock_note.parent_id = "blocked_nb_id"
        mock_note.body = "content"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(edit_note)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(
                note_id="12345678901234567890123456789012",
                old_string="content",
                new_string="modified",
            )

        mock_client.modify_note.assert_not_called()


# === Tests for delete_note with whitelist ===


class TestDeleteNoteWhitelist:
    """Tests for delete_note whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_delete_note_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when note is in a whitelisted notebook."""
        from joplin_mcp.tools.notes import delete_note

        mock_note = MagicMock()
        mock_note.parent_id = "whitelisted_nb_id"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(delete_note)
        result = await fn(note_id="12345678901234567890123456789012")

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        mock_client.delete_note.assert_called_once()
        assert "SUCCESS" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_delete_note_non_whitelisted(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when note is in a non-whitelisted notebook."""
        from joplin_mcp.tools.notes import delete_note

        mock_note = MagicMock()
        mock_note.parent_id = "blocked_nb_id"

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(delete_note)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(note_id="12345678901234567890123456789012")

        mock_client.delete_note.assert_not_called()


# === Tests for find_notes with whitelist ===


class TestFindNotesWhitelist:
    """Tests for find_notes whitelist filtering."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.is_notebook_accessible")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_find_notes_filters_results(
        self,
        mock_get_client,
        mock_is_accessible,
        mock_whitelist_config,
    ):
        """Only notes in whitelisted notebooks should be returned."""
        from joplin_mcp.tools.notes import find_notes

        note_ok = MagicMock()
        note_ok.parent_id = "whitelisted_nb_id"
        note_ok.id = "note_ok_id_000000000000000000000"
        note_ok.title = "Good Note"
        note_ok.updated_time = 1609545600000
        note_ok.is_todo = 0
        note_ok.todo_completed = 0

        note_blocked = MagicMock()
        note_blocked.parent_id = "blocked_nb_id"
        note_blocked.id = "note_blocked_id_0000000000000000"
        note_blocked.title = "Secret Note"
        note_blocked.updated_time = 1609459200000
        note_blocked.is_todo = 0
        note_blocked.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_all_notes.return_value = [note_ok, note_blocked]
        mock_get_client.return_value = mock_client

        def accessible_side_effect(parent_id, whitelist_entries=None):
            return parent_id == "whitelisted_nb_id"

        mock_is_accessible.side_effect = accessible_side_effect

        fn = _get_tool_fn(find_notes)
        result = await fn(query="*", limit=20)

        # is_notebook_accessible should have been called for filtering
        assert mock_is_accessible.call_count >= 1
        # Result should contain the good note but not the blocked one
        assert "Good Note" in result
        assert "Secret Note" not in result


# === Tests for find_notes_with_tag with whitelist ===


class TestFindNotesWithTagWhitelist:
    """Tests for find_notes_with_tag whitelist filtering."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.is_notebook_accessible")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_find_notes_with_tag_filters_results(
        self,
        mock_get_client,
        mock_is_accessible,
        mock_whitelist_config,
    ):
        """Only notes in whitelisted notebooks returned for tag search."""
        from joplin_mcp.tools.notes import find_notes_with_tag

        note_ok = MagicMock()
        note_ok.parent_id = "whitelisted_nb_id"
        note_ok.id = "note_ok_id_000000000000000000000"
        note_ok.title = "Tagged Good Note"
        note_ok.updated_time = 1609545600000
        note_ok.is_todo = 0
        note_ok.todo_completed = 0

        note_blocked = MagicMock()
        note_blocked.parent_id = "blocked_nb_id"
        note_blocked.id = "note_blocked_id_0000000000000000"
        note_blocked.title = "Tagged Secret Note"
        note_blocked.updated_time = 1609459200000
        note_blocked.is_todo = 0
        note_blocked.todo_completed = 0

        mock_client = MagicMock()
        mock_client.search_all.return_value = [note_ok, note_blocked]
        mock_get_client.return_value = mock_client

        def accessible_side_effect(parent_id, whitelist_entries=None):
            return parent_id == "whitelisted_nb_id"

        mock_is_accessible.side_effect = accessible_side_effect

        fn = _get_tool_fn(find_notes_with_tag)
        result = await fn(tag_name="work")

        assert mock_is_accessible.call_count >= 1
        assert "Tagged Good Note" in result
        assert "Tagged Secret Note" not in result


# === Tests for find_notes_in_notebook with whitelist ===


class TestFindNotesInNotebookWhitelist:
    """Tests for find_notes_in_notebook whitelist validation."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_find_notes_in_notebook_whitelisted(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should succeed when target notebook is whitelisted."""
        from joplin_mcp.tools.notes import find_notes_in_notebook

        mock_get_nb_id.return_value = "whitelisted_nb_id"

        note = MagicMock()
        note.parent_id = "whitelisted_nb_id"
        note.id = "note_id_00000000000000000000000"
        note.title = "Note in Whitelisted"
        note.updated_time = 1609545600000
        note.is_todo = 0
        note.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_all_notes.return_value = [note]
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(find_notes_in_notebook)
        result = await fn(notebook_name="AI")

        mock_validate.assert_called_once_with(
            "whitelisted_nb_id",
            whitelist_entries=mock_whitelist_config.notebook_whitelist,
        )
        assert "Note in Whitelisted" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_find_notes_in_notebook_non_whitelisted(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_validate,
        mock_whitelist_config,
    ):
        """Should raise error when target notebook is not whitelisted."""
        from joplin_mcp.tools.notes import find_notes_in_notebook

        mock_get_nb_id.return_value = "blocked_nb_id"
        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(find_notes_in_notebook)
        with pytest.raises(ValueError, match="Notebook not accessible"):
            await fn(notebook_name="Secret")


# === Tests for get_all_notes with whitelist ===


class TestGetAllNotesWhitelist:
    """Tests for get_all_notes whitelist filtering."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.is_notebook_accessible")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_get_all_notes_filters_results(
        self,
        mock_get_client,
        mock_is_accessible,
        mock_whitelist_config,
    ):
        """Only notes in whitelisted notebooks should be returned."""
        from joplin_mcp.tools.notes import get_all_notes

        note_ok = MagicMock()
        note_ok.parent_id = "whitelisted_nb_id"
        note_ok.id = "note_ok_id_000000000000000000000"
        note_ok.title = "Allowed Note"
        note_ok.updated_time = 1609545600000
        note_ok.is_todo = 0
        note_ok.todo_completed = 0

        note_blocked = MagicMock()
        note_blocked.parent_id = "blocked_nb_id"
        note_blocked.id = "note_blocked_id_0000000000000000"
        note_blocked.title = "Hidden Note"
        note_blocked.updated_time = 1609459200000
        note_blocked.is_todo = 0
        note_blocked.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_all_notes.return_value = [note_ok, note_blocked]
        mock_get_client.return_value = mock_client

        def accessible_side_effect(parent_id, whitelist_entries=None):
            return parent_id == "whitelisted_nb_id"

        mock_is_accessible.side_effect = accessible_side_effect

        fn = _get_tool_fn(get_all_notes)
        result = await fn()

        assert mock_is_accessible.call_count >= 1
        assert "Allowed Note" in result
        assert "Hidden Note" not in result


# === Backward compatibility tests ===


class TestNoteWhitelistBackwardCompat:
    """Verify all note tools work normally when no whitelist is configured."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_create_note_works_without_whitelist(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_no_whitelist_config,
    ):
        """create_note succeeds without whitelist."""
        from joplin_mcp.tools.notes import create_note

        mock_get_nb_id.return_value = "nb_id"
        mock_client = MagicMock()
        mock_client.add_note.return_value = "note_id"
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(create_note)
        result = await fn(title="Test", notebook_name="Work", body="content")
        assert "SUCCESS" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_get_note_works_without_whitelist(
        self,
        mock_get_client,
        mock_no_whitelist_config,
    ):
        """get_note succeeds without whitelist."""
        from joplin_mcp.tools.notes import get_note

        mock_note = MagicMock()
        mock_note.parent_id = "nb_id"
        mock_note.title = "Test Note"
        mock_note.body = "content"
        mock_note.id = "12345678901234567890123456789012"
        mock_note.created_time = 1609459200000
        mock_note.updated_time = 1609545600000
        mock_note.is_todo = 0
        mock_note.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(get_note)
        result = await fn(note_id="12345678901234567890123456789012")
        assert result is not None

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_delete_note_works_without_whitelist(
        self,
        mock_get_client,
        mock_no_whitelist_config,
    ):
        """delete_note succeeds without whitelist."""
        from joplin_mcp.tools.notes import delete_note

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(delete_note)
        result = await fn(note_id="12345678901234567890123456789012")
        assert "SUCCESS" in result


# === Error message tests (D7) ===


class TestNoteWhitelistErrorMessages:
    """Verify error messages are generic and do not leak notebook details."""

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_get_note_error_does_not_contain_notebook_id(
        self,
        mock_get_client,
        mock_validate,
        mock_whitelist_config,
    ):
        """Error message should not contain the notebook ID or name."""
        from joplin_mcp.tools.notes import get_note

        blocked_nb_id = "secret_private_nb_id_0000000000"
        mock_note = MagicMock()
        mock_note.parent_id = blocked_nb_id

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note
        mock_get_client.return_value = mock_client

        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(get_note)
        with pytest.raises(ValueError) as exc_info:
            await fn(note_id="12345678901234567890123456789012")

        error_msg = str(exc_info.value)
        assert blocked_nb_id not in error_msg
        assert "secret" not in error_msg.lower()
        assert "private" not in error_msg.lower()
        assert "Notebook not accessible" in error_msg

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.notes.validate_notebook_access")
    @patch("joplin_mcp.tools.notes.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.notes.get_joplin_client")
    async def test_create_note_error_does_not_contain_notebook_name(
        self,
        mock_get_client,
        mock_get_nb_id,
        mock_validate,
        mock_whitelist_config,
    ):
        """Error message should not contain the target notebook name."""
        from joplin_mcp.tools.notes import create_note

        mock_get_nb_id.return_value = "some_nb_id"
        mock_validate.side_effect = ValueError("Notebook not accessible")

        fn = _get_tool_fn(create_note)
        with pytest.raises(ValueError) as exc_info:
            await fn(
                title="Test",
                notebook_name="My Private Diary",
                body="content",
            )

        error_msg = str(exc_info.value)
        assert "My Private Diary" not in error_msg
        assert "Notebook not accessible" in error_msg
