"""Integration tests for end-to-end notebook whitelist workflow.

These tests exercise the full stack from config through notebook_utils
through tools, with only the Joplin API mocked. They verify:
- D2: Hierarchical access (parent whitelist grants child access)
- D3: Startup validation (server starts with valid/invalid whitelist)
- D4: All enforcement points (full tool chain exercised)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from joplin_mcp.config import JoplinMCPConfig
from joplin_mcp.notebook_utils import (
    filter_accessible_notebooks,
    invalidate_notebook_map_cache,
    is_notebook_accessible,
    validate_notebook_access,
    validate_whitelist_at_startup,
)


def _get_tool_fn(tool):
    """Get the underlying function from a tool (handles both wrapped and unwrapped)."""
    if hasattr(tool, "fn"):
        return tool.fn
    return tool


def _make_mock_client(notebooks):
    """Create a mock Joplin client that returns the given notebook list."""
    client = MagicMock()
    client.get_all_notebooks.return_value = notebooks
    client.ping.return_value = "JoplinClipperServer"
    return client


# ---------------------------------------------------------------------------
# Test 1: End-to-end whitelist workflow
# Config -> startup validation -> tool call -> access check
# ---------------------------------------------------------------------------


class TestEndToEndWhitelistWorkflow:
    """Integration test: config with whitelist, startup, tool calls, access checks."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    def test_config_to_access_check_flow(self, mock_notebook_hierarchy):
        """Full workflow: create config, validate startup, check access.

        Exercises: JoplinMCPConfig -> validate_whitelist_at_startup ->
        is_notebook_accessible for allowed and denied notebooks.
        """
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)

        # Step 1: Create config with whitelist
        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=["Projects", "AI"],
        )
        assert config.has_notebook_whitelist is True
        assert config.notebook_whitelist == ["Projects", "AI"]

        # Step 2: Validate at startup (should not raise)
        validate_whitelist_at_startup(config, client)

        # Step 3: Check access for whitelisted notebooks
        client_fn = lambda: client  # noqa: E731
        invalidate_notebook_map_cache()

        assert is_notebook_accessible(
            ids["Projects"], whitelist_entries=config.notebook_whitelist,
            client_fn=client_fn,
        ) is True

        assert is_notebook_accessible(
            ids["AI"], whitelist_entries=config.notebook_whitelist,
            client_fn=client_fn,
        ) is True

        # Step 4: Check access denied for non-whitelisted notebooks
        assert is_notebook_accessible(
            ids["Personal"], whitelist_entries=config.notebook_whitelist,
            client_fn=client_fn,
        ) is False

        assert is_notebook_accessible(
            ids["Diary"], whitelist_entries=config.notebook_whitelist,
            client_fn=client_fn,
        ) is False

    @pytest.mark.asyncio
    async def test_tool_call_with_whitelist_allowed(self, mock_notebook_hierarchy):
        """Tool call succeeds for note in whitelisted notebook."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        # Mock note in a whitelisted notebook (Projects/Work)
        mock_note = MagicMock()
        mock_note.parent_id = ids["Work"]
        mock_note.title = "Work Note"
        mock_note.body = "Work content"
        mock_note.id = "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"
        mock_note.created_time = 1609459200000
        mock_note.updated_time = 1609545600000
        mock_note.is_todo = 0
        mock_note.todo_completed = 0
        client.get_note.return_value = mock_note

        whitelist = ["Projects"]

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=client),
            patch(
                "joplin_mcp.tools.notes.validate_notebook_access",
                side_effect=lambda nb_id, whitelist_entries=None, **kw: (
                    # Use real validation with our mock client
                    validate_notebook_access(
                        nb_id,
                        whitelist_entries=whitelist_entries,
                        client_fn=client_fn,
                    )
                ),
            ),
        ):
            mock_cfg.has_notebook_whitelist = True
            mock_cfg.notebook_whitelist = whitelist
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import get_note

            fn = _get_tool_fn(get_note)
            result = await fn(note_id="a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0")

            assert "Work Note" in result

    @pytest.mark.asyncio
    async def test_tool_call_with_whitelist_denied(self, mock_notebook_hierarchy):
        """Tool call raises ValueError for note in non-whitelisted notebook."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        # Mock note in a non-whitelisted notebook (Personal/Diary)
        mock_note = MagicMock()
        mock_note.parent_id = ids["Diary"]
        client.get_note.return_value = mock_note

        whitelist = ["Projects"]

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=client),
            patch(
                "joplin_mcp.tools.notes.validate_notebook_access",
                side_effect=lambda nb_id, whitelist_entries=None, **kw: (
                    validate_notebook_access(
                        nb_id,
                        whitelist_entries=whitelist_entries,
                        client_fn=client_fn,
                    )
                ),
            ),
        ):
            mock_cfg.has_notebook_whitelist = True
            mock_cfg.notebook_whitelist = whitelist
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import get_note

            fn = _get_tool_fn(get_note)
            with pytest.raises(ValueError, match="Notebook not accessible"):
                await fn(note_id="b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1")


# ---------------------------------------------------------------------------
# Test 2: Hierarchical access integration (D2)
# Whitelist parent, verify child notebook note operations work
# ---------------------------------------------------------------------------


class TestHierarchicalAccessIntegration:
    """Integration test: whitelist parent grants access to child notebook notes."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    def test_parent_whitelist_grants_child_access(self, mock_notebook_hierarchy):
        """Whitelisting 'Projects' grants access to notes in Projects/Work and Projects/Fun."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects"]

        # Direct child: Projects/Work
        assert is_notebook_accessible(
            ids["Work"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Direct child: Projects/Fun
        assert is_notebook_accessible(
            ids["Fun"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Non-child: Personal
        assert is_notebook_accessible(
            ids["Personal"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is False

    def test_deep_hierarchy_access(self):
        """Whitelisting top-level grants access through multiple levels."""
        invalidate_notebook_map_cache()

        # Build a 3-level hierarchy: Root > Middle > Deep
        notebooks = [
            SimpleNamespace(id="root_id_000000000000000000000", title="Root", parent_id=""),
            SimpleNamespace(id="mid_id_0000000000000000000000", title="Middle", parent_id="root_id_000000000000000000000"),
            SimpleNamespace(id="deep_id_000000000000000000000", title="Deep", parent_id="mid_id_0000000000000000000000"),
        ]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        # Whitelist only "Root" -- should grant access to Root/Middle/Deep
        assert is_notebook_accessible(
            "deep_id_000000000000000000000",
            whitelist_entries=["Root"],
            client_fn=client_fn,
        ) is True

    @pytest.mark.asyncio
    async def test_create_note_in_child_of_whitelisted_parent(
        self, mock_notebook_hierarchy
    ):
        """create_note succeeds when target notebook is child of whitelisted parent."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client.add_note.return_value = "new_note_in_work_id"
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects"]

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=client),
            patch(
                "joplin_mcp.tools.notes.get_notebook_id_by_name",
                return_value=ids["Work"],
            ),
            patch(
                "joplin_mcp.tools.notes.validate_notebook_access",
                side_effect=lambda nb_id, whitelist_entries=None, **kw: (
                    validate_notebook_access(
                        nb_id,
                        whitelist_entries=whitelist_entries,
                        client_fn=client_fn,
                    )
                ),
            ),
        ):
            mock_cfg.has_notebook_whitelist = True
            mock_cfg.notebook_whitelist = whitelist
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import create_note

            fn = _get_tool_fn(create_note)
            result = await fn(title="Work Task", notebook_name="Work", body="task body")
            assert "SUCCESS" in result

    @pytest.mark.asyncio
    async def test_create_note_in_non_child_denied(self, mock_notebook_hierarchy):
        """create_note fails when target notebook is NOT a child of whitelisted parent."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects"]

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=client),
            patch(
                "joplin_mcp.tools.notes.get_notebook_id_by_name",
                return_value=ids["Diary"],
            ),
            patch(
                "joplin_mcp.tools.notes.validate_notebook_access",
                side_effect=lambda nb_id, whitelist_entries=None, **kw: (
                    validate_notebook_access(
                        nb_id,
                        whitelist_entries=whitelist_entries,
                        client_fn=client_fn,
                    )
                ),
            ),
        ):
            mock_cfg.has_notebook_whitelist = True
            mock_cfg.notebook_whitelist = whitelist
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import create_note

            fn = _get_tool_fn(create_note)
            with pytest.raises(ValueError, match="Notebook not accessible"):
                await fn(title="Diary Entry", notebook_name="Diary", body="private")


# ---------------------------------------------------------------------------
# Test 3: Backward compatibility (no whitelist configured)
# All operations succeed when whitelist is not set
# ---------------------------------------------------------------------------


class TestBackwardCompatibilityIntegration:
    """Integration test: no whitelist configured, all tools work normally."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    @pytest.mark.asyncio
    async def test_get_note_works_without_whitelist(self):
        """get_note succeeds for any notebook when no whitelist is configured."""
        mock_note = MagicMock()
        mock_note.parent_id = "any_notebook_id_0000000000000000"
        mock_note.title = "Any Note"
        mock_note.body = "content"
        mock_note.id = "12345678901234567890123456789012"
        mock_note.created_time = 1609459200000
        mock_note.updated_time = 1609545600000
        mock_note.is_todo = 0
        mock_note.todo_completed = 0

        mock_client = MagicMock()
        mock_client.get_note.return_value = mock_note

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=mock_client),
        ):
            mock_cfg.has_notebook_whitelist = False
            mock_cfg.notebook_whitelist = None
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import get_note

            fn = _get_tool_fn(get_note)
            result = await fn(note_id="12345678901234567890123456789012")
            assert "Any Note" in result

    @pytest.mark.asyncio
    async def test_create_note_works_without_whitelist(self):
        """create_note succeeds for any notebook when no whitelist is configured."""
        mock_client = MagicMock()
        mock_client.add_note.return_value = "new_note_id"

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=mock_client),
            patch(
                "joplin_mcp.tools.notes.get_notebook_id_by_name",
                return_value="any_nb_id",
            ),
        ):
            mock_cfg.has_notebook_whitelist = False
            mock_cfg.notebook_whitelist = None
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import create_note

            fn = _get_tool_fn(create_note)
            result = await fn(title="Free Note", notebook_name="Anywhere", body="hi")
            assert "SUCCESS" in result

    @pytest.mark.asyncio
    async def test_list_notebooks_returns_all_without_whitelist(self):
        """list_notebooks returns all notebooks when no whitelist is configured."""
        mock_notebooks = [
            MagicMock(id="nb1", title="Work"),
            MagicMock(id="nb2", title="Personal"),
            MagicMock(id="nb3", title="Secret"),
        ]
        mock_client = MagicMock()
        mock_client.get_all_notebooks.return_value = mock_notebooks

        with (
            patch("joplin_mcp.tools.notebooks._module_config") as mock_cfg,
            patch(
                "joplin_mcp.tools.notebooks.get_joplin_client",
                return_value=mock_client,
            ),
            patch("joplin_mcp.tools.notebooks.format_item_list") as mock_format,
        ):
            mock_cfg.has_notebook_whitelist = False
            mock_cfg.notebook_whitelist = None

            mock_format.return_value = "ALL_NOTEBOOKS_LISTED"

            from joplin_mcp.tools.notebooks import list_notebooks

            fn = _get_tool_fn(list_notebooks)
            result = await fn()

            assert result == "ALL_NOTEBOOKS_LISTED"
            # format_item_list should have received all 3 notebooks
            from joplin_mcp.fastmcp_server import ItemType

            mock_format.assert_called_once_with(mock_notebooks, ItemType.notebook)

    @pytest.mark.asyncio
    async def test_delete_note_works_without_whitelist(self):
        """delete_note succeeds for any notebook when no whitelist is configured."""
        mock_client = MagicMock()

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=mock_client),
        ):
            mock_cfg.has_notebook_whitelist = False
            mock_cfg.notebook_whitelist = None
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import delete_note

            fn = _get_tool_fn(delete_note)
            result = await fn(note_id="12345678901234567890123456789012")
            assert "SUCCESS" in result

    def test_config_without_whitelist_property(self):
        """JoplinMCPConfig with no notebook_whitelist has has_notebook_whitelist=False."""
        config = JoplinMCPConfig(token="test_token")
        assert config.has_notebook_whitelist is False
        assert config.notebook_whitelist is None


# ---------------------------------------------------------------------------
# Test 4: Mixed pattern types
# Whitelist with exact path + glob pattern + ID, verify all work together
# ---------------------------------------------------------------------------


class TestMixedPatternTypesIntegration:
    """Integration test: whitelist with IDs, paths, and globs all work together."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    def test_exact_path_and_glob_combined(self, mock_notebook_hierarchy):
        """Whitelist with exact path 'AI' and glob 'Projects/*' both work."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["AI", "Projects/*"]

        # Exact path match: AI
        assert is_notebook_accessible(
            ids["AI"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Glob match: Projects/Work
        assert is_notebook_accessible(
            ids["Work"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Glob match: Projects/Fun
        assert is_notebook_accessible(
            ids["Fun"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Non-matching: Personal
        assert is_notebook_accessible(
            ids["Personal"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is False

        invalidate_notebook_map_cache()
        # Non-matching: Personal/Diary
        assert is_notebook_accessible(
            ids["Diary"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is False

    def test_negation_with_mixed_patterns(self, mock_notebook_hierarchy):
        """Negation pattern excludes specific child from glob match."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects/*", "!Projects/Fun"]

        # Projects/Work should be accessible
        assert is_notebook_accessible(
            ids["Work"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Projects/Fun should be denied by negation
        assert is_notebook_accessible(
            ids["Fun"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is False

    def test_id_based_whitelist_entry(self, mock_notebook_hierarchy):
        """Whitelist with raw notebook ID works alongside path patterns."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        # Mix a raw ID with a path pattern
        whitelist = [ids["Personal"], "AI"]

        # Personal accessible by ID
        assert is_notebook_accessible(
            ids["Personal"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # AI accessible by path
        assert is_notebook_accessible(
            ids["AI"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is True

        invalidate_notebook_map_cache()
        # Projects not accessible (not in whitelist)
        assert is_notebook_accessible(
            ids["Projects"], whitelist_entries=whitelist, client_fn=client_fn,
        ) is False

    @pytest.mark.asyncio
    async def test_find_notes_filters_with_mixed_whitelist(
        self, mock_notebook_hierarchy
    ):
        """find_notes returns only notes in notebooks matching mixed whitelist."""
        ids = mock_notebook_hierarchy["ids"]
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        note_in_work = MagicMock()
        note_in_work.parent_id = ids["Work"]
        note_in_work.id = "work_note_id_00000000000000000000"
        note_in_work.title = "Work Task"
        note_in_work.updated_time = 1609545600000
        note_in_work.is_todo = 0
        note_in_work.todo_completed = 0

        note_in_ai = MagicMock()
        note_in_ai.parent_id = ids["AI"]
        note_in_ai.id = "ai_note_id_0000000000000000000000"
        note_in_ai.title = "AI Research"
        note_in_ai.updated_time = 1609545600000
        note_in_ai.is_todo = 0
        note_in_ai.todo_completed = 0

        note_in_diary = MagicMock()
        note_in_diary.parent_id = ids["Diary"]
        note_in_diary.id = "diary_note_id_000000000000000000"
        note_in_diary.title = "Private Diary Entry"
        note_in_diary.updated_time = 1609459200000
        note_in_diary.is_todo = 0
        note_in_diary.todo_completed = 0

        client.get_all_notes.return_value = [note_in_work, note_in_ai, note_in_diary]

        whitelist = ["AI", "Projects/*"]

        with (
            patch("joplin_mcp.tools.notes._module_config") as mock_cfg,
            patch("joplin_mcp.tools.notes.get_joplin_client", return_value=client),
            patch(
                "joplin_mcp.tools.notes.is_notebook_accessible",
                side_effect=lambda parent_id, whitelist_entries=None, **kw: (
                    is_notebook_accessible(
                        parent_id,
                        whitelist_entries=whitelist_entries,
                        client_fn=client_fn,
                    )
                ),
            ),
        ):
            mock_cfg.has_notebook_whitelist = True
            mock_cfg.notebook_whitelist = whitelist
            mock_cfg.should_show_content.return_value = True
            mock_cfg.should_show_full_content.return_value = True
            mock_cfg.get_max_preview_length.return_value = 300
            mock_cfg.is_smart_toc_enabled.return_value = False
            mock_cfg.get_smart_toc_threshold.return_value = 2000
            mock_cfg.tools = {}

            from joplin_mcp.tools.notes import find_notes

            fn = _get_tool_fn(find_notes)
            result = await fn(query="*", limit=20)

            assert "Work Task" in result
            assert "AI Research" in result
            assert "Private Diary Entry" not in result


# ---------------------------------------------------------------------------
# Test 5: Startup validation integration (D3)
# Server starts with valid/invalid whitelist configurations
# ---------------------------------------------------------------------------


class TestStartupValidationIntegration:
    """Integration test: startup validation with various whitelist configs."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    def test_startup_with_valid_whitelist(self, mock_notebook_hierarchy):
        """Server starts successfully with valid whitelist entries."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)

        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=["Projects", "AI"],
        )

        # Should not raise
        validate_whitelist_at_startup(config, client)

    def test_startup_with_invalid_entries(self, mock_notebook_hierarchy):
        """Server starts successfully even with unresolvable whitelist entries (D3)."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)

        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=["NonExistent/Path", "GhostNotebook"],
        )

        # Per D3: server always starts, never raises
        validate_whitelist_at_startup(config, client)

    def test_startup_with_no_whitelist(self):
        """Server starts successfully with no whitelist configured."""
        client = MagicMock()
        config = JoplinMCPConfig(token="test_token")

        # Should not raise
        validate_whitelist_at_startup(config, client)

    def test_startup_with_empty_whitelist(self, mock_notebook_hierarchy):
        """Server starts successfully with empty whitelist (D3: never crashes)."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)

        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=[],
        )

        # Should not raise, even though empty whitelist means no access
        validate_whitelist_at_startup(config, client)

    def test_startup_with_mixed_valid_invalid(self, mock_notebook_hierarchy):
        """Server starts with mix of valid and invalid entries, logging warnings."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)

        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=["Projects", "NonExistent", "AI"],
        )

        # Should not raise; valid entries still work
        validate_whitelist_at_startup(config, client)

        # Verify accessible notebooks still work after startup
        client_fn = lambda: client  # noqa: E731
        invalidate_notebook_map_cache()
        ids = {
            nb.id: nb.title for nb in notebooks
        }
        projects_id = [
            nb.id for nb in notebooks if nb.title == "Projects"
        ][0]

        assert is_notebook_accessible(
            projects_id,
            whitelist_entries=config.notebook_whitelist,
            client_fn=client_fn,
        ) is True

    def test_startup_auto_creates_default_notebook(self):
        """When whitelist resolves to zero notebooks, auto-create default (D9)."""
        invalidate_notebook_map_cache()

        # Empty notebook list -- nothing matches the whitelist
        client = _make_mock_client([])
        client.add_notebook.return_value = "auto_created_id"

        config = JoplinMCPConfig(
            token="test_token",
            notebook_whitelist=["SomePattern"],
        )

        # Should auto-create "MCP Access" notebook
        validate_whitelist_at_startup(config, client)
        client.add_notebook.assert_called_once_with(title="MCP Access")


# ---------------------------------------------------------------------------
# Test 6: Filter accessible notebooks integration
# filter_accessible_notebooks with real notebook_utils logic
# ---------------------------------------------------------------------------


class TestFilterAccessibleNotebooksIntegration:
    """Integration test: filter_accessible_notebooks with real hierarchy."""

    def setup_method(self):
        invalidate_notebook_map_cache()

    def test_filter_returns_only_accessible(self, mock_notebook_hierarchy):
        """filter_accessible_notebooks returns only whitelisted notebooks."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects", "AI"]

        result = filter_accessible_notebooks(
            notebooks, whitelist_entries=whitelist, client_fn=client_fn,
        )

        result_titles = {nb.title for nb in result}
        # Projects itself, its children (Work, Fun via hierarchical), and AI
        assert "Projects" in result_titles
        assert "Work" in result_titles
        assert "Fun" in result_titles
        assert "AI" in result_titles
        # Personal and Diary should be excluded
        assert "Personal" not in result_titles
        assert "Diary" not in result_titles

    def test_filter_with_none_returns_empty(self):
        """filter_accessible_notebooks returns [] when whitelist is None."""
        notebooks = [SimpleNamespace(id="nb1", title="Work")]
        result = filter_accessible_notebooks(notebooks, whitelist_entries=None)
        assert result == []

    def test_filter_with_glob_pattern(self, mock_notebook_hierarchy):
        """filter_accessible_notebooks works with glob patterns."""
        notebooks = mock_notebook_hierarchy["notebooks"]
        client = _make_mock_client(notebooks)
        client_fn = lambda: client  # noqa: E731

        whitelist = ["Projects/*"]

        result = filter_accessible_notebooks(
            notebooks, whitelist_entries=whitelist, client_fn=client_fn,
        )

        result_titles = {nb.title for nb in result}
        assert "Work" in result_titles
        assert "Fun" in result_titles
        # Projects itself is not matched by "Projects/*" (only children)
        # Personal and Diary should be excluded
        assert "Personal" not in result_titles
        assert "Diary" not in result_titles
