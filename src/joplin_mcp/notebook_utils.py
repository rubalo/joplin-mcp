"""Notebook utilities for path resolution, caching, and lookup."""

import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import pathspec

logger = logging.getLogger(__name__)


# === NOTEBOOK MAP BUILDING ===


def _build_notebook_map(notebooks: List[Any]) -> Dict[str, Dict[str, Optional[str]]]:
    """Build a map of notebook_id -> {title, parent_id}."""
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for nb in notebooks or []:
        try:
            nb_id = getattr(nb, "id", None)
            if not nb_id:
                continue
            mapping[nb_id] = {
                "title": getattr(nb, "title", "Untitled"),
                "parent_id": getattr(nb, "parent_id", None),
            }
        except Exception:
            # Be resilient to unexpected notebook structures
            continue
    return mapping


def _compute_notebook_path(
    notebook_id: Optional[str],
    notebooks_map: Dict[str, Dict[str, Optional[str]]],
    sep: str = " / ",
) -> Optional[str]:
    """Compute full notebook path from root to the specified notebook.

    Returns a string like "Parent / Child / Notebook" or None if unavailable.
    """
    if not notebook_id:
        return None

    parts: List[str] = []
    seen: set[str] = set()
    curr = notebook_id
    while curr and curr not in seen:
        seen.add(curr)
        info = notebooks_map.get(curr)
        if not info:
            break
        title = (info.get("title") or "Untitled").strip()
        parts.append(title)
        curr = info.get("parent_id")

    if not parts:
        return None
    return sep.join(reversed(parts))


# === NOTEBOOK MAP CACHE ===


_NOTEBOOK_MAP_CACHE: Dict[str, Any] = {"built_at": 0.0, "map": None}
_DEFAULT_NOTEBOOK_TTL_SECONDS = 90  # sensible default; adjustable via env var


def _get_notebook_cache_ttl() -> int:
    try:
        env_val = os.getenv("JOPLIN_MCP_NOTEBOOK_CACHE_TTL")
        if env_val:
            ttl = int(env_val)
            # Clamp to reasonable bounds to avoid accidental huge/small values
            return max(5, min(ttl, 3600))
    except Exception:
        pass
    return _DEFAULT_NOTEBOOK_TTL_SECONDS


def get_notebook_map_cached(
    force_refresh: bool = False,
    client_fn: Optional[Callable] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Return cached notebook map with TTL; refresh if stale or forced.

    Args:
        force_refresh: Force cache refresh regardless of TTL
        client_fn: Optional function returning joplin client (for dependency injection)
    """
    ttl = _get_notebook_cache_ttl()
    now = time.monotonic()

    if not force_refresh:
        built_at = _NOTEBOOK_MAP_CACHE.get("built_at", 0.0) or 0.0
        cached_map = _NOTEBOOK_MAP_CACHE.get("map")
        if cached_map is not None and (now - built_at) < ttl:
            return cached_map

    # Get client - use provided function or import default
    if client_fn is None:
        from joplin_mcp.fastmcp_server import get_joplin_client
        client_fn = get_joplin_client

    client = client_fn()
    fields_list = "id,title,parent_id"
    notebooks = client.get_all_notebooks(fields=fields_list)
    nb_map = _build_notebook_map(notebooks)
    _NOTEBOOK_MAP_CACHE["map"] = nb_map
    _NOTEBOOK_MAP_CACHE["built_at"] = now
    return nb_map


def invalidate_notebook_map_cache() -> None:
    """Invalidate the cached notebook map so next access refreshes it."""
    _NOTEBOOK_MAP_CACHE["built_at"] = 0.0
    _NOTEBOOK_MAP_CACHE["map"] = None
    # Also invalidate the whitelist spec cache since it depends on notebook paths
    _WHITELIST_SPEC_CACHE["built_at"] = 0.0
    _WHITELIST_SPEC_CACHE["spec"] = None
    _WHITELIST_SPEC_CACHE["entries"] = None


# === WHITELIST PATHSPEC MATCHING ===


_WHITELIST_SPEC_CACHE: Dict[str, Any] = {
    "built_at": 0.0,
    "spec": None,
    "entries": None,
}

# Regex for 32-char hex IDs (Joplin notebook/note IDs)
_HEX_ID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def _build_whitelist_spec(
    whitelist_entries: List[str],
) -> pathspec.PathSpec:
    """Build a pathspec.PathSpec from whitelist pattern entries.

    Patterns follow gitignore/gitwildmatch semantics:
    - 'AI' matches only 'AI' exactly
    - 'AI/*' matches direct children of AI
    - 'AI/**' matches all descendants of AI recursively
    - '!Projects/Secret' negates (excludes even if matched by prior pattern)
    - Patterns evaluated in order; last match wins for negation

    Args:
        whitelist_entries: List of pattern strings.

    Returns:
        Compiled PathSpec object.
    """
    return pathspec.PathSpec.from_lines("gitwildmatch", whitelist_entries)


def _get_whitelist_spec(
    whitelist_entries: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Optional[pathspec.PathSpec]:
    """Return cached compiled PathSpec, rebuilding if stale or entries changed.

    Args:
        whitelist_entries: The whitelist patterns to compile.
            If None, returns None (no whitelist configured).
        force_refresh: Force rebuild regardless of TTL.

    Returns:
        Compiled PathSpec or None if no whitelist.
    """
    if whitelist_entries is None:
        return None
    if not whitelist_entries:
        # Empty list = deny all; return an empty spec that matches nothing
        return _build_whitelist_spec([])

    ttl = _get_notebook_cache_ttl()
    now = time.monotonic()

    if not force_refresh:
        cached_spec = _WHITELIST_SPEC_CACHE.get("spec")
        cached_entries = _WHITELIST_SPEC_CACHE.get("entries")
        built_at = _WHITELIST_SPEC_CACHE.get("built_at", 0.0) or 0.0
        if (
            cached_spec is not None
            and cached_entries == whitelist_entries
            and (now - built_at) < ttl
        ):
            return cached_spec

    spec = _build_whitelist_spec(whitelist_entries)
    _WHITELIST_SPEC_CACHE["spec"] = spec
    _WHITELIST_SPEC_CACHE["entries"] = list(whitelist_entries)
    _WHITELIST_SPEC_CACHE["built_at"] = now
    return spec


def _matches_whitelist(
    notebook_path: str,
    notebook_id: str,
    spec: pathspec.PathSpec,
    whitelist_entries: List[str],
) -> bool:
    """Check if a notebook path or ID matches the whitelist spec.

    Matching logic:
    1. Check the full path against the PathSpec (gitignore semantics)
    2. Also check all ancestor prefixes so whitelisting 'Projects' matches
       'Projects/Work/Tasks'
    3. Check the raw notebook_id for literal 32-char hex ID patterns

    Args:
        notebook_path: Full path like 'Projects/Work/Tasks'
        notebook_id: The notebook's ID (32-char hex)
        spec: Compiled PathSpec object
        whitelist_entries: Original whitelist entries (for ID matching)

    Returns:
        True if the notebook is accessible.
    """
    # Check full path
    if spec.match_file(notebook_path):
        return True

    # Check ancestor paths (so whitelisting "Projects" matches "Projects/Work")
    parts = notebook_path.split("/")
    for i in range(1, len(parts)):
        ancestor = "/".join(parts[:i])
        if spec.match_file(ancestor):
            # Ancestor matched, but check if a later negation excludes the full path
            # We need to verify the full path is not negated
            # Since pathspec handles negation for the full path already,
            # and we're checking ancestors, we need to ensure no negation
            # pattern targets the full path specifically
            if not _has_negation_for_path(notebook_path, whitelist_entries):
                return True

    # Check literal notebook ID (for patterns that are raw 32-char hex IDs)
    if notebook_id in whitelist_entries:
        return True

    return False


def _has_negation_for_path(path: str, whitelist_entries: List[str]) -> bool:
    """Check if any negation pattern in the whitelist specifically targets this path.

    Uses last-match-wins semantics: evaluates all patterns in order,
    tracking whether the path is included or excluded.

    Args:
        path: The full notebook path to check.
        whitelist_entries: The whitelist pattern list.

    Returns:
        True if the path is negated (excluded) by the patterns.
    """
    # Build a spec from just the negation patterns applied to this specific path
    # We replay all patterns in order to determine the final state
    included = False
    for entry in whitelist_entries:
        if entry.startswith("!"):
            # Negation pattern
            neg_pattern = entry[1:]
            neg_spec = pathspec.PathSpec.from_lines("gitwildmatch", [neg_pattern])
            if neg_spec.match_file(path):
                included = False  # Negated
        else:
            pos_spec = pathspec.PathSpec.from_lines("gitwildmatch", [entry])
            if pos_spec.match_file(path):
                included = True  # Included
            # Also check if this is an ancestor match
            parts = path.split("/")
            for i in range(1, len(parts)):
                ancestor = "/".join(parts[:i])
                if pos_spec.match_file(ancestor):
                    included = True
                    break

    return not included


def is_notebook_accessible(
    notebook_id: str,
    whitelist_entries: Optional[List[str]] = None,
    force_refresh: bool = False,
    client_fn: Optional[Callable] = None,
) -> bool:
    """Check if a notebook is accessible under the current whitelist.

    Args:
        notebook_id: The notebook ID to check.
        whitelist_entries: Whitelist patterns. None = deny by default when called
            without config context. Pass the config's notebook_whitelist here.
        force_refresh: Force cache refresh.
        client_fn: Optional client factory for dependency injection.

    Returns:
        True if the notebook is accessible, False otherwise.
    """
    # If whitelist_entries is None, deny by default (caller must explicitly
    # pass the config's whitelist; None means "not configured as a list")
    if whitelist_entries is None:
        return False

    # Empty list = deny all
    if not whitelist_entries:
        return False

    # Get the compiled pathspec
    spec = _get_whitelist_spec(whitelist_entries, force_refresh=force_refresh)
    if spec is None:
        return False

    # Get the notebook map to compute the path
    nb_map = get_notebook_map_cached(
        force_refresh=force_refresh, client_fn=client_fn
    )

    if notebook_id not in nb_map:
        logger.debug(
            "Notebook ID not found in map for whitelist check: %s", notebook_id
        )
        return False

    # Compute the full path using "/" separator for pathspec matching
    notebook_path = _compute_notebook_path(notebook_id, nb_map, sep="/")
    if not notebook_path:
        return False

    return _matches_whitelist(notebook_path, notebook_id, spec, whitelist_entries)


def validate_notebook_access(
    notebook_id: str,
    whitelist_entries: Optional[List[str]] = None,
    force_refresh: bool = False,
    client_fn: Optional[Callable] = None,
) -> None:
    """Validate that a notebook is accessible, raising ValueError if denied.

    Error messages are intentionally generic to avoid revealing notebook
    details (per D7).

    Args:
        notebook_id: The notebook ID to validate.
        whitelist_entries: Whitelist patterns from config.
        force_refresh: Force cache refresh.
        client_fn: Optional client factory.

    Raises:
        ValueError: If the notebook is not accessible.
    """
    if not is_notebook_accessible(
        notebook_id,
        whitelist_entries=whitelist_entries,
        force_refresh=force_refresh,
        client_fn=client_fn,
    ):
        raise ValueError("Notebook not accessible")


def filter_accessible_notebooks(
    notebooks: List[Any],
    whitelist_entries: Optional[List[str]] = None,
    client_fn: Optional[Callable] = None,
) -> List[Any]:
    """Filter a list of notebooks to only those accessible under the whitelist.

    Args:
        notebooks: List of notebook objects (must have .id attribute or 'id' key).
        whitelist_entries: Whitelist patterns from config. If None, returns
            empty list (deny by default).
        client_fn: Optional client factory.

    Returns:
        Filtered list of accessible notebooks.
    """
    if whitelist_entries is None:
        return []
    if not whitelist_entries:
        return []

    result = []
    for nb in notebooks:
        nb_id = getattr(nb, "id", None) or (
            nb.get("id") if isinstance(nb, dict) else None
        )
        if nb_id and is_notebook_accessible(
            nb_id, whitelist_entries=whitelist_entries, client_fn=client_fn
        ):
            result.append(nb)
    return result


# === NOTEBOOK PATH RESOLUTION ===


def _find_notebook_suggestions(
    search_term: str,
    notebooks_map: Dict[str, Dict[str, Optional[str]]],
    limit: int = 5,
) -> List[str]:
    """Find notebook paths containing search_term (case-insensitive).

    Args:
        search_term: Term to search for in notebook titles
        notebooks_map: Map of notebook_id -> {title, parent_id}
        limit: Maximum number of suggestions to return

    Returns:
        List of full notebook paths containing the search term
    """
    search_lower = search_term.lower()
    matching_paths = []

    for nb_id, info in notebooks_map.items():
        title = info.get("title", "")
        if search_lower in title.lower():
            full_path = _compute_notebook_path(nb_id, notebooks_map, sep="/")
            if full_path:
                # Sort key: exact match first, then by path length (shorter = more relevant)
                is_exact = title.lower() == search_lower
                matching_paths.append((not is_exact, len(full_path), full_path))

    # Sort by (not_exact, length) and return just the paths
    matching_paths.sort()
    return [path for _, _, path in matching_paths[:limit]]


def _resolve_notebook_by_path(path: str) -> str:
    """Resolve notebook ID from path like 'Parent/Child/Notebook'.

    Args:
        path: Notebook path with '/' separators (e.g., 'Projects/Work/Tasks')

    Returns:
        str: The notebook ID of the final path component

    Raises:
        ValueError: If path is empty or any component not found/ambiguous
    """
    parts = [p.strip() for p in path.split("/") if p.strip()]
    if not parts:
        raise ValueError("Empty notebook path")

    notebooks_map = get_notebook_map_cached(force_refresh=True)

    current_parent: Optional[str] = None
    for part in parts:
        matches = [
            nb_id for nb_id, info in notebooks_map.items()
            if info["title"].lower() == part.lower()
            and (info.get("parent_id") or None) == current_parent
        ]
        if not matches:
            # Provide suggestions for the missing component
            suggestions = _find_notebook_suggestions(part, notebooks_map)
            if suggestions:
                suggestion_str = ", ".join(f"'{s}'" for s in suggestions)
                raise ValueError(
                    f"Notebook '{part}' not found in path '{path}'. "
                    f"Did you mean: {suggestion_str}?"
                )
            raise ValueError(f"Notebook '{part}' not found in path '{path}'")
        if len(matches) > 1:
            raise ValueError(f"Multiple notebooks named '{part}' in path '{path}'")
        current_parent = matches[0]

    return current_parent


def get_notebook_id_by_name(name: str) -> str:
    """Get notebook ID by name or path with helpful error messages.

    Args:
        name: Notebook name or path (e.g., 'Work' or 'Projects/Work/Tasks')

    Returns:
        str: The notebook ID

    Raises:
        ValueError: If notebook not found or multiple matches
    """
    # If path contains '/', resolve by path
    if "/" in name:
        return _resolve_notebook_by_path(name)

    # Otherwise, use flat name matching via generic helper
    from joplin_mcp.fastmcp_server import _get_item_id_by_name, get_joplin_client

    client = get_joplin_client()
    return _get_item_id_by_name(
        name=name,
        item_type="notebook",
        fetch_fn=client.get_all_notebooks,
        fields="id,title,created_time,updated_time,parent_id",
    )
