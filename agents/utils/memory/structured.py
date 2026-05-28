"""
StructuredMemory: YAML-based entries with metadata.

Each memory entry uses name as primary key (skills-style).
Agents add new entries and delete stale ones instead of rewriting everything.

Storage: {memory_dir}/memory/{YYYY-MM-DD}.yaml
Backward compatible: falls back to loading .txt files from BasicMemory.
"""

from pathlib import Path
from datetime import date
from dataclasses import dataclass, asdict
from typing import Optional, List
import re

import yaml


def _strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


FIELD_LIMITS = {"name": 64, "description": 256, "content": 1024}
MAX_ENTRIES = 500
RESERVED_WORDS = {"anthropic", "claude"}


@dataclass
class MemoryEntry:
    """A single structured memory entry (skills-style, name is primary key)."""
    name: str          # primary key: lowercase, hyphens, numbers only
    description: str
    content: str
    added: str  # ISO date string


class StructuredMemory:
    """
    YAML-based structured memory for forecasting agents.

    Each memory entry uses name as primary key (skills-style).
    Agents add new entries and delete stale ones instead of rewriting everything.

    Storage: {memory_dir}/memory/{YYYY-MM-DD}.yaml
    Backward compatible: falls back to loading .txt files from BasicMemory.
    """

    def __init__(self, agent_id: str, memory_dir: Optional[str] = None,
                 max_entries: int = None, field_limits: dict = None):
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._entries: List[MemoryEntry] = []
        self._current_date: Optional[date] = None
        self._max_entries = max_entries if max_entries is not None else MAX_ENTRIES
        self._field_limits = field_limits if field_limits is not None else FIELD_LIMITS

        if memory_dir:
            self._memory_dir = Path(memory_dir) / "memory"

    def set_date(self, current_date: date) -> None:
        """Load the most recent memory snapshot before current_date."""
        self._current_date = current_date
        self._entries = []

        if not self._memory_dir or not self._memory_dir.exists():
            return

        # Try .yaml files first, then fall back to .txt
        most_recent_yaml = self._find_most_recent(current_date, "*.yaml")
        if most_recent_yaml:
            self._entries = self._load_yaml(most_recent_yaml)
            return

        most_recent_txt = self._find_most_recent(current_date, "*.txt")
        if most_recent_txt:
            txt_content = most_recent_txt.read_text(encoding='utf-8').strip()
            file_date = date.fromisoformat(most_recent_txt.stem)
            self._entries = self._migrate_txt(txt_content, str(file_date))

    def _find_most_recent(self, current_date: date, glob_pattern: str) -> Optional[Path]:
        """Find the most recent file matching glob_pattern with date < current_date."""
        files = sorted(self._memory_dir.glob(glob_pattern))
        most_recent = None
        for f in files:
            try:
                file_date = date.fromisoformat(f.stem)
                if file_date < current_date:
                    most_recent = f
            except ValueError:
                continue
        return most_recent

    def get(self) -> str:
        """Render entries into compact text for prompt injection."""
        if not self._entries:
            return ""
        lines = []
        for e in self._entries:
            lines.append(f"[{e.name}] {e.description} ({e.added})")
            if e.content:
                lines.append(f"  {e.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_index(self) -> str:
        """Render entry index (name + description) for prompt injection. No content."""
        if not self._entries:
            return ""
        lines = []
        for e in self._entries:
            desc = e.description or "(no description)"
            lines.append(f"[{e.name}] {desc} ({e.added})")
        return "\n".join(lines)

    def retrieve(self, entry_name: str) -> Optional[str]:
        """Retrieve full content of a single entry by name. Returns None if not found."""
        entry_name = self._normalize_lookup_name(entry_name)
        for e in self._entries:
            if e.name == entry_name:
                lines = [
                    f"[{e.name}]",
                    f"Description: {e.description}",
                    f"Content: {e.content}",
                    f"Added: {e.added}",
                ]
                return "\n".join(lines)
        return None

    def _normalize_lookup_name(self, raw: str) -> str:
        """Normalize a name for lookup (strip, lowercase, collapse whitespace to hyphens)."""
        return raw.strip().lower().replace(" ", "-")

    def add_entry(self, name: str, description: str, content: str) -> str:
        """
        Add a new memory entry. Returns the validated name on success,
        or raises ValueError on duplicate/invalid name.

        Name is auto-normalized (lowercase, hyphens). Fields are truncated.
        If entry count exceeds max_entries, the oldest entry is dropped.
        """
        name = self._validate_name(name)
        description = _strip_xml_tags(description).strip()[:self._field_limits["description"]]
        content = _strip_xml_tags(content).strip()[:self._field_limits["content"]]

        if not description:
            description = name  # Fallback: use name as description

        # Reject duplicates
        if any(e.name == name for e in self._entries):
            raise ValueError(f"Entry '{name}' already exists — use update_entry to modify it")

        entry = MemoryEntry(
            name=name,
            description=description,
            content=content,
            added=str(self._current_date) if self._current_date else "",
        )
        self._entries.append(entry)

        # Enforce max entries (drop oldest first)
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

        self._save(self._current_date)
        return name

    def update_entry(self, entry_name: str, *,
                     description: Optional[str] = None,
                     content: Optional[str] = None) -> bool:
        """Partial update of an existing entry by name. Returns True if found."""
        entry_name = self._normalize_lookup_name(entry_name)
        for e in self._entries:
            if e.name == entry_name:
                if description is not None:
                    e.description = _strip_xml_tags(description).strip()[:self._field_limits["description"]]
                if content is not None:
                    e.content = _strip_xml_tags(content).strip()[:self._field_limits["content"]]
                self._save(self._current_date)
                return True
        return False

    def delete_entry(self, entry_name: str) -> bool:
        """Delete an entry by name. Returns True if found and deleted."""
        entry_name = self._normalize_lookup_name(entry_name)
        for i, e in enumerate(self._entries):
            if e.name == entry_name:
                self._entries.pop(i)
                self._save(self._current_date)
                return True
        return False

    def update(self, new_memory: str, save_date: Optional[date] = None) -> None:
        """
        Backward-compat full replacement.

        If new_memory looks like YAML (list of dicts), parse it.
        Otherwise treat as plain text and create a single migrated entry.
        """
        new_memory = new_memory.strip()
        if not new_memory:
            self._entries = []
            self._save(save_date or self._current_date)
            return

        # Try YAML parse first
        try:
            data = yaml.safe_load(new_memory)
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                self._entries = self._parse_entry_list(data)
                self._save(save_date or self._current_date)
                return
        except yaml.YAMLError:
            pass

        # Fallback: treat as plain text migration
        added = str(save_date or self._current_date or "")
        self._entries = self._migrate_txt(new_memory, added)
        self._save(save_date or self._current_date)

    def _save(self, save_date: Optional[date]) -> None:
        """Persist entries as YAML to disk."""
        if not self._memory_dir or not save_date:
            return
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        path = self._memory_dir / f"{save_date}.yaml"
        data = [asdict(e) for e in self._entries]
        path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True), encoding='utf-8')

    def _load_yaml(self, path: Path) -> List[MemoryEntry]:
        """Load entries from a YAML file."""
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                return []
            return self._parse_entry_list(data)
        except (yaml.YAMLError, Exception):
            return []

    def _parse_entry_list(self, data: list) -> List[MemoryEntry]:
        """Parse a list of dicts into MemoryEntry objects.

        Backward compatible: old YAML files with ``id``/``type``/``qids`` fields
        are handled (id is ignored, type/qids folded into description).
        """
        entries = []
        seen_names: set = set()
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                continue
            try:
                raw_name = str(d.get("name", ""))
                try:
                    name = self._validate_name(raw_name) if raw_name else f"migrated-entry-{i}"
                except ValueError:
                    name = f"migrated-entry-{i}"

                # Dedup: append suffix if name collision
                base_name = name
                suffix = 2
                while name in seen_names:
                    name = f"{base_name}-{suffix}"[:self._field_limits["name"]]
                    suffix += 1
                seen_names.add(name)

                # Backward compat: generate description from old type/qids if missing
                description = d.get("description")
                if description is None:
                    parts = []
                    old_type = d.get("type", "")
                    old_qids = d.get("qids", "")
                    if old_type:
                        parts.append(f"[{old_type}]")
                    if old_qids:
                        parts.append(f"({old_qids})")
                    content_preview = str(d.get("content", ""))[:150]
                    if content_preview:
                        parts.append(content_preview)
                    description = " ".join(parts) if parts else name
                description = str(description)[:self._field_limits["description"]]

                entries.append(MemoryEntry(
                    name=name,
                    description=description,
                    content=str(d.get("content", ""))[:self._field_limits["content"]],
                    added=str(d.get("added", "")),
                ))
            except Exception:
                continue
        return entries

    def _migrate_txt(self, txt_content: str, added_date: str) -> List[MemoryEntry]:
        """Convert plain text memory into structured entries (one per paragraph)."""
        if not txt_content:
            return []

        paragraphs = [p.strip() for p in txt_content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [txt_content.strip()]

        entries = []
        seen_names: set = set()
        for i, para in enumerate(paragraphs):
            # Use first line (up to limit) as name, rest as content
            lines = para.split("\n", 1)
            raw_name = lines[0].strip()
            content = (lines[1].strip() if len(lines) > 1 else para.strip())[:self._field_limits["content"]]
            try:
                name = self._validate_name(raw_name) if raw_name else f"migrated-entry-{i+1}"
            except ValueError:
                name = f"migrated-entry-{i+1}"

            # Dedup
            base_name = name
            suffix = 2
            while name in seen_names:
                name = f"{base_name}-{suffix}"[:self._field_limits["name"]]
                suffix += 1
            seen_names.add(name)

            description = content[:self._field_limits["description"]]

            entries.append(MemoryEntry(
                name=name,
                description=description,
                content=content,
                added=added_date,
            ))

        return entries[:self._max_entries]

    def _validate_name(self, raw: str) -> str:
        """Normalize and validate a memory entry name (skills-style).

        Auto-normalizes: strip XML tags, lowercase, replace spaces/underscores
        with hyphens, remove invalid chars, collapse multiple hyphens,
        strip leading/trailing hyphens, truncate to limit.

        Raises ValueError if result is empty or contains reserved words.
        """
        name = _strip_xml_tags(raw).strip().lower()
        name = re.sub(r'[\s_]+', '-', name)           # spaces/underscores -> hyphens
        name = re.sub(r'[^a-z0-9-]', '', name)        # strip invalid chars
        name = re.sub(r'-{2,}', '-', name)            # collapse multiple hyphens
        name = name.strip('-')                         # no leading/trailing hyphens
        name = name[:self._field_limits["name"]]

        if not name:
            raise ValueError("Name is empty after normalization")
        for word in RESERVED_WORDS:
            if word in name:
                raise ValueError(f"Name cannot contain reserved word '{word}'")
        return name

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        """Total rendered character count."""
        return len(self.get())

    @property
    def entry_count(self) -> int:
        return len(self._entries)
