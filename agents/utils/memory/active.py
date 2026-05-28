"""
ActiveMemory: Question-specific DataFrame (mem_df) + reduced StructuredMemory.

Combined memory: question-specific DataFrame (mem_df) + reduced StructuredMemory.

mem_df: One row per question, keyed by qid. Stored as CSV.
meta-insights: StructuredMemory with reduced limits (15 entries, 400-char content).

Storage (new directory format):
  {memory_dir}/memory/{YYYY-MM-DD}/mem.csv    (question-specific notes)
  {memory_dir}/memory/{YYYY-MM-DD}/meta.yaml  (meta-insights)

Backward compat loading from flat format:
  {memory_dir}/memory/memo_{YYYY-MM-DD}.csv
  {memory_dir}/memory/{YYYY-MM-DD}.yaml
"""

from pathlib import Path
from datetime import date
from typing import Optional
import csv

import pandas as pd
import yaml

from agents.utils.memory.structured import StructuredMemory, MemoryEntry, asdict

ACTIVE_META_MAX_ENTRIES = 15
ACTIVE_META_FIELD_LIMITS = {"name": 64, "description": 256, "content": 400}
ACTIVE_MEM_CHAR_LIMIT = 1000
MEM_DF_COLUMNS = ["qid", "question", "last_updated", "memory", "category"]


class ActiveMemory:
    """
    Combined memory: question-specific DataFrame (mem_df) + reduced StructuredMemory.

    mem_df: One row per question, keyed by qid. Stored as CSV.
    meta-insights: StructuredMemory with reduced limits (15 entries, 400-char content).

    Storage (new directory format):
      {memory_dir}/memory/{YYYY-MM-DD}/mem.csv    (question-specific notes)
      {memory_dir}/memory/{YYYY-MM-DD}/meta.yaml  (meta-insights)

    Backward compat loading from flat format:
      {memory_dir}/memory/memo_{YYYY-MM-DD}.csv
      {memory_dir}/memory/{YYYY-MM-DD}.yaml
    """

    def __init__(self, agent_id: str, memory_dir: Optional[str] = None,
                 max_entries: int = ACTIVE_META_MAX_ENTRIES):
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._mem_df: pd.DataFrame = pd.DataFrame(columns=MEM_DF_COLUMNS)
        self._current_date: Optional[date] = None

        # Meta-insight layer — ephemeral (no independent disk writes).
        # ActiveMemory manages all persistence via save().
        self._meta = StructuredMemory(
            agent_id, memory_dir=None,
            max_entries=max_entries,
            field_limits=ACTIVE_META_FIELD_LIMITS,
        )

        if memory_dir:
            self._memory_dir = Path(memory_dir) / "memory"

    def set_date(self, current_date: date) -> None:
        """Load most recent mem CSV and meta-insight YAML before current_date."""
        self._current_date = current_date
        self._mem_df = pd.DataFrame(columns=MEM_DF_COLUMNS)
        self._meta._current_date = current_date
        self._meta._entries = []

        if not self._memory_dir or not self._memory_dir.exists():
            return

        # Find most recent date directory or flat files before current_date
        most_recent_csv = None
        most_recent_yaml = None

        # Try new directory format: memory/YYYY-MM-DD/
        for d in sorted(self._memory_dir.iterdir()):
            if d.is_dir():
                try:
                    dir_date = date.fromisoformat(d.name)
                    if dir_date < current_date:
                        csv_path = d / "mem.csv"
                        yaml_path = d / "meta.yaml"
                        if csv_path.exists():
                            most_recent_csv = csv_path
                        if yaml_path.exists():
                            most_recent_yaml = yaml_path
                except ValueError:
                    continue

        # Fall back to old flat format
        if most_recent_csv is None:
            most_recent_csv = self._find_most_recent_flat(current_date, "memo_*.csv")
        if most_recent_yaml is None:
            most_recent_yaml = self._find_most_recent_flat(current_date, "*.yaml")

        if most_recent_csv:
            self._mem_df = self._load_csv(most_recent_csv)
        if most_recent_yaml:
            self._meta._entries = self._meta._load_yaml(most_recent_yaml)

    def _find_most_recent_flat(self, current_date: date, glob_pattern: str) -> Optional[Path]:
        """Find the most recent flat file matching glob_pattern with date < current_date."""
        files = sorted(self._memory_dir.glob(glob_pattern))
        most_recent = None
        for f in files:
            if f.is_dir():
                continue
            try:
                stem = f.stem
                date_str = stem.replace("memo_", "") if stem.startswith("memo_") else stem
                file_date = date.fromisoformat(date_str)
                if file_date < current_date:
                    most_recent = f
            except ValueError:
                continue
        return most_recent

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load mem_df from a CSV file. Drops confidence column from old format."""
        try:
            df = pd.read_csv(path, dtype={"qid": str})
            # Drop old confidence column if present
            if "confidence" in df.columns:
                df = df.drop(columns=["confidence"])
            # Ensure all expected columns exist
            for col in MEM_DF_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[MEM_DF_COLUMNS].copy()
        except Exception:
            return pd.DataFrame(columns=MEM_DF_COLUMNS)

    # --- Retrieval ---

    def get(self) -> str:
        """Render meta-insights for prompt injection (delegates to StructuredMemory)."""
        return self._meta.get()

    def get_mem_df(self) -> pd.DataFrame:
        """Return a copy of the question-specific memory DataFrame."""
        return self._mem_df.copy()

    def mem_summary(self, expanded_qids: set = None) -> str:
        """Compact summary of mem_df. Full memory text shown for expanded_qids."""
        if self._mem_df.empty:
            return "(empty)"

        expanded_qids = {str(q) for q in expanded_qids} if expanded_qids else set()
        compact_lines = []
        expanded_lines = []

        for _, row in self._mem_df.iterrows():
            qid = str(row['qid'])
            q_trunc = str(row["question"])[:50]
            if len(str(row["question"])) > 50:
                q_trunc += "..."
            cat = row.get("category", "") or ""

            if qid in expanded_qids:
                expanded_lines.append(
                    f"  [{qid}] {q_trunc} (updated: {row['last_updated']})\n"
                    f"    {row.get('memory', '')}"
                )
            else:
                compact_lines.append(
                    f"  {qid} | {q_trunc} | {row['last_updated']} | {cat}"
                )

        sections = []
        if expanded_lines:
            sections.append(
                "Entries you interacted with today (showing stored memory):\n"
                + "\n".join(expanded_lines)
            )
        if compact_lines:
            header = "  QID | Question | Last Updated | Category"
            sections.append(
                "Other entries:\n" + header + "\n" + "\n".join(compact_lines)
            )

        return "\n\n".join(sections) if sections else "(empty)"

    # --- Mutation ---

    def mem_add(self, qid: str, question: str, memory: str,
                category: str = "") -> None:
        """Add or upsert a question-specific memory entry."""
        qid = str(qid).strip()
        memory = memory.strip()[:ACTIVE_MEM_CHAR_LIMIT]
        question = question.strip()
        category = (category or "").strip()
        updated = str(self._current_date) if self._current_date else ""

        # Preserve existing category when caller doesn't provide one
        if not category:
            old = self._mem_df.loc[self._mem_df["qid"] == qid, "category"]
            if not old.empty and old.iloc[0]:
                category = str(old.iloc[0]).strip()

        # Upsert: remove existing row for this qid
        self._mem_df = self._mem_df[self._mem_df["qid"] != qid]

        new_row = pd.DataFrame([{
            "qid": qid,
            "question": question,
            "last_updated": updated,
            "memory": memory,
            "category": category,
        }])
        self._mem_df = pd.concat([self._mem_df, new_row], ignore_index=True)

    def mem_update(self, qid: str, memory: str,
                   category: str = None) -> None:
        """Update an existing mem_df entry. If qid not found, treated as add with blank question."""
        qid = str(qid).strip()
        mask = self._mem_df["qid"] == qid
        if mask.any():
            self._mem_df.loc[mask, "memory"] = memory.strip()[:ACTIVE_MEM_CHAR_LIMIT]
            self._mem_df.loc[mask, "last_updated"] = str(self._current_date) if self._current_date else ""
            if category is not None and category.strip():
                self._mem_df.loc[mask, "category"] = category.strip()
        else:
            self.mem_add(qid, "", memory, category or "")

    def mem_delete(self, qid: str) -> bool:
        """Delete a mem_df entry by qid. Returns True if found and deleted."""
        qid = str(qid).strip()
        before = len(self._mem_df)
        self._mem_df = self._mem_df[self._mem_df["qid"] != qid]
        return len(self._mem_df) < before

    # --- Meta-insight delegation ---

    def add_entry(self, name: str, description: str, content: str) -> str:
        """Add a meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.add_entry(name, description, content)

    def delete_entry(self, entry_name: str) -> bool:
        """Delete a meta-insight entry by name."""
        return self._meta.delete_entry(entry_name)

    def update_entry(self, entry_name: str, **kwargs) -> bool:
        """Update a meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.update_entry(entry_name, **kwargs)

    def get_index(self) -> str:
        """Render meta-insight index (delegates to StructuredMemory)."""
        return self._meta.get_index()

    def retrieve(self, entry_name: str) -> Optional[str]:
        """Retrieve full meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.retrieve(entry_name)

    @property
    def _max_entries(self) -> int:
        """Max meta-insight entries (for _handle_memory_action compat)."""
        return self._meta._max_entries

    @property
    def entry_count(self) -> int:
        """Number of meta-insight entries."""
        return self._meta.entry_count

    @property
    def mem_count(self) -> int:
        """Number of question-specific mem entries."""
        return len(self._mem_df)

    # --- Persistence ---

    def save(self, save_date: Optional[date] = None) -> None:
        """Write both mem CSV and meta-insight YAML to per-day directory."""
        save_date = save_date or self._current_date
        if not self._memory_dir or not save_date:
            return

        date_dir = self._memory_dir / str(save_date)
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save mem_df as CSV
        csv_path = date_dir / "mem.csv"
        self._mem_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)

        # Save meta-insights YAML
        yaml_path = date_dir / "meta.yaml"
        data = [asdict(e) for e in self._meta._entries]
        yaml_path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True), encoding='utf-8')

    def _save(self, save_date: Optional[date] = None) -> None:
        """Alias for save(), for compatibility with _run_memory_update_loop."""
        self.save(save_date)

    def __bool__(self) -> bool:
        return not self._mem_df.empty or bool(self._meta)

    def __len__(self) -> int:
        """Total rendered character count of meta-insights."""
        return len(self._meta)
