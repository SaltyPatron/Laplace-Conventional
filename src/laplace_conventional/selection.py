from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .corpus import ManifestEntry, write_manifest
from .providers import TEXT, provider_for


@dataclass(frozen=True)
class SelectionDecision:
    path: str
    selected: bool
    reason: str
    matched_rule: str | None


def _rule_value(rule: dict, key: str, default=None):
    return rule[key] if key in rule else default


def decide(path: str, rules: list[dict], *, default_selected: bool = True) -> SelectionDecision:
    selected = bool(default_selected)
    reason = "default selected" if selected else "default excluded"
    matched: str | None = None
    for index, rule in enumerate(rules):
        pattern = str(_rule_value(rule, "glob", ""))
        if not pattern:
            raise ValueError(f"selection rule {index} has no glob")
        if fnmatch.fnmatchcase(path, pattern):
            selected = bool(_rule_value(rule, "selected", True))
            explicit_reason = str(_rule_value(rule, "reason", "")).strip()
            if not selected and not explicit_reason:
                raise ValueError(f"selection exclusion {pattern!r} requires a reason")
            reason = explicit_reason or f"selected by {pattern}"
            matched = pattern
    return SelectionDecision(path, selected, reason, matched)


def apply_selection(
    entries: Iterable[ManifestEntry],
    rules: list[dict],
    *,
    default_selected: bool = True,
    enabled_providers: set[str] | None = None,
) -> tuple[list[ManifestEntry], list[SelectionDecision], dict]:
    physical = list(entries)
    decisions = [decide(e.path, rules, default_selected=default_selected) for e in physical]
    decision_by_path = {d.path: d for d in decisions}
    enabled = set(enabled_providers) if enabled_providers is not None else None

    # Dedupe is resolved *after* selection. This prevents an excluded copy from
    # becoming the canonical duplicate target for a selected copy.
    selected_entries: list[ManifestEntry] = []
    first_selected_hash: dict[str, str] = {}
    for entry in physical:
        if not decision_by_path[entry.path].selected:
            continue
        duplicate_of = None
        if entry.sha256 is not None:
            duplicate_of = first_selected_hash.get(entry.sha256)
            if duplicate_of is None:
                first_selected_hash[entry.sha256] = entry.path
        selected_entries.append(replace(entry, duplicate_of=duplicate_of))

    unique_selected = [e for e in selected_entries if e.duplicate_of is None]

    def active_provider(entry: ManifestEntry):
        provider = provider_for(entry)
        if provider is None:
            return None
        if provider.name == TEXT.name:
            return provider
        if enabled is not None and provider.name not in enabled:
            return None
        return provider

    unsupported = sum(e.size for e in unique_selected if e.accessible and active_provider(e) is None)
    inaccessible = sum(e.size for e in unique_selected if not e.accessible)
    selected_bytes = sum(e.size for e in unique_selected)

    reason_counts: dict[str, dict[str, int]] = {}
    for entry in physical:
        decision = decision_by_path[entry.path]
        if decision.selected:
            continue
        bucket = reason_counts.setdefault(decision.reason, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry.size

    providers: dict[str, dict[str, int | str]] = {}
    for entry in unique_selected:
        provider = active_provider(entry)
        name = provider.name if provider else "unsupported"
        bucket = providers.setdefault(name, {"files": 0, "bytes": 0})
        bucket["files"] = int(bucket["files"]) + 1
        bucket["bytes"] = int(bucket["bytes"]) + entry.size
        if provider:
            bucket["modality"] = provider.modality
            bucket["objective"] = provider.objective
            bucket["model_family"] = provider.model_family

    summary = {
        "physical_files": len(physical),
        "physical_bytes": sum(e.size for e in physical),
        "selected_files_before_dedupe": len(selected_entries),
        "selected_unique_files": len(unique_selected),
        "selected_unique_bytes": selected_bytes,
        "selected_duplicate_files": sum(1 for e in selected_entries if e.duplicate_of is not None),
        "excluded_files": sum(1 for d in decisions if not d.selected),
        "excluded_bytes": sum(e.size for e in physical if not decision_by_path[e.path].selected),
        "trainable_selected_bytes": sum(e.size for e in unique_selected if active_provider(e) is not None),
        "unsupported_selected_bytes": unsupported,
        "inaccessible_selected_files": sum(1 for e in unique_selected if not e.accessible),
        "inaccessible_selected_bytes": inaccessible,
        "coverage_complete": unsupported == 0 and inaccessible == 0,
        "providers": providers,
        "exclusion_reasons": reason_counts,
    }
    return selected_entries, decisions, summary


def write_selection(
    selected_entries: list[ManifestEntry],
    decisions: list[SelectionDecision],
    summary: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True) + "\n")
    write_manifest(selected_entries, summary, out_dir)
