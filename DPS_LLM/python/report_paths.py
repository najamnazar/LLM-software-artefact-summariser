"""Policy for which report files scripts may write to.

``evaluation-results/results.txt`` is a hand-curated consolidated report: it carries an
author's narrative, section ordering and a closing ``END OF CONSOLIDATED RESULTS`` marker.
It is not a machine log. Several analysis scripts nonetheless defaulted to appending their
output to it, so every run silently grew the curated document and interleaved stale numbers
with current ones — and because the append is silent, the damage was only visible by
noticing the file had grown.

This module is shared rather than copied into each script so the list of protected files
cannot drift between them.
"""

from __future__ import annotations

from pathlib import Path

# Files that are written by hand and must never be appended to automatically.
# summary_length_stats.txt is deliberately NOT listed: its own script rewrites it whole
# each run, so it is generated output rather than a curated document.
CURATED_REPORTS = frozenset({
    'results.txt',
})


def guard_report_target(path: Path, override: bool, override_flag: str = '--allow-curated-append') -> None:
    """Refuse to write to a curated report unless the caller explicitly opted in.

    Raises SystemExit with an explanation rather than an exception trace, because this is
    a usage error the person running the script needs to read and act on.
    """
    if path.name not in CURATED_REPORTS or override:
        return
    raise SystemExit(
        f"Refusing to append to {path}: it is a hand-curated report, not a machine log, "
        f"and appending would interleave this run's numbers with the curated narrative.\n"
        f"  Write to a generated file : --append-to evaluation-results/<name>.txt\n"
        f"  Print without writing     : --append-to ''\n"
        f"  Append anyway             : {override_flag}"
    )
