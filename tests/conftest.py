"""Truly silent on pass: no dots, no progress bar, no summary.
On failure: full hex dump traceback.

Usage: pytest tests/test_all.py -q --tb=short --no-header
"""


def pytest_report_teststatus(report, config):
    """Suppress dots for passed tests."""
    if report.when == "call":
        if report.passed:
            return "passed", "", ""
        if report.failed:
            return "failed", "F", ""


def pytest_sessionstart(session):
    """Override terminal reporter methods after it is fully configured."""
    tr = session.config.pluginmanager.getplugin("terminalreporter")
    if tr is None:
        return

    # Suppress progress bar
    if hasattr(tr, '_write_progress_information_filling_space'):
        tr._write_progress_information_filling_space = lambda: None

    # Suppress summary line on pass (keep on failure)
    if hasattr(tr, 'summary_stats'):
        orig = tr.summary_stats
        def _summary_stats():
            if tr.stats.get("failed"):
                orig()
        tr.summary_stats = _summary_stats
