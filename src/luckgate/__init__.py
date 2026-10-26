"""luckgate: did the winner of a search beat luck?

Selection-bias statistics for the best of many trials, and a pass/fail
checklist that uses them.
"""

from luckgate.checklist import (
    Checklist,
    CheckResult,
    Evidence,
    GateConfig,
    Verdict,
)
from luckgate.pbo import PBOResult, cscv_pbo
from luckgate.sharpe import (
    DSRReport,
    SharpeStats,
    dsr,
    dsr_report,
    dsr_report_from_stats,
    expected_max_sharpe,
    min_track_record_length,
    psr,
    sharpe_ratio,
    sharpe_se,
    sharpe_stats,
    trial_sharpe_std,
)

__version__ = "0.1.0"

__all__ = [
    "CheckResult",
    "Checklist",
    "DSRReport",
    "Evidence",
    "GateConfig",
    "PBOResult",
    "SharpeStats",
    "Verdict",
    "__version__",
    "cscv_pbo",
    "dsr",
    "dsr_report",
    "dsr_report_from_stats",
    "expected_max_sharpe",
    "min_track_record_length",
    "psr",
    "sharpe_ratio",
    "sharpe_se",
    "sharpe_stats",
    "trial_sharpe_std",
]
