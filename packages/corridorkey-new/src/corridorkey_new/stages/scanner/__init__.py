"""Scanner stage (stage 0).

Public API::

    from corridorkey_new.stages.scanner import scan, Clip, ScanResult, SkippedPath
"""

from corridorkey_new.stages.scanner.contracts import Clip, ScanResult, SkippedPath
from corridorkey_new.stages.scanner.orchestrator import scan

__all__ = ["scan", "Clip", "ScanResult", "SkippedPath"]
