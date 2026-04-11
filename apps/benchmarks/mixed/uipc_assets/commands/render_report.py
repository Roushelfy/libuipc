from __future__ import annotations

from ..core.report_schema import build_summary_payload, collect_report_data, write_report_files


def run(args) -> int:
    run_root = args.run_root.resolve()
    report_data = collect_report_data(run_root)
    summary = build_summary_payload(report_data)
    write_report_files(run_root, summary)
    print(f"report written: {run_root / 'reports'}")
    return 0

