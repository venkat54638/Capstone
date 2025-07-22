from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def generate_evidently_report(reference_data, current_data, report_path='drift_reports/evidently_report.html'):
    """Create and save an Evidently drift report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(report_path)
    print(f"Evidently report saved to {report_path}")
