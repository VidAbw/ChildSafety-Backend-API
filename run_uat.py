"""
Automated UAT Suite Execution Script with HTML/Console Report Generation
Usage:
    python run_uat.py
"""
import sys
import subprocess
from pathlib import Path


def main():
    print("=" * 70)
    print("  CHILD SAFETY GUARDIAN - AUTOMATED USER ACCEPTANCE TESTING (UAT)")
    print("=" * 70)
    
    report_path = Path("evaluation_results/uat_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/uat",
        "-v",
        f"--html={report_path}",
        "--self-contained-html",
    ]

    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print(" [SUCCESS] All User Acceptance Testing scenarios PASSED!")
        print(f" View detailed HTML Acceptance Report at: {report_path.resolve()}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(" [FAILURE] Some UAT Scenarios failed. Please inspect report.")
        print("=" * 70)
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
