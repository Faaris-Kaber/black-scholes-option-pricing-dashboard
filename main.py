"""Convenience launcher: ``python main.py`` starts the Streamlit dashboard."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Run the dashboard with this Python environment and forward CLI options."""
    return subprocess.call(
        [
            sys.executable, "-X", "utf8", "-m", "streamlit", "run",
            str(Path(__file__).with_name("app.py")), *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
