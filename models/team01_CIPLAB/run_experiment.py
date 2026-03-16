from __future__ import annotations

import argparse

try:
    from . import hat_backend
except ImportError:
    import hat_backend


def main(exp_path: str, dry_run: bool = False) -> None:
    hat_backend.run_experiment(exp_path, dry_run=dry_run)


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run the minimal team01_CIPLAB HAT-L experiment JSON."
    )
    parser.add_argument("--exp", required=True, help="Path to the experiment JSON.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print what would run.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    main(cli_args.exp, dry_run=cli_args.dry_run)
