"""Fetch optional upstream workloads outside the tracked source tree."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.manifest import EX4DGS


def checkout_ex4dgs(root: Path) -> None:
    target = root / "third_party" / "Ex4DGS"
    if target.exists():
        print(f"using existing checkout: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", EX4DGS.url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", EX4DGS.commit], check=True)
    print(f"checked out {EX4DGS.name} at {EX4DGS.commit}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch optional official workload source outside Git tracking")
    parser.add_argument("--ex4dgs", action="store_true", help="fetch the pinned Ex4DGS checkout")
    args = parser.parse_args(argv)
    if args.ex4dgs:
        checkout_ex4dgs(ROOT)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
