"""Utility launcher that prepares a dedicated virtual environment and runs fcpm_py.py."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[launcher] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _venv_bin_path(venv_path: Path, executable: str) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return venv_path / scripts_dir / executable


def _ensure_venv(venv_path: Path) -> None:
    if venv_path.exists():
        return
    print(f"[launcher] Creating virtual environment at {venv_path}")
    _run([sys.executable, "-m", "venv", str(venv_path)])


def _requirements_fingerprint(requirements_file: Path) -> str:
    return hashlib.sha256(requirements_file.read_bytes()).hexdigest()


def _install_requirements(venv_path: Path, requirements_file: Path, force: bool) -> None:
    pip_path = _venv_bin_path(venv_path, "pip.exe" if os.name == "nt" else "pip")
    marker_file = venv_path / ".fcmpy-requirements.sha256"
    required_hash = _requirements_fingerprint(requirements_file)

    if not force and marker_file.exists() and marker_file.read_text().strip() == required_hash:
        print("[launcher] Requirements already satisfied — skipping install")
        return

    print("[launcher] Upgrading pip/setuptools/wheel")
    _run([str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"])
    print("[launcher] Installing base dependencies")
    _run([str(pip_path), "install", "-r", str(requirements_file)])
    marker_file.write_text(required_hash)


def _install_editable_package(venv_path: Path, extras: Iterable[str]) -> None:
    pip_path = _venv_bin_path(venv_path, "pip.exe" if os.name == "nt" else "pip")
    extras_str = f"[{','.join(extras)}]" if extras else ""
    target = f".{extras_str}"
    print(f"[launcher] Installing project in editable mode: -e {target}")
    _run([str(pip_path), "install", "-e", target])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a venv, install deps, and run fcpm_py.py")
    parser.add_argument("--venv", default=".venv", help="Directory that will host the virtual environment")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to the base requirements file")
    parser.add_argument("--skip-run", action="store_true", help="Prepare the environment but do not run fcpm_py.py")
    parser.add_argument("--force-reinstall", action="store_true", help="Reinstall requirements even if unchanged")
    parser.add_argument("--no-editable", action="store_true", help="Skip installing the package in editable mode")
    parser.add_argument(
        "--extras",
        nargs="*",
        default=(),
        help="Optional extras to install together with the editable package (e.g., ml ml-tf viz)",
    )
    parser.add_argument(
        "fcpm_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to fcpm_py.py. Use -- to separate launcher args from script args.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    venv_path = (repo_root / args.venv).resolve()
    requirements_file = (repo_root / args.requirements).resolve()

    if not requirements_file.exists():
        raise FileNotFoundError(f"Unable to locate requirements file at {requirements_file}")

    _ensure_venv(venv_path)
    _install_requirements(venv_path, requirements_file, force=args.force_reinstall)

    if args.extras and args.no_editable:
        extras_list = ", ".join(args.extras)
        raise ValueError(
            f"Extras ({extras_list}) were provided but --no-editable was set. Either remove --no-editable or "
            "omit extras."
        )

    if not args.no_editable:
        _install_editable_package(venv_path, args.extras)

    if args.skip_run:
        print("[launcher] Environment ready. Skipping fcpm_py.py execution as requested.")
        return

    python_path = _venv_bin_path(venv_path, "python.exe" if os.name == "nt" else "python")
    fcpm_script = (repo_root / "fcpm_py.py").resolve()
    if not fcpm_script.exists():
        raise FileNotFoundError(f"Could not find fcpm_py.py at {fcpm_script}")

    forwarded_args = list(args.fcpm_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    cmd = [str(python_path), str(fcpm_script)] + forwarded_args
    _run(cmd)


if __name__ == "__main__":
    main()
