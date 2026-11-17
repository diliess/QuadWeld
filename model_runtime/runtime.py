"""Select isolated Python environments and launch model or simulator workers."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SAM_CHECKPOINT = ROOT / "thirdparty/sam3/assets/model/sam3.1_multiplex.pt"


def child_environment():
    """Each interpreter loads its own Python and native libraries."""
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV",
                "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CUDA_VISIBLE_DEVICES"):
        environment.pop(key, None)
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return environment


def python_candidates(role, explicit=None):
    configured = explicit or os.environ.get(f"QUADWELD_{role.upper()}_PYTHON")
    if configured:
        return [Path(configured).expanduser().absolute()]
    candidates = []
    if role == "isaac":
        candidates.append(ROOT / ".venv/bin/python")
    candidates += [Path(sys.executable), ROOT / f".venv-{role}/bin/python"]
    name = {"isaac": "isaacsim", "sam": "sam3"}[role]
    for directory in ("miniconda3", "miniforge3", "anaconda3"):
        candidates.append(Path.home() / directory / "envs" / name / "bin/python")
    registry = Path.home() / ".conda/environments.txt"
    if registry.is_file():
        candidates += [Path(line.strip()) / "bin/python" for line in registry.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
    return list(dict.fromkeys(path.absolute() for path in candidates))


def find_python(role, explicit=None):
    modules = {"isaac": ["isaacsim", "PIL", "numpy"],
               "sam": ["sam3", "torch", "torchvision", "cv2", "PIL"]}[role]
    probe = ("import importlib.util,json,sys; "
             "print(json.dumps({'version':list(sys.version_info[:2]),"
             "'missing':[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]}))")
    failures = []
    for executable in python_candidates(role, explicit):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            if explicit or os.environ.get(f"QUADWELD_{role.upper()}_PYTHON"):
                failures.append(f"Not an executable file: {executable}")
            continue
        try:
            result = subprocess.run([str(executable), "-c", probe, *modules], cwd=ROOT,
                                    env=child_environment(), capture_output=True, text=True, timeout=15)
            info = json.loads(result.stdout.strip()) if result.returncode == 0 else None
            if info and info["version"] == [3, 12] and not info["missing"]:
                return str(executable), None
            failures.append(f"{executable}: {info or result.stderr.strip()[-400:]}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            failures.append(f"{executable}: {error}")
    detail = "; ".join(failures)
    return None, (f"Set --{role}-python or QUADWELD_{role.upper()}_PYTHON to a Python 3.12 environment "
                  f"containing {', '.join(modules)}. {detail}").strip()


def preflight(args, roles):
    interpreters, issues = {}, []
    for role in roles:
        executable, error = find_python(role, getattr(args, f"{role}_python"))
        interpreters[role] = executable
        if error:
            issues.append(error)
    weights = []
    if "sam" in roles:
        weights.append(SAM_CHECKPOINT)
    for path in weights:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing model file: {path}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if gpu.returncode or str(args.gpu) not in gpu.stdout.split():
            issues.append(f"NVIDIA GPU {args.gpu} unavailable: {gpu.stderr.strip() or gpu.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired) as error:
        issues.append(f"NVIDIA GPU check failed: {error}")
    return {"ready": not issues, "interpreters": interpreters, "issues": issues}


def run_process(command, log_path):
    entry = command[2] if command[1] == "-m" else Path(command[1]).name
    print(f"Running {entry}; log: {log_path}", flush=True)
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Process exited with status {code}; see {log_path}")
