#!/usr/bin/env bash
set -euo pipefail

RVI_REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVI_RELAY_COMMIT="eab21451f99e1a40fbb244f556de766d153c88f5"
RVI_RELAY_LOCK_SHA256="693489b8ebb68350b9603fad07486c05e60fcd84aa2842305e19d2c6e26b5685"
RVI_RELAY_DIR="${RVI_RELAY_DIR:-${RVI_REPOSITORY_ROOT}/third_party/Relay-OPD}"
RVI_GPU_VENV_DIR="${RVI_GPU_VENV_DIR:-${RVI_REPOSITORY_ROOT}/.venv-gpu}"
RVI_PYTHON_BIN="${PYTHON_BIN:-python3.12}"

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "Neither sha256sum nor shasum is available." >&2
        return 20
    fi
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "The confirmatory GPU environment requires Linux x86_64." >&2
    exit 21
fi

if ! command -v "${RVI_PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${RVI_PYTHON_BIN}" >&2
    exit 25
fi
RVI_PYTHON_VERSION="$("${RVI_PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
if [[ "${RVI_PYTHON_VERSION}" != "3.12.13" ]]; then
    echo "The confirmatory GPU environment requires Python 3.12.13; found ${RVI_PYTHON_VERSION}." >&2
    exit 26
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to verify the host driver and GPU." >&2
    exit 27
fi

if [[ ! -e "${RVI_RELAY_DIR}" ]]; then
    mkdir -p "$(dirname "${RVI_RELAY_DIR}")"
    mkdir "${RVI_RELAY_DIR}"
    git -C "${RVI_RELAY_DIR}" init
    git -C "${RVI_RELAY_DIR}" remote add origin \
        https://github.com/ZJU-REAL/Relay-OPD.git
    git -C "${RVI_RELAY_DIR}" fetch --depth 1 origin "${RVI_RELAY_COMMIT}"
    git -C "${RVI_RELAY_DIR}" checkout --detach FETCH_HEAD
elif [[ ! -d "${RVI_RELAY_DIR}/.git" ]]; then
    echo "Refusing non-Git Relay path: ${RVI_RELAY_DIR}" >&2
    exit 22
fi

RVI_RELAY_REMOTE="$(git -C "${RVI_RELAY_DIR}" remote get-url origin)"
if [[ "${RVI_RELAY_REMOTE}" != "https://github.com/ZJU-REAL/Relay-OPD.git" && \
      "${RVI_RELAY_REMOTE}" != "git@github.com:ZJU-REAL/Relay-OPD.git" ]]; then
    echo "Unexpected Relay origin: ${RVI_RELAY_REMOTE}" >&2
    exit 28
fi
RVI_ACTUAL_RELAY_COMMIT="$(git -C "${RVI_RELAY_DIR}" rev-parse HEAD)"
if [[ "${RVI_ACTUAL_RELAY_COMMIT}" != "${RVI_RELAY_COMMIT}" ]]; then
    echo "Relay checkout must be ${RVI_RELAY_COMMIT}; found ${RVI_ACTUAL_RELAY_COMMIT}." >&2
    echo "Use a new RVI_RELAY_DIR; this script will not overwrite an existing checkout." >&2
    exit 23
fi
RVI_RELAY_DIRTY="$(git -C "${RVI_RELAY_DIR}" status --porcelain --untracked-files=all)"
if [[ -n "${RVI_RELAY_DIRTY}" ]]; then
    echo "Relay checkout is dirty; refusing to install a modified confirmatory stack." >&2
    echo "Use a fresh RVI_RELAY_DIR. This script will not reset local changes." >&2
    exit 29
fi

RVI_RELAY_LOCK_PATH="${RVI_RELAY_DIR}/relay-opd/environment/requirements.lock.txt"
RVI_ACTUAL_LOCK_SHA256="$(hash_file "${RVI_RELAY_LOCK_PATH}")"
if [[ "${RVI_ACTUAL_LOCK_SHA256}" != "${RVI_RELAY_LOCK_SHA256}" ]]; then
    echo "Relay dependency lock SHA256 mismatch." >&2
    exit 24
fi

PYTHON_BIN="${RVI_PYTHON_BIN}" VENV_DIR="${RVI_GPU_VENV_DIR}" \
    bash "${RVI_RELAY_DIR}/relay-opd/environment/create_locked_env.sh"

"${RVI_GPU_VENV_DIR}/bin/python" -m pip install \
    --no-deps --no-build-isolation -e "${RVI_REPOSITORY_ROOT}"
RVI_RELAY_DIR="${RVI_RELAY_DIR}" "${RVI_GPU_VENV_DIR}/bin/python" \
    "${RVI_REPOSITORY_ROOT}/scripts/check_environment.py" --profile gpu

echo
echo "RvI-OPD GPU environment created at ${RVI_GPU_VENV_DIR}"
echo "Activate it with: source ${RVI_GPU_VENV_DIR}/bin/activate"
