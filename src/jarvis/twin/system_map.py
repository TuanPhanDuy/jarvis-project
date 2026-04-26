"""System map: snapshot running processes, listening ports, disk mounts, and git repos.

Uses psutil for process/port/disk data. All functions return plain dicts suitable
for storage in the knowledge graph or comparison via diff.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def _safe_proc_info(proc) -> dict | None:
    try:
        return {
            "pid": proc.pid,
            "name": proc.name(),
            "status": proc.status(),
            "cpu_percent": proc.cpu_percent(interval=None),
        }
    except Exception:
        return None


def snapshot_processes(top_n: int = 20) -> list[dict]:
    """Return top-N processes by CPU usage."""
    try:
        import psutil
        procs = []
        for p in psutil.process_iter():
            info = _safe_proc_info(p)
            if info:
                procs.append(info)
        procs.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
        return procs[:top_n]
    except ImportError:
        return []
    except Exception:
        return []


def snapshot_listening_ports() -> list[dict]:
    """Return all listening TCP/UDP ports."""
    try:
        import psutil
        ports = []
        for conn in psutil.net_connections(kind="inet"):
            if conn.status in ("LISTEN", ""):  # LISTEN for TCP, "" for UDP
                laddr = conn.laddr
                ports.append({
                    "port": laddr.port if laddr else 0,
                    "host": laddr.ip if laddr else "",
                    "proto": "tcp" if conn.type == 1 else "udp",
                    "pid": conn.pid,
                })
        return ports
    except ImportError:
        return []
    except Exception:
        return []


def snapshot_disk_mounts() -> list[dict]:
    """Return disk mount points and usage."""
    try:
        import psutil
        mounts = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                mounts.append({
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / 1e9, 1),
                    "used_pct": usage.percent,
                })
            except Exception:
                continue
        return mounts
    except ImportError:
        return []
    except Exception:
        return []


def snapshot_git_repos(search_roots: list[str] | None = None) -> list[dict]:
    """Find git repositories under common root paths."""
    roots = search_roots or [os.path.expanduser("~"), os.getcwd()]
    repos = []
    seen: set[str] = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # Search up to 3 levels deep to avoid scanning entire filesystem
        for git_dir in root_path.glob("*/.git"):
            repo_path = str(git_dir.parent)
            if repo_path not in seen:
                seen.add(repo_path)
                repos.append({"path": repo_path, "name": git_dir.parent.name})
        for git_dir in root_path.glob("*/*/.git"):
            repo_path = str(git_dir.parent)
            if repo_path not in seen:
                seen.add(repo_path)
                repos.append({"path": repo_path, "name": git_dir.parent.name})
    return repos[:20]


def take_full_snapshot() -> dict:
    """Return a complete system snapshot dict."""
    return {
        "ts": time.time(),
        "processes": snapshot_processes(),
        "ports": snapshot_listening_ports(),
        "disks": snapshot_disk_mounts(),
        "git_repos": snapshot_git_repos(),
    }
