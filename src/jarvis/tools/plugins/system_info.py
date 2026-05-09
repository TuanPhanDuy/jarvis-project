"""Plugin: system_info — safe structured system diagnostics via psutil."""
from __future__ import annotations


def handle(tool_input: dict) -> str:
    try:
        import psutil

        query = str(tool_input.get("query", "all")).lower().strip()

        sections: list[str] = []

        def _do_cpu():
            pct = psutil.cpu_percent(interval=0.5)
            count = psutil.cpu_count()
            freq = psutil.cpu_freq()
            freq_str = f"{freq.current:.0f} MHz" if freq else "n/a"
            sections.append(f"**CPU**: {pct}% used, {count} cores, {freq_str}")

        def _do_memory():
            m = psutil.virtual_memory()
            swap = psutil.swap_memory()
            sections.append(
                f"**Memory**: {m.percent}% used "
                f"({_fmt(m.used)} / {_fmt(m.total)}, {_fmt(m.available)} free)\n"
                f"**Swap**: {swap.percent}% used ({_fmt(swap.used)} / {_fmt(swap.total)})"
            )

        def _do_disk():
            lines = ["**Disk**:"]
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    lines.append(f"  {part.mountpoint}: {usage.percent}% used ({_fmt(usage.used)} / {_fmt(usage.total)})")
                except PermissionError:
                    continue
            sections.append("\n".join(lines))

        def _do_processes():
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    procs.append(p.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            top = sorted(procs, key=lambda x: x.get("cpu_percent") or 0, reverse=True)[:10]
            rows = ["**Top 10 Processes (by CPU)**:", "| PID | Name | CPU% | MEM% |", "|-----|------|------|------|"]
            for p in top:
                rows.append(f"| {p['pid']} | {(p['name'] or '')[:20]} | {p.get('cpu_percent', 0):.1f} | {p.get('memory_percent', 0):.1f} |")
            sections.append("\n".join(rows))

        def _do_network():
            stats = psutil.net_io_counters()
            sections.append(
                f"**Network**: sent {_fmt(stats.bytes_sent)}, "
                f"recv {_fmt(stats.bytes_recv)}, "
                f"packets sent {stats.packets_sent}, recv {stats.packets_recv}"
            )

        dispatch = {
            "cpu": _do_cpu,
            "memory": _do_memory,
            "disk": _do_disk,
            "processes": _do_processes,
            "network": _do_network,
        }

        if query == "all":
            for fn in dispatch.values():
                fn()
        elif query in dispatch:
            dispatch[query]()
        else:
            return f"ERROR: unknown query '{query}'. Use: cpu, memory, disk, processes, network, all"

        return "\n\n".join(sections)

    except ImportError:
        return "ERROR: psutil not installed. Run: uv add psutil"
    except Exception as e:
        return f"ERROR: system_info failed — {e}"


def _fmt(bytes_val: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val //= 1024
    return f"{bytes_val:.1f} PB"


SCHEMA: dict = {
    "name": "system_info",
    "description": (
        "Get structured system diagnostics: CPU usage, memory, disk, top processes, network I/O. "
        "Uses psutil — safe, no shell commands, no injection risk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "enum": ["cpu", "memory", "disk", "processes", "network", "all"],
                "description": "Which metric to retrieve. Default: 'all'.",
            },
        },
        "required": [],
    },
}
