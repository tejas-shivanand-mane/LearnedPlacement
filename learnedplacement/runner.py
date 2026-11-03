"""Resumable single-process runner. Nothing executes on import."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import zipfile

from .metrics import summarize
from .model import Case, Plan, make_cases, seed_for, validate_case, violations
from .planners import ALL_METHODS, METHODS, plan_method
from .simulator import BudgetExpired, check_budget, generate_trace, simulate


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_snapshot():
    paths = sorted((ROOT / "learnedplacement").glob("*.py"))
    return {str(p.relative_to(ROOT)): p.read_text() for p in paths}


def validate_config(config):
    methods = config.get("methods", list(METHODS))
    if not methods or len(set(methods)) != len(methods) or any(m not in ALL_METHODS for m in methods):
        raise ValueError("Methods must be known, nonempty and unique")
    dependencies = {"joint_local_no_backup": ["joint_local"],
                    "joint_local": list(METHODS[:4]), "joint_greedy": list(METHODS[:4]),
                    "matched_failure_unaware": ["matched_failure_aware"],
                    "matched_no_backup": ["matched_failure_aware"]}
    for i, method in enumerate(methods):
        if any(parent not in methods[:i] for parent in dependencies.get(method, [])):
            raise ValueError(f"Missing or misordered dependencies for {method}")
    if "matched_failure_aware" in methods:
        if type(config.get("matched_candidates")) is not int or config["matched_candidates"] <= 0:
            raise ValueError("matched_candidates must be a positive integer")
        if not isinstance(config.get("matched_seconds"), (int, float)) or not 0 < config["matched_seconds"] < float("inf"):
            raise ValueError("matched_seconds must be positive and finite")
    profiles = config.get("profiles", [{"name": "reference"}])
    names = [p["name"] for p in profiles]
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Profile names must be unique letters/digits/underscores")
    for profile in profiles:
        if set(profile) - {"name", "description", "workload", "deadline_factor", "cost_budget_factor"}:
            raise ValueError("Unknown profile setting")
        for key in ("deadline_factor", "cost_budget_factor"):
            if key in profile and (not isinstance(profile[key], (int, float)) or not 0 < profile[key] < float("inf")):
                raise ValueError(f"Invalid profile {key}")
    for key in ("instances_per_regime", "planning_traces", "evaluation_traces",
                "greedy_candidates", "local_candidates", "save_every_traces"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["evaluation_traces"] < 2:
        raise ValueError("At least two evaluation traces required")
    for key in ("greedy_seconds", "local_seconds", "deadline_factor",
                "cost_budget_factor", "simulation_horizon_factor"):
        if not isinstance(config[key], (int, float)) or not 0 < config[key] < float("inf"):
            raise ValueError(f"Invalid {key}")
    shards = config["workload"]["shards"]
    if not config["active_nodes"] or any(type(n) is not int or n <= 0 or n % shards for n in config["active_nodes"]):
        raise ValueError("Active-node counts must be positive multiples of shards")
    if not config["regimes"] or len(set(config["regimes"])) != len(config["regimes"]):
        raise ValueError("Regimes must be nonempty and unique")


def prepare(config, output):
    validate_config(config)
    sources = source_snapshot()
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}
    fingerprint = hashlib.sha256(json.dumps({"config": config, "sources": hashes}, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Configuration or simulator source changed. Choose a NEW output directory; do not mix runs.")
        return manifest
    cases = make_cases(config)
    for c in cases:
        atomic_json(output / "cases" / c.case_id / "case.json", c.to_dict())
    for name, content in sources.items():
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    manifest = dict(schema=2, fingerprint=fingerprint, config=config, sources=hashes,
                    cases=[c.case_id for c in cases], methods=config.get("methods", list(METHODS)),
                    python=platform.python_version(), platform=platform.platform(),
                    created_unix_s=time.time(),
                    caveat="Synthetic surrogate, not measured HSDP. No learned policy or exact optimizer.")
    atomic_json(path, manifest)
    return manifest


def status(output):
    output = Path(output)
    if not (output / "manifest.json").exists():
        return {"state": "not_started"}
    m = read_json(output / "manifest.json")
    total = len(m["cases"])*len(m["methods"])
    done = sum((output/"cases"/c/f"{method}.summary.json").exists()
               for c in m["cases"] for method in m["methods"])
    saved = read_json(output/"status.json") if (output/"status.json").exists() else {}
    return dict(saved, completed_comparisons=done, total_comparisons=total,
                fraction_complete=done/total if total else 1.)


def build_summary(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    results = []
    for cid in manifest["cases"]:
        directory = output / "cases" / cid
        c = Case.from_dict(read_json(directory / "case.json"))
        for method in manifest["methods"]:
            file = directory / f"{method}.summary.json"
            if file.exists():
                result = read_json(file)
                results.append(dict(case_id=cid, regime=c.regime, profile=c.profile, active_nodes=c.replicas*c.shards,
                                    method=method, deadline_s=c.deadline_s,
                                    cost_budget_usd=c.cost_budget_usd, **result))
    atomic_json(output / "summary.json", {"results": results, "status": status(output)})
    if results:
        flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in results]
        fields = sorted(set().union(*(r.keys() for r in flat)))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
        tmp = output / "summary.csv.tmp"
        tmp.write_text(buf.getvalue())
        os.replace(tmp, output / "summary.csv")
    return results


def run(config_path=None, output="results/pilot_v2", max_minutes=30, config=None):
    """Run/resume with a cooperative wall-clock cap (including planning/evaluation).

    Completed plans and batches of observations are retained. On resume, only a
    partially evaluated trace or globally interrupted planner is restarted.
    """
    if not isinstance(max_minutes, (int, float)) or not 0 < max_minutes < float("inf"):
        raise ValueError("max_minutes must be positive and finite")
    if config is None:
        config = read_json(config_path or ROOT / "configs" / "pilot_v2.json")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run directory is locked: {lock}. Stop the old runner first. "
                           "After a killed kernel, remove this file only after confirming it is no longer running.") from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stop_at = time.monotonic() + max_minutes*60
    current = None
    try:
        manifest = prepare(config, output)
        atomic_json(output / "status.json", dict(state="running"))
        for cid in manifest["cases"]:
            check_budget(stop_at)
            directory = output / "cases" / cid
            c = Case.from_dict(read_json(directory / "case.json"))
            trace_key = c.trace_key or cid
            validate_case(c)
            plans_path = directory / "plans.json"
            plans = read_json(plans_path) if plans_path.exists() else {}
            planning_traces = None
            evaluation_traces = {}
            for method in manifest["methods"]:
                current = dict(case_id=cid, method=method)
                check_budget(stop_at)
                if (directory / f"{method}.summary.json").exists():
                    continue
                if method not in plans:
                    if planning_traces is None and method != "joint_local_no_backup":
                        planning_traces = []
                        for i in range(config["planning_traces"]):
                            check_budget(stop_at)
                            planning_traces.append(generate_trace(c, seed_for(config["seed"], trace_key, "planning", i), stop_at))
                    print(f"{cid}: planning {method}", flush=True)
                    candidate = plan_method(c, method, planning_traces, config, plans, stop_at)
                    check_budget(stop_at)  # A globally interrupted planner is retried on resume.
                    plans[method] = candidate
                    atomic_json(plans_path, plans)
                p = Plan.from_dict(plans[method]["plan"])
                errors = violations(c, p)
                if errors:
                    raise AssertionError(f"{method} generated invalid plan: {errors}")
                healthy_path = directory / f"{method}.failure_free.json"
                if not healthy_path.exists():
                    atomic_json(healthy_path, simulate(c, p, [], stop_at))
                healthy = read_json(healthy_path)
                obs_path = directory / f"{method}.observations.json"
                rows = read_json(obs_path) if obs_path.exists() else []
                print(f"{cid}: evaluating {method} ({len(rows)}/{config['evaluation_traces']} saved)", flush=True)
                try:
                    for i in range(len(rows), config["evaluation_traces"]):
                        check_budget(stop_at)
                        if i not in evaluation_traces:
                            evaluation_traces[i] = generate_trace(c, seed_for(config["seed"], trace_key, "evaluation", i), stop_at)
                        row = simulate(c, p, evaluation_traces[i], stop_at)
                        rows.append(dict(trace_index=i, trace_seed=seed_for(config["seed"], trace_key, "evaluation", i), **row))
                        if len(rows) % config["save_every_traces"] == 0:
                            atomic_json(obs_path, rows)
                            atomic_json(output/"status.json", dict(state="running", **current, saved_traces=len(rows)))
                finally:
                    atomic_json(obs_path, rows)
                summary = summarize(c, rows)
                summary.update(planning_s=plans[method]["planning_s"],
                               failure_free_completion_s=healthy["completion_s"],
                               failure_penalty_s=(summary["mean_completion_s"]-healthy["completion_s"]
                                                  if summary["mean_completion_s"] is not None and healthy["completed"] else None),
                               candidates=plans[method]["candidates"],
                               planner_status=plans[method]["status"],
                               structural_feasible=True,
                               allocated_nodes=len(p.allocated),
                               backup_gib=len(p.backups)*c.shard_gib,
                               no_feasible_plan_found_on_planning=(plans[method].get("planning_point_feasible") is False))
                atomic_json(directory / f"{method}.summary.json", summary)
                print(f"  time={summary['mean_completion_s']} s, cost={summary['mean_cost_usd']}, "
                      f"deadline={summary['deadline_success']:.3f}, "
                      f"screen_pass={summary['statistical_screen_passed']}", flush=True)
            build_summary(output)
        atomic_json(output/"status.json", dict(state="finished"))
    except BudgetExpired:
        atomic_json(output/"status.json", dict(state="paused_time_limit", **(current or {})))
        print("Time budget reached. Run the same cell/command to resume.", flush=True)
    except KeyboardInterrupt:
        atomic_json(output/"status.json", dict(state="paused_interrupt", **(current or {})))
        print("Interrupted; completed observations saved. Run again to resume.", flush=True)
    except Exception as error:
        atomic_json(output/"status.json", dict(state="error", error=repr(error), **(current or {})))
        raise
    finally:
        try:
            if (output/"manifest.json").exists():
                build_summary(output)
        finally:
            lock.unlink(missing_ok=True)
    return status(output)


def paired_report(output, selected="joint_local", baseline="separate", repeats=1000):
    """Within-case paired bootstrap; never selects a plan using evaluation traces."""
    output = Path(output)
    manifest = read_json(output/"manifest.json")
    comparisons = []
    for cid in manifest["cases"]:
        d = output/"cases"/cid
        paths = [d/f"{m}.summary.json" for m in (selected, baseline)]
        if not all(p.exists() for p in paths):
            continue
        a, b = [read_json(d/f"{m}.observations.json") for m in (selected, baseline)]
        plans = read_json(d/"plans.json")
        matched = {selected, baseline} == {"matched_failure_aware", "matched_failure_unaware"}
        if matched:
            x, y = plans[selected], plans[baseline]
            if x["candidate_pool_sha256"] != y["candidate_pool_sha256"] or x["candidates"] != y["candidates"]:
                raise ValueError("Matched candidate pool mismatch")
        summary_a, summary_b = [read_json(path) for path in paths]
        if len(a) != len(b) or any(x["trace_seed"] != y["trace_seed"] for x, y in zip(a, b)):
            raise ValueError("Paired trace mismatch")
        if not all(r["completed"] for r in a+b):
            comparisons.append(dict(case_id=cid, status="censored_no_mean_completion_comparison"))
            continue
        rng = random.Random(seed_for(cid, selected, baseline, "bootstrap"))
        delta = [x["completion_s"]-y["completion_s"] for x, y in zip(a, b)]
        boot = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(repeats))
        healthy_a, healthy_b = (s.get("failure_free_completion_s") for s in (summary_a, summary_b))
        healthy_delta = healthy_a-healthy_b if healthy_a is not None and healthy_b is not None else None
        comparisons.append(dict(case_id=cid, profile=Case.from_dict(read_json(d/"case.json")).profile,
                                status="complete", selected=selected, baseline=baseline,
                                matched_candidate_pool=matched,
                                matched_candidates=plans[selected]["candidates"] if matched else None,
                                failure_free_time_difference_s=healthy_delta,
                                failure_penalty_difference_s=statistics.mean(delta)-healthy_delta if healthy_delta is not None else None,
                                failure_penalty_difference_ci95=([boot[int(.025*repeats)]-healthy_delta,
                                                                 boot[min(repeats-1, int(.975*repeats))]-healthy_delta]
                                                                if healthy_delta is not None else None),
                                time_difference_s=statistics.mean(delta),
                                time_difference_ci95=[boot[int(.025*repeats)], boot[min(repeats-1, int(.975*repeats))]],
                                time_gain_percent=100*(1-statistics.mean(x["completion_s"] for x in a)/statistics.mean(y["completion_s"] for y in b)),
                                cost_difference_usd=statistics.mean(x["cost_usd"]-y["cost_usd"] for x, y in zip(a, b)),
                                selected_screen=read_json(paths[0])["statistical_screen_passed"],
                                baseline_screen=read_json(paths[1])["statistical_screen_passed"]))
    atomic_json(output/f"paired-{selected}-vs-{baseline}.json", comparisons)
    return comparisons


def export_run(output, destination=None):
    output = Path(output).resolve()
    if (output/".run.lock").exists():
        raise RuntimeError("Stop/pause the runner before exporting")
    if not (output/"manifest.json").exists():
        raise ValueError("No run manifest to export")
    destination = Path(destination).resolve() if destination else output.with_suffix(".zip")
    if destination == output or output in destination.parents:
        raise ValueError("Place the archive outside the run directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, str(path.relative_to(output)))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs"/"pilot_v2.json"))
    parser.add_argument("--output", default="results/pilot_v2")
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    print(json.dumps(status(args.output) if args.status else run(args.config, args.output, args.max_minutes), indent=2))


if __name__ == "__main__":
    main()
