"""Inference: detect family, solve all instances in parallel across CPU cores
under a RAM budget, write output_actions.csv. Always writes a row per instance."""

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

MODEL_PATH = "model.pt"
SAFETY_MARGIN = 15
TIME_LIMIT_DEFAULT = int(os.environ.get("SOLVE_TIME_LIMIT", 25 * 60))
MEM_BUDGET_BYTES = float(os.environ.get("SOLVE_MEM_BUDGET", 20e9))  # for search structures


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


_SOLVER = None


def _init_worker(model_path, max_nodes, max_unique):
    global _SOLVER
    import puzzle_solver
    _SOLVER = puzzle_solver.Solver(model_path=model_path,
                                   max_nodes=max_nodes, max_unique=max_unique)


def _solve_chunk(args):
    chunk, wall_budget, per_inst_cap = args
    out = []
    worker_deadline = time.time() + wall_budget
    for iid, state in chunk:
        now = time.time()
        if now >= worker_deadline:
            out.append((iid, ""))
            continue
        inst_deadline = min(now + per_inst_cap, worker_deadline)
        try:
            acts = _SOLVER.solve_one(state, inst_deadline)
        except Exception:
            acts = []
        out.append((iid, " ".join(acts)))
    return out


def _resolve_model_path():
    use_nn = False
    if os.path.exists("meta.json"):
        try:
            with open("meta.json") as f:
                use_nn = bool(json.load(f).get("use_nn", False))
        except Exception:
            use_nn = False
    return MODEL_PATH if (use_nn and os.path.exists(MODEL_PATH)) else None


def _plan_resources(family, N, cpu, measured_bpn=None):
    """Generic games are TIME-bound: parallelism (workers) drives coverage, so
    keep workers maxed and bound memory via the PER-WORKER node cap instead of
    dropping workers. Only drop workers in the extreme case where even a tiny cap
    at full workers would still blow the RAM budget."""
    W = max(1, min(8, cpu))
    if family in ("gf2", "slide"):
        return W, 200_000, 200_000
    if measured_bpn and measured_bpn > 0:
        bytes_per_node = measured_bpn * 1.3   # estimate already ~4x pickle size
    else:
        bytes_per_node = (N * 8 + 700) * 2.5  # conservative fallback guess
    cap = int(MEM_BUDGET_BYTES / (W * bytes_per_node))
    while W > 1 and cap < 4000:               # keep parallelism unless hopeless
        W -= 1
        cap = int(MEM_BUDGET_BYTES / (W * bytes_per_node))
    cap = max(4000, min(200_000, cap))
    return W, cap, cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int, default=TIME_LIMIT_DEFAULT)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    budget = args.time_limit - SAFETY_MARGIN
    instances = load_jsonl(args.input)
    n = len(instances)

    # probe once to learn family + state size, then plan resources
    import puzzle_solver
    probe = puzzle_solver.Solver(model_path=None)
    cpu = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    # estimate per-node memory from the ACTUAL state size (deterministic & safe):
    # heap holds state copies + key + dict overhead. Scales with the game's state.
    import pickle as _pk
    measured_bpn = None
    if probe.family == "generic" and instances:
        try:
            sample = instances[: min(20, len(instances))]
            sb = max(len(_pk.dumps(x["state"])) for x in sample)
            measured_bpn = sb * 4.0 + 900.0  # in-memory > pickle; + key + dict entries
        except Exception:
            measured_bpn = None
    W, max_nodes, max_unique = _plan_resources(probe.family, probe.N, cpu, measured_bpn)
    model_path = _resolve_model_path() if probe.family == "generic" else None
    del probe
    print(f"family-aware solve: workers={W} max_unique={max_unique} model={model_path}")

    items = [(inst["instance_id"], inst["state"]) for inst in instances]
    chunks = [[] for _ in range(W)]
    for i, it in enumerate(items):
        chunks[i % W].append(it)
    per_inst_cap = max(1.0, budget * W / max(1, n) * 4.0)
    tasks = [(c, budget, per_inst_cap) for c in chunks if c]

    results = {}
    if W == 1:
        _init_worker(model_path, max_nodes, max_unique)
        for t in tasks:
            for iid, a in _solve_chunk(t):
                results[iid] = a
    else:
        with Pool(W, initializer=_init_worker,
                  initargs=(model_path, max_nodes, max_unique)) as pool:
            for part in pool.map(_solve_chunk, tasks):
                for iid, a in part:
                    results[iid] = a

    solved = sum(1 for v in results.values() if v)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        w.writeheader()
        for inst in instances:
            w.writerow({"instance_id": inst["instance_id"],
                        "actions": results.get(inst["instance_id"], "")})

    print(f"final: solved {solved}/{n}, time {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
