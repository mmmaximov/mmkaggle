"""Per-hidden-game adaptation.

Detect the family. For gf2 / slide families nothing needs training (exact /
fast heuristic solvers). For the generic family, fit V(s) on backward random
walks with MIN-depth dedup (de-biases the random-walk-depth label), then run a
quick self-validation geo-vs-NN on freshly scrambled states and record which
heuristic to use. Always leaves model.pt + meta.json so solve.py can proceed.
"""

import argparse
import json
import os
import random
import time

import numpy as np

import gym
import common


MODEL_PATH = "model.pt"
META_PATH = "meta.json"
SAFETY_MARGIN = 60
TIME_LIMIT_DEFAULT = int(os.environ.get("TRAIN_TIME_LIMIT", 50 * 60))


def collect_min_depth(env, deadline, max_walk, target_unique=120_000,
                      min_states=2000, hard_deadline=None):
    """Backward walks; keep the minimum observed depth per unique state.
    Runs until `deadline`, but keeps going past it (up to `hard_deadline`) until
    at least `min_states` unique states are gathered, so data is never empty."""
    rng = random.Random(0)
    best = {}
    while len(best) < target_unique:
        now = time.time()
        if now >= deadline and len(best) >= min_states:
            break
        if hard_deadline is not None and now >= hard_deadline:
            break
        L = rng.randint(1, max_walk)
        env.reset(seed=rng.randint(0, 10**9))
        for depth in range(1, L + 1):
            a = rng.choice(env.valid_actions())
            env.step(a)
            s = common.to_jsonable(env.get_state())
            k = common.state_key(s)
            cur = best.get(k)
            if cur is None or depth < cur[0]:
                best[k] = (depth, s)
    pairs = list(best.values())
    random.Random(1).shuffle(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int, default=TIME_LIMIT_DEFAULT)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max_walk", type=int, default=80)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN

    import puzzle_solver
    solver = puzzle_solver.Solver(model_path=None)
    family = solver.family
    print(f"env_id={getattr(gym,'ENV_ID','?')} family={family}")

    meta = {"env_id": getattr(gym, "ENV_ID", "?"), "family": family, "use_nn": False}

    if family in ("gf2", "slide"):
        # exact / fast solver: no model needed
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"train.py done ({family}, no model) in {time.time()-start:.1f}s")
        return

    # ---- generic family: train V, then validate geo vs NN ----
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from model import ValueNet
        torch.manual_seed(args.seed)
        torch.set_num_threads(min(8, os.cpu_count() or 1))

        env = gym.make_env()
        env.reset()
        N = len(env.encode_state(env.get_state())["content_values"])
        bytes_per_row = N * common.TOKEN_FEAT_DIM * 4 + N * 8 + 400
        max_states = int(max(5_000, min(120_000, 3.5e9 / bytes_per_row)))
        # budget AFTER heavy imports (torch import can cost tens of seconds)
        t0 = time.time()
        remaining = max(1.0, deadline - t0)
        data_deadline = t0 + 0.35 * remaining
        train_deadline = t0 + 0.85 * remaining  # reserve ~15% for validation
        print(f"collecting data (min-depth dedup), N={N}, cap={max_states}...")
        pairs = collect_min_depth(env, data_deadline, args.max_walk,
                                  target_unique=max_states,
                                  min_states=min(2000, max_states),
                                  hard_deadline=train_deadline)
        print(f"  {len(pairs)} unique states")

        n = len(pairs)
        labels = np.array([d for d, _ in pairs], dtype=np.float32)
        tokens = np.empty((n, N, common.TOKEN_FEAT_DIM), dtype=np.float32)
        for i, (_, s) in enumerate(pairs):
            tokens[i] = common.encode_tokens(env, s)
        del pairs

        def to_t(tk):
            B, Nn, _ = tk.shape
            p = common.split_token_features(tk.reshape(B * Nn, -1))
            return (torch.from_numpy(p["dense"].reshape(B, Nn, -1)),
                    torch.from_numpy(p["content_value"].reshape(B, Nn)),
                    torch.from_numpy(p["target_value"].reshape(B, Nn)))

        model = ValueNet()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        lf = nn.SmoothL1Loss()
        ep = 0
        while time.time() < train_deadline and n >= 2:
            idx = np.random.permutation(n)
            for s in range(0, n, 256):
                if time.time() >= train_deadline:
                    break
                sel = idx[s:s + 256]
                if len(sel) < 2:
                    continue
                d, cv, tv = to_t(tokens[sel]); y = torch.from_numpy(labels[sel])
                pred = model(d, cv, tv); loss = lf(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
            ep += 1
        torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
        print(f"  trained {ep} epochs")

        # ---- self-validation: geo vs NN coverage on fresh scrambles ----
        meta["use_nn"] = _validate(env, model, deadline)
    except Exception as e:
        print(f"  generic training skipped/failed: {repr(e)}")
        meta["use_nn"] = False

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"train.py done (generic, use_nn={meta['use_nn']}) in {time.time()-start:.1f}s")


def _validate(env, model, deadline):
    """Return True iff NN heuristic solves strictly more fresh scrambles than geo
    within a small budget. Conservative: default to geo on ties."""
    import torch
    import search_v2 as SV
    val_deadline = min(deadline, time.time() + 90)
    env.reset(); solved_key = SV.fast_key(env.get_state())
    H = SV.Heuristic(env)
    geo = H.h_geo

    def v_fn(state):
        tk = common.encode_tokens(env, state)[None]
        B, N, _ = tk.shape
        p = common.split_token_features(tk.reshape(B * N, -1))
        d = torch.from_numpy(p["dense"].reshape(B, N, -1))
        cv = torch.from_numpy(p["content_value"].reshape(B, N))
        tv = torch.from_numpy(p["target_value"].reshape(B, N))
        with torch.no_grad():
            return float(model(d, cv, tv)[0])

    lengths = list(getattr(gym, "SCRAMBLE_LENGTHS_DEFAULT", [30]))
    probes = []
    rng = random.Random(7)
    for i in range(16):
        L = rng.choice(lengths)
        st, _ = env.scramble(length=L, seed=rng.randint(0, 10**9))
        if not env.is_solved():
            probes.append(common.to_jsonable(st))
    cov_geo = cov_nn = 0
    for st in probes:
        if time.time() >= val_deadline:
            break
        if SV.solve_perm(env, st, solved_key, geo, time.time() + 1.5):
            cov_geo += 1
    for st in probes:
        if time.time() >= val_deadline:
            break
        if SV.solve_perm(env, st, solved_key, v_fn, time.time() + 1.5):
            cov_nn += 1
    print(f"  validation coverage  geo={cov_geo}  nn={cov_nn}  (of {len(probes)})")
    return cov_nn > cov_geo


if __name__ == "__main__":
    main()
