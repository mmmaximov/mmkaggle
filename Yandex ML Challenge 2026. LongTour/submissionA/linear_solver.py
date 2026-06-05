"""Domain-general detector + exact solver for GF(2)-linear ("toggle") puzzles.

A puzzle is GF(2)-linear if every action, applied from ANY state, XORs a fixed
set of binary cells (state-independent mask), actions are involutions and the
full action set is always available. Then reaching the goal is solving

        sum_{j in pressed} mask_j  ==  (current XOR goal)      (over GF(2))

Minimum number of moves = minimum-Hamming-weight solution x, found by Gaussian
elimination + nullspace coset minimization. Order of presses is irrelevant.
"""

import itertools
import numpy as np


def _cells_binary_vector(env, state):
    """Per-cell binary content vector from the unified encoding."""
    obs = env.encode_state(state)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    return cv


def detect_linear_gf2(env, num_probe_states=6, seed=0):
    """Return dict with masks/goal if the env is GF(2)-linear, else None."""
    rng = np.random.RandomState(seed)

    env.reset()
    solved = env.get_state()
    goal_vec = _cells_binary_vector(env, solved)

    # Domain must be binary {0,1}.
    if not np.all(np.isin(goal_vec, [0, 1])):
        return None

    base_actions = list(env.valid_actions())
    if not base_actions:
        return None

    N = len(goal_vec)

    # Gather probe states: solved + a few scrambles.
    probe_states = [solved]
    for k in range(num_probe_states):
        env.reset(seed=int(rng.randint(0, 10**9)))
        try:
            st, _ = env.scramble(length=int(rng.randint(3, 15)),
                                 seed=int(rng.randint(0, 10**9)))
        except Exception:
            return None
        probe_states.append(st)

    masks = {}
    for a in base_actions:
        ref = None
        for st in probe_states:
            env.set_state(st)
            v_before = _cells_binary_vector(env, env.get_state())
            # action set must be state-independent for a free subset choice
            if a not in env.valid_actions():
                return None
            env.step(a)
            v_after = _cells_binary_vector(env, env.get_state())
            if not np.all(np.isin(v_after, [0, 1])):
                return None
            delta = (v_before.astype(np.int64) ^ v_after.astype(np.int64))
            if ref is None:
                ref = delta
            elif not np.array_equal(ref, delta):
                return None  # mask depends on state -> not linear
        masks[a] = ref

    return {"actions": base_actions, "masks": masks, "goal": goal_vec, "N": N}


def _solve_gf2(columns, target, n_cells):
    """Solve XOR of selected columns == target over GF(2).

    columns: list of int bitmasks over cells (len = num actions)
    target:  int bitmask over cells
    Returns (particular_solution_actionmask:int, nullspace:list[int]) or None.
    Packs action-membership into the high bits so we track which columns used.
    """
    A = len(columns)
    # augmented vectors: low n_cells bits = cell pattern, then 1 bit per action
    aug = []
    for j, col in enumerate(columns):
        aug.append(col | (1 << (n_cells + j)))
    tgt = target  # action bits = 0

    pivot_rows = []  # (cell_bit, vector)
    used_bits = []
    vecs = aug[:]
    for bit in range(n_cells):
        # find a vector with this cell bit set, not already a pivot
        piv = None
        for v in vecs:
            if (v >> bit) & 1 and not any((v >> b) & 1 for b in used_bits):
                # ensure leading bit is `bit`: eliminate lower used bits first
                pass
        # simpler: standard elimination
        break

    # --- do clean elimination ---
    pivots = {}  # cell_bit -> reduced vector with that leading cell bit
    basis = list(aug)
    for v in basis:
        cur = v
        for b in range(n_cells):
            if (cur >> b) & 1:
                if b in pivots:
                    cur ^= pivots[b]
                else:
                    pivots[b] = cur
                    break
    # nullspace: vectors whose cell-part reduced to 0 but action-part nonzero
    nullspace = []
    for v in basis:
        cur = v
        for b in range(n_cells):
            if (cur >> b) & 1 and b in pivots:
                cur ^= pivots[b]
        if (cur & ((1 << n_cells) - 1)) == 0 and cur != 0:
            nullspace.append(cur >> n_cells)
    # reduce target through pivots, accumulating action bits
    cur = tgt
    for b in range(n_cells):
        if (cur >> b) & 1:
            if b in pivots:
                cur ^= pivots[b]
            else:
                return None  # unsolvable
    if (cur & ((1 << n_cells) - 1)) != 0:
        return None
    particular = cur >> n_cells

    # dedup nullspace into independent basis over action bits
    null_basis = []
    for nv in nullspace:
        cur = nv
        for bv in null_basis:
            hb = bv & (-bv)
            if cur & hb:
                cur ^= bv
        if cur:
            null_basis.append(cur)
    return particular, null_basis


def _popcount(x):
    return bin(x).count("1")


def solve_linear(env, state, info, max_null_enum=20):
    """Return minimal (near-minimal) action list, or None."""
    actions = info["actions"]
    masks = info["masks"]
    goal = info["goal"]
    N = info["N"]

    cur_vec = _cells_binary_vector(env, state).astype(np.int64)
    delta = (cur_vec ^ goal.astype(np.int64))

    def vec_to_int(v):
        x = 0
        for i in range(N):
            if v[i]:
                x |= (1 << i)
        return x

    target = vec_to_int(delta)
    columns = [vec_to_int(masks[a]) for a in actions]

    res = _solve_gf2(columns, target, N)
    if res is None:
        return None
    particular, null_basis = res

    best = particular
    best_w = _popcount(particular)
    k = len(null_basis)
    if 0 < k <= max_null_enum:
        for bits in range(1, 1 << k):
            x = particular
            bb = bits
            idx = 0
            while bb:
                if bb & 1:
                    x ^= null_basis[idx]
                bb >>= 1
                idx += 1
            w = _popcount(x)
            if w < best_w:
                best_w = w
                best = x

    out = [actions[j] for j in range(len(actions)) if (best >> j) & 1]
    return out
