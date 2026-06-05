"""Domain-general search for permutation puzzles: geometric heuristic + weighted A*.

Heuristic is built purely from the unified encoding (positions / content / target),
so it works for any hidden game. Two components:
  * geo: scale-recovered Manhattan displacement of each item to its goal cell
         (assignment within interchangeable value-classes), admissible-ish.
  * mismatch: number of cells whose (type,value) != target — cheap universal signal.

Search: batched weighted A* (f = g + W*h). Falls back to higher W / greedy rollout
so it (almost) never returns empty on solvable instances.
"""

import heapq
import time
import numpy as np

import gym


def fast_key(state):
    """Fast hashable key without json."""
    if isinstance(state, dict):
        parts = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, np.ndarray):
                parts.append((k, v.tobytes()))
            elif isinstance(v, list):
                parts.append((k, np.asarray(v).tobytes()))
            else:
                parts.append((k, v))
        return tuple(parts)
    arr = np.asarray(state)
    return arr.tobytes()


class Heuristic:
    """Built once from the solved state of `env`."""

    def __init__(self, env):
        self.env = env
        solved = env.get_state()
        obs = env.encode_state(solved)
        self.pos = np.asarray(obs["positions"], dtype=np.float64)  # (N,3)
        self.tt = np.asarray(obs["target_types"], dtype=np.int64)
        self.tv = np.asarray(obs["target_values"], dtype=np.int64)
        self.N = len(self.tv)
        # goal vector for the FAST heuristic (raw state, no encode_state) — ~50x
        # cheaper than h_geo, which lets the time-bound generic search explore far
        # more nodes (heuristic affects only guidance, never correctness).
        self.goal_raw = self._raw_vec(solved)

        # recover per-axis grid step to express displacement in "cell steps"
        self.step = np.ones(3)
        for ax in range(3):
            vals = np.unique(np.round(self.pos[:, ax], 6))
            if len(vals) > 1:
                d = np.diff(np.sort(vals))
                d = d[d > 1e-9]
                self.step[ax] = d.min() if len(d) else 1.0

        # goal: for each value-class (type,value), the set of target positions
        from collections import defaultdict
        self.goal_cells = defaultdict(list)
        EMPTY = gym.CONTENT_EMPTY
        for i in range(self.N):
            if self.tt[i] == EMPTY:
                continue
            self.goal_cells[(int(self.tt[i]), int(self.tv[i]))].append(i)
        self.EMPTY = EMPTY

    @staticmethod
    def _raw_vec(state):
        if isinstance(state, dict):
            return np.concatenate([np.asarray(state[k]).ravel().astype(np.float64)
                                   for k in sorted(state)])
        return np.asarray(state).ravel().astype(np.float64)

    def h_fast(self, state):
        rv = self._raw_vec(state)
        if rv.shape != self.goal_raw.shape:  # structural mismatch -> fall back
            return self.h_mismatch(state)
        return float(np.count_nonzero(rv != self.goal_raw))

    def _grid(self, pos):
        return pos / self.step  # (N,3) in cell-step units

    def h_geo(self, state):
        obs = self.env.encode_state(state)
        ct = np.asarray(obs["content_types"], dtype=np.int64)
        cv = np.asarray(obs["content_values"], dtype=np.int64)
        gpos = self._grid(self.pos)
        from collections import defaultdict
        cur_cells = defaultdict(list)
        for i in range(self.N):
            if ct[i] == self.EMPTY:
                continue
            cur_cells[(int(ct[i]), int(cv[i]))].append(i)

        total = 0.0
        for key, cur_idx in cur_cells.items():
            goal_idx = self.goal_cells.get(key, [])
            if not goal_idx:
                # value with no goal slot: count as misplaced (1 each)
                total += len(cur_idx)
                continue
            if len(cur_idx) == 1 and len(goal_idx) == 1:
                total += np.abs(gpos[cur_idx[0]] - gpos[goal_idx[0]]).sum()
            else:
                # interchangeable class: greedy nearest assignment (cheap proxy)
                G = gpos[goal_idx]
                for ci in cur_idx:
                    d = np.abs(G - gpos[ci]).sum(axis=1)
                    total += d.min()
        return float(total)

    def h_mismatch(self, state):
        obs = self.env.encode_state(state)
        ct = np.asarray(obs["content_types"], dtype=np.int64)
        cv = np.asarray(obs["content_values"], dtype=np.int64)
        return float(np.sum((ct != self.tt) | (cv != self.tv)))


def weighted_astar(env, initial_state, solved_key, h_fn, deadline,
                   W=2.0, max_nodes=200_000):
    start = initial_state
def weighted_astar(env, initial_state, solved_key, h_fn, deadline,
                   W=2.0, max_nodes=200_000, max_unique=250_000):
    start = initial_state
    sk = fast_key(start)
    if sk == solved_key:
        return []
    # parents/g hold only compact keys+ints; the heavy state object travels in
    # the heap entry and is freed once popped -> bounded memory.
    parents = {sk: (None, None)}
    g = {sk: 0}
    h0 = h_fn(start)
    cnt = 0
    heap = [(W * h0, cnt, 0, sk, start)]
    expanded = 0

    while heap:
        if expanded >= max_nodes or time.time() >= deadline:
            break
        if len(g) >= max_unique:  # memory guard: bail out to caller/fallback
            break
        f, _, gc, k, st = heapq.heappop(heap)
        if k == solved_key:
            return _reconstruct(k, parents)
        if gc > g.get(k, gc):  # stale heap entry (a better path was found)
            continue
        try:
            env.set_state(st)
            valid = env.valid_actions()
        except Exception:
            continue
        for a in valid:
            try:
                env.set_state(st)
                env.step(a)
                ns = env.get_state()
            except Exception:
                continue
            nk = fast_key(ns)
            ng = gc + 1
            if g.get(nk, 1 << 30) <= ng:
                continue
            parents[nk] = (k, a)
            g[nk] = ng
            if nk == solved_key:
                return _reconstruct(nk, parents)
            hv = h_fn(ns)
            cnt += 1
            heapq.heappush(heap, (ng + W * hv, cnt, ng, nk, ns))
        expanded += 1
    return None


def _reconstruct(end_k, parents):
    actions = []
    cur = end_k
    while True:
        pk, a = parents.get(cur, (None, None))
        if pk is None or a is None:
            break
        actions.append(a)
        cur = pk
    actions.reverse()
    return actions


def solve_perm(env, state, solved_key, h_fn, deadline, max_nodes=200_000,
               max_unique=250_000):
    """Weighted A* with the full budget on W=2 (best coverage on these puzzles,
    where every solved instance already hits the 2.0 score cap, so this is a pure
    coverage game). Falls through to greedier W only if W=2 hit the NODE cap (not
    the time deadline), giving a second shot with a different search shape."""
    state = gym.to_jsonable(state)
    for W in (2.0, 3.0, 5.0):
        if time.time() >= deadline:
            break
        res = weighted_astar(env, state, solved_key, h_fn, deadline,
                             W=W, max_nodes=max_nodes, max_unique=max_unique)
        if isinstance(res, list):
            return res
    return None
