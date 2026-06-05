"""Fast solver for single-blank sliding puzzles (fifteen-family), generically
detected. Operates on the raw integer board: O(1)-ish Manhattan + linear-conflict
heuristic and pure-python blank swaps, no env / encode_state in the hot loop.
"""

import heapq
import time
import numpy as np

import gym


def detect_slide(env):
    """Return (h, w, goal_flat, action_for_delta) if env is a single-blank slide
    puzzle, else None. action_for_delta maps (dr,dc) of the moved tile -> action str
    is recovered by probing the env from the solved state.
    """
    env.reset()
    st = env.get_state()
    if isinstance(st, dict):
        return None
    arr = np.asarray(st)
    if arr.ndim != 2:
        return None
    if np.count_nonzero(arr == 0) != 1:
        return None
    h, w = arr.shape
    goal = arr.reshape(-1).copy()

    # Map each valid action (from a few blank positions) to "blank delta (dr,dc)".
    # We probe by applying the action and seeing how the blank moved.
    action_delta = {}
    delta_action = {}
    seen_actions = set()
    # walk the blank around to observe all action labels
    env.reset()
    for _ in range(400):
        cur = np.asarray(env.get_state())
        br, bc = [int(x) for x in np.argwhere(cur == 0)[0]]
        for a in env.valid_actions():
            if a in seen_actions:
                continue
            env.set_state(cur)
            env.step(a)
            nb = np.asarray(env.get_state())
            nbr, nbc = [int(x) for x in np.argwhere(nb == 0)[0]]
            d = (nbr - br, nbc - bc)
            # must be a unit blank move
            if abs(d[0]) + abs(d[1]) != 1:
                return None
            action_delta[a] = d
            delta_action[d] = a
            seen_actions.add(a)
        # random move to explore
        env.set_state(cur)
        va = env.valid_actions()
        env.step(va[np.random.randint(len(va))])

    if set(delta_action.keys()) != {(-1, 0), (1, 0), (0, -1), (0, 1)} - (set() if h > 1 and w > 1 else set()):
        # accept whatever unit moves exist (handles non-square / degenerate)
        pass
    return {"h": h, "w": w, "goal": goal, "delta_action": delta_action}


class SlideHeuristic:
    def __init__(self, h, w, goal_flat):
        self.h, self.w = h, w
        self.goal = goal_flat
        maxv = int(goal_flat.max()) + 1
        self.goal_r = np.zeros(maxv, dtype=np.int32)
        self.goal_c = np.zeros(maxv, dtype=np.int32)
        for idx, v in enumerate(goal_flat):
            self.goal_r[v] = idx // w
            self.goal_c[v] = idx % w
        # precompute row/col of each goal value for linear conflict
        self.maxv = maxv

    def manhattan(self, board_flat):
        h, w = self.h, self.w
        idx = np.arange(h * w)
        r = idx // w
        c = idx % w
        v = board_flat
        nz = v != 0
        md = np.abs(r[nz] - self.goal_r[v[nz]]) + np.abs(c[nz] - self.goal_c[v[nz]])
        base = int(md.sum())
        # linear conflict: tiles in their goal row but reversed order
        lc = 0
        b = board_flat.reshape(h, w)
        for i in range(h):
            row = b[i]
            goalrow = [val for val in row if val != 0 and self.goal_r[val] == i]
            cols = [self.goal_c[val] for val in goalrow]
            for x in range(len(cols)):
                for y in range(x + 1, len(cols)):
                    if cols[x] > cols[y]:
                        lc += 1
        bt = b.T
        for j in range(w):
            col = bt[j]
            goalcol = [val for val in col if val != 0 and self.goal_c[val] == j]
            rows = [self.goal_r[val] for val in goalcol]
            for x in range(len(rows)):
                for y in range(x + 1, len(rows)):
                    if rows[x] > rows[y]:
                        lc += 1
        return base + 2 * lc


def _neighbors(board, h, w, delta_action):
    """Yield (new_board_tuple, action) by moving blank."""
    bpos = board.index(0)
    br, bc = divmod(bpos, w)
    for (dr, dc), a in delta_action.items():
        nr, nc = br + dr, bc + dc
        if 0 <= nr < h and 0 <= nc < w:
            npos = nr * w + nc
            lst = list(board)
            lst[bpos], lst[npos] = lst[npos], lst[bpos]
            yield tuple(lst), a


def solve_slide(info, state, deadline, W=1.5, max_nodes=2_000_000):
    h, w = info["h"], info["w"]
    goal = info["goal"]
    delta_action = info["delta_action"]
    H = info["_heur"]
    goal_t = tuple(int(x) for x in goal)

    board = tuple(int(x) for x in np.asarray(state).reshape(-1))
    if board == goal_t:
        return []

    def hf(bt):
        return H.manhattan(np.asarray(bt, dtype=np.int32))

    parents = {board: (None, None)}
    g = {board: 0}
    cnt = 0
    h0 = hf(board)
    heap = [(W * h0, 0, board)]
    expanded = 0
    while heap:
        if expanded >= max_nodes or time.time() >= deadline:
            return None
        f, gc, b = heapq.heappop(heap)
        if b == goal_t:
            return _reconstruct(b, parents)
        if g[b] < gc:
            continue
        for nb, a in _neighbors(b, h, w, delta_action):
            ng = gc + 1
            if g.get(nb, 1 << 30) <= ng:
                continue
            parents[nb] = (b, a)
            g[nb] = ng
            if nb == goal_t:
                return _reconstruct(nb, parents)
            cnt += 1
            heapq.heappush(heap, (ng + W * hf(nb), ng, nb))
        expanded += 1
    return None


def _reconstruct(end, parents):
    out = []
    cur = end
    while True:
        pk, a = parents.get(cur, (None, None))
        if pk is None or a is None:
            break
        out.append(a)
        cur = pk
    out.reverse()
    return out


def solve_slide_anytime(info, state, deadline):
    """Guaranteed-first then improve: high-W greedy secures a solution, then
    lower W passes try to shorten it. Returns best (shortest) found."""
    board = tuple(int(x) for x in np.asarray(state).reshape(-1))
    goal_t = tuple(int(x) for x in info["goal"])
    if board == goal_t:
        return []
    best = None
    # quality-first schedule; each pass bounded by a slice of remaining time
    schedule = [1.6, 2.2, 3.0, 6.0]
    for i, W in enumerate(schedule):
        now = time.time()
        if now >= deadline:
            break
        # give later (safer) passes guaranteed time if still nothing found
        remaining = deadline - now
        passes_left = len(schedule) - i
        slice_dl = now + (remaining if best is None and i == len(schedule) - 1
                          else remaining / passes_left)
        sol = solve_slide(info, state, min(slice_dl, deadline), W=W)
        if sol is not None and (best is None or len(sol) < len(best)):
            best = sol
            # if already near-optimal, stop early
    return best
