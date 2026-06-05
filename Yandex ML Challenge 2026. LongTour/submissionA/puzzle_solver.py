"""Family dispatcher: detect the hidden puzzle's structure once, then route each
instance to the strongest applicable solver.

Families
  gf2     : GF(2)-linear toggle puzzles  -> exact min-weight solve (Gaussian elim)
  slide   : single-blank sliding puzzles -> fast Manhattan+LC weighted A* (anytime)
  generic : everything else              -> weighted A* with geometric heuristic,
                                            escalating W (guaranteed-first fallback)

All detection is domain-general (probes the unified encoding / action descriptors),
so it transfers to unseen hidden games sharing the same protocol.
"""

import os
import time
import numpy as np

import gym
import common
import linear_solver as LIN
import slide_solver as SL
import search_v2 as SV


class Solver:
    def __init__(self, model_path=None, max_nodes=200_000, max_unique=200_000):
        self.env = gym.make_env()
        self.env.reset()
        self.solved_key = SV.fast_key(self.env.get_state())
        self.family = None
        self.info = None
        self.heur = None
        self.v_fn = None
        self.max_nodes = max_nodes
        self.max_unique = max_unique
        # number of cells (state size proxy for memory budgeting)
        try:
            self.N = len(self.env.encode_state(self.env.get_state())["content_values"])
        except Exception:
            self.N = 64

        # 1) GF(2)-linear (toggle) family
        try:
            gf2 = LIN.detect_linear_gf2(self.env)
        except Exception:
            gf2 = None
        if gf2 is not None:
            self.family = "gf2"
            self.info = gf2
            return

        # 2) single-blank slide family
        self.env.reset()
        try:
            slide = SL.detect_slide(self.env)
        except Exception:
            slide = None
        if slide is not None:
            self.family = "slide"
            slide["_heur"] = SL.SlideHeuristic(slide["h"], slide["w"], slide["goal"])
            self.info = slide
            return

        # 3) generic permutation: geometric heuristic (+ optional learned V)
        self.env.reset()
        self.family = "generic"
        self.heur = SV.Heuristic(self.env)
        self.h_fn = self.heur.h_fast   # fast raw-mismatch -> ~10x more nodes/sec
        if model_path and os.path.exists(model_path):
            self._load_nn(model_path)

    def _load_nn(self, model_path):
        try:
            import torch
            from model import ValueNet
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
            model = ValueNet()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            torch.set_num_threads(1)
            env = self.env

            def v_fn(state):
                tk = common.encode_tokens(env, state)[None]
                B, N, _ = tk.shape
                p = common.split_token_features(tk.reshape(B * N, -1))
                d = torch.from_numpy(p["dense"].reshape(B, N, -1))
                cv = torch.from_numpy(p["content_value"].reshape(B, N))
                tv = torch.from_numpy(p["target_value"].reshape(B, N))
                with torch.no_grad():
                    return float(model(d, cv, tv)[0])
            # blend learned V with free geo (max -> stronger, still cheap-ish)
            geo = self.heur.h_geo
            self.h_fn = lambda s: max(v_fn(s), geo(s))
        except Exception:
            pass  # fall back to geo silently

    def solve_one(self, state, deadline):
        if self.family == "gf2":
            return LIN.solve_linear(self.env, state, self.info) or []
        if self.family == "slide":
            return SL.solve_slide_anytime(self.info, state, deadline) or []
        # generic
        sol = SV.solve_perm(self.env, state, self.solved_key, self.h_fn,
                            deadline, max_nodes=self.max_nodes,
                            max_unique=self.max_unique)
        return sol if isinstance(sol, list) else []
