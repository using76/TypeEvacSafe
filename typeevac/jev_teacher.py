# -*- coding: utf-8 -*-
"""교사 데이터 생성 — TypeSafe Jev(교사)로 시뮬레이션을 굴리며 결정점마다 (상황, 질문, 교사 확률)을 기록한다.
학생(JEV-EVAC, Qwen 기반)은 이 파일로 증류 SFT 한다. 결과 보정(2단계)용으로 에피소드 종료 후 각 에이전트의 결말도 붙인다.

사용: python jev_teacher.py --cases 40 --agents 40 --out /home/work/BULC_DATA/evac_big/teacher --threads 16
출력: <out>/dec_<stamp>.jsonl  한 줄 = 한 결정
  {case, seed, t, agent, role, sex, age, evidence, questions{head:[(id,desc)]}, probs{head:{id:p}}, choice{head:id},
   outcome{exited, down, dead, rescued, t_exit, fed_final}}
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env2 import EvacEnv2, ROLES, MODES
from jev_decide import JevDecider, build_evidence, build_questions


def make_roster(n, rng):
    """대공간 일반 구성: 보호자+아이 쌍 15%, 노약자 15%, 부상자 8%(그중 절반 보행 불능), 나머지 성인, 소방관 2."""
    n_pair = max(1, int(n * 0.15)); n_eld = int(n * 0.15); n_inj = max(2, int(n * 0.08))
    R = []
    for _ in range(n_pair): R.append(dict(role="guardian", sex=rng.choice(["male", "female"]), age=int(rng.integers(28, 50))))
    for k in range(n_pair): R.append(dict(role="child", sex=rng.choice(["male", "female"]), age=int(rng.integers(4, 12))))
    for k in range(n_pair): R[k]["guardian_of"] = n_pair + k
    for _ in range(n_eld): R.append(dict(role="elderly", sex=rng.choice(["male", "female"]), age=int(rng.integers(66, 88))))
    for j in range(n_inj):
        R.append(dict(role="injured", sex=rng.choice(["male", "female"]), age=int(rng.integers(20, 65)), cannot_walk=bool(j % 2)))
    while len(R) < n - 2:
        R.append(dict(role="adult", sex=rng.choice(["male", "female"]), age=int(rng.integers(19, 65))))
    for _ in range(2): R.append(dict(role="firefighter", sex="male", age=int(rng.integers(28, 45))))
    return R[:n]


class RecordingDecider(JevDecider):
    """decide() 를 감싸 상황·질문·교사 확률을 rows 에 쌓는다."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []

    def decide(self, env, idx):
        specs, items = [], []
        for i in idx:
            ev, exits, vic = build_evidence(env, i)
            qs = build_questions(ev, exits, vic, ev["me"]["role"])
            items.append((ev, qs)); specs.append((i, ev, exits, vic, qs))
        t0 = time.time(); answers = self.ask(items); dt = time.time() - t0
        out = []
        for (i, ev, exits, vic, qs), pr in zip(specs, answers):
            res = dict(agent=int(i), probs=pr)
            for h in qs:
                res[h] = max(pr[h], key=pr[h].get)
            res["target_exit_k"] = None
            if "target_exit" in res and res["mode"] in ("evacuate", "breakthrough", "follow_crowd", "escort"):
                res["target_exit_k"] = int(res["target_exit"][1:]) - 1
            elif len(exits) == 1:
                res["target_exit_k"] = 0
            res["rescue_agent"] = -1
            if res["mode"] == "rescue" and (not vic or res.get("rescue_target", "none") == "none" or res.get("rescue_feasible") == "no"):
                res["mode"] = "evacuate"
                if "target_exit" in res: res["target_exit_k"] = int(res["target_exit"][1:]) - 1
                elif len(exits) == 1: res["target_exit_k"] = 0
            if res["mode"] == "rescue":
                res["rescue_agent"] = int(next(v["agent"] for v in vic if v["id"] == res["rescue_target"]))
            res["p_survive"] = pr["survive"].get("likely", 0.5)
            res["pace_mult"] = {"run": 1.4, "walk": 1.0, "slow": 0.6, "stop": 0.0}.get(res.get("pace", "walk"), 1.0)
            out.append(res)
            self.rows.append(dict(t=round(float(env.t[i]), 1), agent=int(i), role=ROLES[int(env.role[i])],
                                  evidence=ev, questions={h: qs[h][1] for h in qs},
                                  probs={h: {k: round(v, 4) for k, v in pr[h].items()} for h in pr},
                                  choice={h: res[h] for h in qs}))
        self.log.append(dict(n=len(idx), t=float(env.t[idx[0]]) if idx else 0, seconds=round(dt, 2)))
        return out


def run_case(case, agents, seed, dev, fields, macro_dt, premove, threads, max_steps=2000):
    rng = np.random.default_rng(seed)
    roster = make_roster(agents, rng)
    env = EvacEnv2(fields, 1, agents, device=dev, seed=seed, n_ff=2, macro="jev", macro_dt=macro_dt, premove=premove,
                   roster=roster, case_filter=lambda n: n == case + ".npz" or n.startswith(case))
    dec = RecordingDecider(backend="typesafe", threads=threads)
    env.decider = dec
    obs = env.reset()
    for s in range(max_steps):
        obs, r, done, cdone, info = env.step(env.baseline_action())
        if cdone.all(): break
    st = env.stats()
    out = dict(exited=env.exited.cpu().numpy(), down=env.down.cpu().numpy(), dead=env.dead.cpu().numpy(),
               rescued=env.rescued.cpu().numpy(), t_exit=env.t_exit.cpu().numpy(),
               fed=(env.fed_tox + env.fed_heat).cpu().numpy())
    rows = []
    for r_ in dec.rows:
        i = r_["agent"]
        r_["case"] = case; r_["seed"] = seed
        r_["profile"] = roster[i]
        r_["outcome"] = dict(exited=bool(out["exited"][i]), down=bool(out["down"][i]), dead=bool(out["dead"][i]),
                             rescued=bool(out["rescued"][i]), t_exit=float(out["t_exit"][i]), fed_final=float(out["fed"][i]))
        rows.append(r_)
    return rows, st, sum(l["seconds"] for l in dec.log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="/home/work/BULC_DATA/evac_big/fields")
    ap.add_argument("--out", default="/home/work/BULC_DATA/evac_big/teacher")
    ap.add_argument("--cases", type=int, default=40, help="샘플할 케이스 수")
    ap.add_argument("--agents", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--macro_dt", type=float, default=15.0); ap.add_argument("--premove", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--prefix", default=None)
    ap.add_argument("--max_cells", type=int, default=250000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.fields, "*.npz")))
    if a.prefix:
        pre = tuple(a.prefix.split(",")); files = [f for f in files if f.startswith(pre)]
    keep = []
    for f in files:
        z = np.load(os.path.join(a.fields, f + ".npz"))
        if int(z["WALL"].size) <= a.max_cells: keep.append(f)
        z.close()
    rng = random.Random(a.seed); rng.shuffle(keep)
    picks = keep[:a.cases]
    stamp = time.strftime("%m%d_%H%M")
    path = os.path.join(a.out, "dec_%s.jsonl" % stamp)
    n_dec = 0; t0 = time.time(); api_s = 0.0
    with open(path, "w") as f:
        for k, c in enumerate(picks):
            try:
                rows, st, sec = run_case(c, a.agents, a.seed * 1000 + k, dev, a.fields, a.macro_dt, a.premove, a.threads)
            except Exception as e:
                print("[fail] %s: %s" % (c, str(e)[:160]), flush=True); continue
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush(); n_dec += len(rows); api_s += sec
            print("[%d/%d] %-34s 결정 %4d (누적 %6d) exit %.2f dead %.2f resc %d | API %.0fs 총 %.0fs" % (
                k + 1, len(picks), c[:34], len(rows), n_dec, st["exit_all"], st["dead_all"], st["rescued"], sec, time.time() - t0), flush=True)
    print("완료: 결정 %d개 → %s (API %.0f s, 전체 %.0f s)" % (n_dec, path, api_s, time.time() - t0))


if __name__ == "__main__":
    main()
