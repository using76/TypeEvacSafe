# -*- coding: utf-8 -*-
"""사용자 시험(2026-09-18): 30×30×3 m 홀, 문 2(남·북 중앙), 남문에서 −x 10·+y 10 에 2 MW 화원, 피난자 100명이 화원 주변에 밀집.
성인 남·여, 어린이(보호자 연결), 노약자, 부상자(보행 가능·불능)로 프로필을 주고, 판단층(규칙 / 27B / TypeSafe)이
모드·출구·구조를 정하며 이동은 SFM. 0.5 s 마다 전원 좌표를 CSV 로 낸다.
사용: python hall_test.py --macro rule,llama --out runs/hall_test [--macro_dt 30]
출력: <out>/<macro>_positions.csv (t, agent, role, sex, age, x, y, state, mode, target_exit), <macro>_traj.png, summary.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env2 import EvacEnv2, ROLES, MODES
from jev_decide import JevDecider

STATE = ["waiting", "moving", "down", "exited", "dead"]
COL = {"adult": "tab:blue", "child": "tab:cyan", "elderly": "tab:purple", "guardian": "tab:green", "firefighter": "tab:red", "injured": "tab:orange"}


def make_roster(seed=0):
    """100명: 보호자 10(남5 여5) + 성인 남 30·여 25 = 성인 65, 어린이 10(보호자 1:1), 노약자 15, 부상자 10(보행 5·불능 5) → 100, + 소방관 2."""
    rng = np.random.RandomState(seed); R = []
    def add(role, sex, age, **kw): R.append(dict(role=role, sex=sex, age=int(age), **kw))
    for _ in range(5): add("guardian", "male", rng.randint(28, 50))
    for _ in range(5): add("guardian", "female", rng.randint(28, 50))
    for k in range(10): add("child", "male" if k % 2 else "female", rng.randint(4, 11))
    for k in range(10): R[k]["guardian_of"] = 10 + k
    for _ in range(30): add("adult", "male", rng.randint(20, 60))
    for _ in range(25): add("adult", "female", rng.randint(20, 60))
    for k in range(15): add("elderly", "male" if k % 2 else "female", rng.randint(68, 88))
    for _ in range(5): add("injured", "male", rng.randint(25, 60))
    for _ in range(5): add("injured", "female", rng.randint(25, 60), cannot_walk=True)
    for _ in range(2): add("firefighter", "male", rng.randint(28, 45))
    assert len(R) == 102
    return R


def run(fields, case, macro, roster, seed, dev, macro_dt, premove, backend_kw, max_steps=2400, rec_every=5, spawn=(5.0, 10.0, 2.5, 9.0)):
    N = len(roster)
    env = EvacEnv2(fields, 1, N, device=dev, seed=seed, n_ff=2, macro="jev" if macro != "rule" else "rule",
                   case_filter=lambda n: n.startswith(case), macro_dt=macro_dt, premove=premove, roster=roster,
                   spawn=tuple(spawn))                                            # 군중 초기 배치(원환)
    if macro != "rule":
        env.decider = JevDecider(backend=macro, **backend_kw)
    obs = env.reset()
    rows = []; t0 = time.time(); traj = [env.pos.clone().cpu()]
    def rec():
        t, P, st, md, tg = env.snapshot()
        for i in range(N):
            r = roster[i]
            rows.append((round(t, 1), i, r["role"], r["sex"], r["age"], round(float(P[i, 0]), 2), round(float(P[i, 1]), 2), STATE[int(st[i])], MODES[int(md[i])], int(tg[i])))
    rec()
    for s in range(max_steps):
        obs, r, done, cdone, info = env.step(env.baseline_action())
        traj.append(env.pos.clone().cpu())
        if (s + 1) % rec_every == 0: rec()
        if cdone.all(): break
    st = env.stats(); st["wall_s"] = round(time.time() - t0, 1); st["steps"] = s + 1; st["t_end"] = round(float(env.t[0]), 1)
    st["decider_calls"] = len(env.decider.log) if macro != "rule" else 0
    st["decider_seconds"] = round(sum(l["seconds"] for l in env.decider.log), 1) if macro != "rule" else 0
    st["decisions"] = int(sum(l["n"] for l in env.decider.log)) if macro != "rule" else 0
    st["t_exit_p90"] = float(env.t_exit[env.exited].quantile(0.9)) if env.exited.any() else None
    st["exit_by_door"] = {"SOUTH": int((env.exited & (env.tgt_exit == 0)).sum()), "NORTH": int((env.exited & (env.tgt_exit == 1)).sum())}
    return env, rows, torch.stack(traj).numpy(), st


def plot(env, T, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    b = 0; N = env.N
    fig, ax = plt.subplots(figsize=(9, 9))
    ext = [float(env.x0[b]) - env.dx / 2, float(env.x0[b]) + env.NX * env.dx - env.dx / 2, float(env.y0[b]) - env.dx / 2, float(env.y0[b]) + env.NY * env.dx - env.dx / 2]
    ax.imshow(env.WALL[b].cpu().numpy().T, origin="lower", cmap="gray_r", extent=ext, alpha=0.45)
    fi = min(env.NT - 1, 120); KS = env.KS[b, fi].float().cpu().numpy().T
    ax.imshow(np.clip(KS, 0, 2), origin="lower", cmap="Greys", extent=ext, alpha=0.35, vmin=0, vmax=2)
    for i in range(N):
        r = ROLES[int(env.role[i])]; c = COL[r]
        ax.plot(T[:, i, 0], T[:, i, 1], color=c, lw=0.7 if r != "firefighter" else 1.5, alpha=0.7)
        end = "^" if env.exited[i] else ("x" if env.dead[i] else ("v" if env.down[i] else "s"))
        ax.plot(T[-1, i, 0], T[-1, i, 1], marker=end, color=c, ms=6, mec="k")
    ax.plot(float(env.fire_xy[b, 0]), float(env.fire_xy[b, 1]), "r*", ms=18)
    for r, c in COL.items(): ax.plot([], [], color=c, label=r)
    ax.plot([], [], "k^", label="exited"); ax.plot([], [], "kv", label="down"); ax.plot([], [], "kx", label="dead")
    ax.legend(fontsize=7, loc="upper right"); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="/home/work/BULC_DATA/evac_big/fields"); ap.add_argument("--case", default="HALL_30x30_2MW"); ap.add_argument("--agents", type=int, default=0, help="0 이면 명단 전체(102)")
    ap.add_argument("--macro", default="rule,llama"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--macro_dt", type=float, default=30.0); ap.add_argument("--premove", type=float, default=15.0)
    ap.add_argument("--out", default="runs/hall_test"); ap.add_argument("--url", default="http://127.0.0.1:8081"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--spawn", default="5,10,2.5,9")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roster = make_roster(a.seed)
    if a.agents and a.agents < len(roster):                       # 인원 축소: 보호자·아이 쌍과 부상자는 유지
        keep = list(range(20)) + list(range(len(roster) - 12, len(roster)))
        keep += [i for i in range(20, len(roster) - 12)][: max(0, a.agents - len(keep))]
        keep = sorted(set(keep))[:a.agents]
        idx = {o: n for n, o in enumerate(keep)}
        roster = [dict(roster[o]) for o in keep]
        for r in roster:
            if r.get("guardian_of") is not None: r["guardian_of"] = idx.get(r["guardian_of"])
        roster = [{k: v for k, v in r.items() if not (k == "guardian_of" and v is None)} for r in roster]
    json.dump(roster, open(os.path.join(a.out, "roster.json"), "w"), indent=1)
    summary = {}
    for macro in a.macro.split(","):
        kw = dict(url=a.url, threads=a.threads) if macro == "llama" else (dict(threads=a.threads) if macro == "typesafe" else {})
        env, rows, T, st = run(a.fields, a.case, macro, roster, a.seed, dev, a.macro_dt, a.premove, kw,
                               spawn=[float(v) for v in a.spawn.split(",")])
        with open(os.path.join(a.out, "%s_positions.csv" % macro), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "agent", "role", "sex", "age", "x", "y", "state", "mode", "target_exit"]); w.writerows(rows)
        plot(env, T, os.path.join(a.out, "%s_traj.png" % macro), "%s · %s · exit %.2f down %.2f dead %.2f rescued %d" % (a.case, macro, st["exit_all"], st["down_all"], st["dead_all"], st["rescued"]))
        if macro != "rule":
            with open(os.path.join(a.out, "%s_decisions.jsonl" % macro), "w") as f:
                for l in env.decider.log: f.write(json.dumps(l) + "\n")
        summary[macro] = st
        print("[%s] %s" % (macro, json.dumps(st, ensure_ascii=False)), flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
