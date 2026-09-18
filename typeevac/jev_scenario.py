# -*- coding: utf-8 -*-
"""시나리오 시험 — 규칙 매크로 vs Jev 판단층(TypeSafe / 27B)로 같은 케이스·같은 배치를 돌려 역할별 경로·모드·결과를 비교.

저수준 이동은 기준선(rust_evac nav)으로 고정한다 — 판단층이 바꾸는 것은 목표 출구·모드·구조 대상뿐이므로
"에이전트별 역할과 이동 경로가 의도대로 바뀌는가"를 판단층 효과만으로 본다.
사용: python jev_scenario.py --case TC_home1_s10_near_exit_f --agents 32 --macro rule,typesafe --out runs/scn_home1
출력: <out>/<macro>_traj.png(역할별 궤적·모드 변화점), <out>/<macro>_events.jsonl, <out>/summary.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env2 import EvacEnv2, ROLES, MODES, R_CHILD
from jev_decide import JevDecider

COL = {"adult": "tab:blue", "child": "tab:cyan", "elderly": "tab:purple", "guardian": "tab:green", "firefighter": "tab:red"}


def run_one(fields, case, macro, agents, seed, dev, time_scale, max_steps, backend_kw, macro_dt=5.0, premove=5.0):
    env = EvacEnv2(fields, 1, agents, device=dev, seed=seed, n_ff=2, time_scale=time_scale, macro="jev" if macro != "rule" else "rule",
                   roles=dict(adult=0.5, child=5 / 32, elderly=5 / 32, guardian=5 / 32), case_filter=lambda n: n.startswith(case), macro_dt=macro_dt, premove=premove)
    if macro != "rule":
        env.decider = JevDecider(backend=macro, **backend_kw)
    obs = env.reset()
    traj, modes, tgts = [env.pos.clone().cpu()], [env.mode.clone().cpu()], [env.tgt_exit.clone().cpu()]
    events = []; prev = dict(exited=env.exited.clone(), down=env.down.clone(), rescued=env.rescued.clone(), attached=env.attached.clone(), mode=env.mode.clone(), tgt=env.tgt_exit.clone())
    t0 = time.time()
    for s in range(max_steps):
        obs, r, done, cdone, info = env.step(env.baseline_action())
        traj.append(env.pos.clone().cpu()); modes.append(env.mode.clone().cpu()); tgts.append(env.tgt_exit.clone().cpu())
        t = float(env.t[0])
        for k, cur in (("exited", env.exited), ("down", env.down), ("rescued", env.rescued)):
            new = cur & ~prev[k]
            for i in torch.nonzero(new).flatten().tolist():
                events.append(dict(t=round(t, 1), agent=i, role=ROLES[int(env.role[i])], event=k)); prev[k] = cur.clone()
        att = (env.attached >= 0) & (prev["attached"] < 0)
        for i in torch.nonzero(att).flatten().tolist():
            events.append(dict(t=round(t, 1), agent=i, role=ROLES[int(env.role[i])], event="attached_by", by=int(env.attached[i]), by_role=ROLES[int(env.role[int(env.attached[i])])]))
        prev["attached"] = env.attached.clone()
        ch = (env.mode != prev["mode"]) & env.active & ~env.done
        for i in torch.nonzero(ch).flatten().tolist():
            events.append(dict(t=round(t, 1), agent=i, role=ROLES[int(env.role[i])], event="mode", mode=MODES[int(env.mode[i])]))
        prev["mode"] = env.mode.clone()
        ch = (env.tgt_exit != prev["tgt"]) & env.active & ~env.done
        for i in torch.nonzero(ch).flatten().tolist():
            events.append(dict(t=round(t, 1), agent=i, role=ROLES[int(env.role[i])], event="target_exit", k=int(env.tgt_exit[i])))
        prev["tgt"] = env.tgt_exit.clone()
        if cdone.all(): break
    st = env.stats(); st["wall_s"] = round(time.time() - t0, 1); st["steps"] = s + 1; st["t_end"] = round(float(env.t[0]), 1)
    st["decider_calls"] = len(env.decider.log) if macro != "rule" else 0
    st["decider_seconds"] = round(sum(l["seconds"] for l in env.decider.log), 1) if macro != "rule" else 0
    return env, torch.stack(traj).numpy(), torch.stack(modes).numpy(), torch.stack(tgts).numpy(), events, st


def plot(env, T, M, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    b = 0; N = env.N
    fig, ax = plt.subplots(figsize=(11, 8))
    ext = [float(env.x0[b]) - env.dx / 2, float(env.x0[b]) + env.NX * env.dx - env.dx / 2, float(env.y0[b]) - env.dx / 2, float(env.y0[b]) + env.NY * env.dx - env.dx / 2]
    ax.imshow(env.WALL[b].cpu().numpy().T, origin="lower", cmap="gray_r", extent=ext, alpha=0.45)
    fi = min(env.NT - 1, int(env.NT * 0.6)); KS = env.KS[b, fi].float().cpu().numpy().T
    ax.imshow(np.clip(KS, 0, 2), origin="lower", cmap="Greys", extent=ext, alpha=0.35, vmin=0, vmax=2)
    for i in range(N):
        r = ROLES[int(env.role[i])]; c = COL[r]
        ax.plot(T[:, i, 0], T[:, i, 1], color=c, lw=0.9 if r != "firefighter" else 1.6, alpha=0.85)
        ax.plot(T[0, i, 0], T[0, i, 1], marker="o", color=c, ms=3)
        end = "^" if env.exited[i] else ("x" if env.down[i] or env.dead[i] else "s")
        ax.plot(T[-1, i, 0], T[-1, i, 1], marker=end, color=c, ms=6, mec="k")
        ch = np.nonzero(np.diff(M[:, i]))[0]
        for s in ch:
            ax.plot(T[s + 1, i, 0], T[s + 1, i, 1], marker="*", color="k", ms=5, alpha=0.6)
    ax.plot(float(env.fire_xy[b, 0]), float(env.fire_xy[b, 1]), "r*", ms=16)
    for e in json.loads(str(np.load(env.files[0])["exits_json"])) if hasattr(env, "files") else []:
        pass
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    for r, c in COL.items():
        ax.plot([], [], color=c, label=r)
    ax.plot([], [], "k*", label="mode change"); ax.plot([], [], "k^", label="exited"); ax.plot([], [], "kx", label="down")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout(); plt.savefig(out_png, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="/home/work/BULC_DATA/evac_big/fields")
    ap.add_argument("--case", default="TC_home1_s10_near_exit_f"); ap.add_argument("--agents", type=int, default=32)
    ap.add_argument("--macro", default="rule,typesafe"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_scale", type=float, default=1.0); ap.add_argument("--max_steps", type=int, default=2400)
    ap.add_argument("--out", default="runs/scn"); ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--macro_dt", type=float, default=5.0, help="판단층 호출 주기(s) — 27B 는 15 권장")
    ap.add_argument("--premove", type=float, default=5.0, help="출발 지연 상한(s)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    for macro in a.macro.split(","):
        kw = dict(url=a.url) if macro == "llama" else {}
        env, T, M, G, events, st = run_one(a.fields, a.case, macro, a.agents, a.seed, dev, a.time_scale, a.max_steps, kw, a.macro_dt if macro != "rule" else 5.0, a.premove)
        summary[macro] = st
        with open(os.path.join(a.out, "%s_events.jsonl" % macro), "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        plot(env, T, M, os.path.join(a.out, "%s_traj.png" % macro), "%s · %s · exit %.2f down %.2f dead %.2f rescued %d" % (
            a.case, macro, st["exit_all"], st["down_all"], st["dead_all"], st["rescued"]))
        # 역할별 모드 점유율
        act = M[1:]; occ = {}
        for r in range(5):
            m = (env.role.cpu().numpy() == r)
            if m.any():
                cnt = np.bincount(act[:, m].flatten(), minlength=len(MODES)); occ[ROLES[r]] = {MODES[k]: round(float(cnt[k] / cnt.sum()), 3) for k in range(len(MODES)) if cnt[k]}
        st["mode_share"] = occ
        st["n_mode_changes"] = int(sum(1 for e in events if e["event"] == "mode")); st["n_exit_switch"] = int(sum(1 for e in events if e["event"] == "target_exit"))
        print("[%s] %s" % (macro, json.dumps({k: v for k, v in st.items() if k not in ("mode_share",)}, ensure_ascii=False)))
        print("    mode_share:", json.dumps(occ))
        if macro != "rule":
            with open(os.path.join(a.out, "%s_decisions.jsonl" % macro), "w") as f:
                for l in env.decider.log: f.write(json.dumps(l) + "\n")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
