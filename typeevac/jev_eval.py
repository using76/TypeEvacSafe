# -*- coding: utf-8 -*-
"""행동 평가 — "몇 명 나갔나"가 아니라 "스스로 상황을 읽고 경로를 짰나, 각자 역할을 했나"를 잰다.

사용자 기준(2026-09-18): 탈출률은 포화되어 판정력이 없다. 아래 네 축으로 판단층을 비교한다.
  A 상황 인지  : 위험이 바뀔 때 결정이 바뀌는가(반응 지연), 위험한 출구·경로를 피하는가, 근거 없이 흔들리지 않는가
  B 경로 계획  : 실제 경로가 대안보다 덜 위험했는가(노출 후회), 우회율, 목표 변경의 정당성
  C 역할 수행  : 보호자-피보호자 동행·동반 탈출, 소방관 역주행·접근·구조, 성인의 조건부 구조, 약자의 추종
  D 자율성     : 같은 케이스에서 개인 상태에 따라 결정이 갈리는가(획일적이지 않은가)
입력: hall_test/jev_scenario 의 <macro>_positions.csv, 선택 <macro>_decisions.jsonl, roster.json, 위험장 npz.
사용: python jev_eval.py --run runs/hall_test --macro rule,llama,typesafe --field .../HALL_30x30_2MW.npz
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

SMOKE_BAD, HEAT_BAD = 0.5, 60.0          # 이 이상이면 '위험이 바뀌었다'고 본다


def load_positions(path):
    by_agent = defaultdict(list); by_t = defaultdict(list)
    for r in csv.DictReader(open(path)):
        r["t"] = round(float(r["t"]), 1); r["x"] = float(r["x"]); r["y"] = float(r["y"])
        r["agent"] = int(r["agent"]); r["target_exit"] = int(r["target_exit"])
        by_agent[r["agent"]].append(r); by_t[r["t"]].append(r)
    for v in by_agent.values(): v.sort(key=lambda z: z["t"])
    return by_agent, by_t


class Field:
    def __init__(self, npz):
        z = np.load(npz)
        self.dx = float(z["dx"]); self.x0 = float(z["x0"]); self.y0 = float(z["y0"])
        self.KS = z["KS"].astype(np.float32); self.T = z["T"].astype(np.float32)
        self.DIST_E = z["DIST_E"].astype(np.float32); self.WALL = z["WALL"]
        self.exits = json.loads(str(z["exits_json"]))
        self.nx, self.ny = self.WALL.shape

    def cell(self, x, y):
        return (int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1)),
                int(np.clip(round((y - self.y0) / self.dx), 0, self.ny - 1)))

    def hazard(self, t, x, y):
        i, j = self.cell(x, y); k = min(self.KS.shape[0] - 1, int(round(t)))
        return float(self.KS[k, i, j]), float(self.T[k, i, j])

    def dist(self, k, x, y):
        i, j = self.cell(x, y); d = float(self.DIST_E[k, i, j])
        return d if math.isfinite(d) else None

    def route_exposure(self, k, x, y, t0, v=1.0, step=1.0):
        """출구 k 로 내리막을 따라갔을 때 앞으로 받을 노출(∫K_s dt 근사) — 대안 비교용."""
        i, j = self.cell(x, y); acc = 0.0; tt = t0
        for _ in range(400):
            d = float(self.DIST_E[k, i, j])
            if not math.isfinite(d) or d < self.dx: break
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and not self.WALL[a, b]:
                        dd = float(self.DIST_E[k, a, b])
                        if math.isfinite(dd) and (best is None or dd < best[0]): best = (dd, a, b)
            if best is None or best[0] >= d: break
            i, j = best[1], best[2]
            kk = min(self.KS.shape[0] - 1, int(round(tt)))
            acc += float(self.KS[kk, i, j]) * (self.dx / max(v, 0.2))
            tt += self.dx / max(v, 0.2)
        return acc


def eval_macro(run, macro, fld, roster):
    pos_path = os.path.join(run, "%s_positions.csv" % macro)
    if not os.path.exists(pos_path): return None
    by_agent, by_t = load_positions(pos_path)
    dec_path = os.path.join(run, "%s_decisions.jsonl" % macro)
    decs = [json.loads(l) for l in open(dec_path)] if os.path.exists(dec_path) else []
    times = sorted(by_t)
    N = len(by_agent)
    last = {i: rows[-1]["state"] for i, rows in by_agent.items()}
    out = {"n_agents": N, "t_end": times[-1],
           "R_exit": round(sum(v == "exited" for v in last.values()) / N, 3),
           "R_dead": round(sum(v == "dead" for v in last.values()) / N, 3),
           "R_stuck": round(sum(v in ("moving", "waiting", "down") for v in last.values()) / N, 3)}

    # ── A 상황 인지 ────────────────────────────────────────────────────────────
    # A1 위험 반응: 개인 국소 위험(K_s>0.5 또는 T>60)이 처음 닥친 시각 → 그 뒤 모드/목표 변경까지 지연
    lat = []; reacted = 0; exposed = 0
    for i, rows in by_agent.items():
        t_haz = None; t_chg = None; prev = (rows[0]["mode"], rows[0]["target_exit"])
        for r in rows:
            if r["state"] in ("exited", "dead"): break
            ks, T = fld.hazard(r["t"], r["x"], r["y"])
            if t_haz is None and (ks > SMOKE_BAD or T > HEAT_BAD): t_haz = r["t"]
            cur = (r["mode"], r["target_exit"])
            if cur != prev:
                prev = cur
                if t_haz is not None and t_chg is None: t_chg = r["t"]
        if t_haz is not None:
            exposed += 1
            if t_chg is not None: lat.append(t_chg - t_haz); reacted += 1
    out["A1_hazard_reaction_rate"] = round(reacted / exposed, 3) if exposed else None
    out["A1_reaction_lag_s"] = round(float(np.median(lat)), 1) if lat else None
    out["A1_exposed_agents"] = exposed

    # A2 위험 출구 회피: 탈출 시점에 자기가 쓴 출구 vs 대안 출구의 '앞으로 받을 노출'
    #    (음수 = 더 나쁜 출구를 골랐다, 0 = 최선)
    reg = []; chose_worse = 0; chose_n = 0
    for i, rows in by_agent.items():
        start = rows[0]
        ks_list = []
        for k in range(len(fld.exits)):
            if fld.dist(k, start["x"], start["y"]) is None: continue
            ks_list.append((k, fld.route_exposure(k, start["x"], start["y"], start["t"])))
        if len(ks_list) < 2: continue
        chosen = start["target_exit"]
        mine = dict(ks_list).get(chosen)
        best = min(v for _, v in ks_list)
        if mine is None: continue
        chose_n += 1; reg.append(mine - best)
        if mine > best + 0.5: chose_worse += 1
    out["A2_exposure_regret_mean"] = round(float(np.mean(reg)), 2) if reg else None
    out["A2_worse_exit_rate"] = round(chose_worse / chose_n, 3) if chose_n else None

    # A3 목표 변경의 질: 바꾼 출구가 그 시점 노출 기준으로 더 나았는가(개선 스위치 비율)
    sw = 0; sw_good = 0
    for i, rows in by_agent.items():
        for a, b in zip(rows, rows[1:]):
            if a["target_exit"] != b["target_exit"] and b["state"] not in ("exited", "dead"):
                sw += 1
                e_old = fld.route_exposure(a["target_exit"], b["x"], b["y"], b["t"])
                e_new = fld.route_exposure(b["target_exit"], b["x"], b["y"], b["t"])
                if e_new < e_old - 0.05: sw_good += 1
    out["A3_switches"] = sw; out["A3_improving_switch_rate"] = round(sw_good / sw, 3) if sw else None

    # A4 위험 출구 회피: 목표 출구가 '경로 노출 상위'인데 더 나은 대안이 있을 때 얼마나 자주 갈아탔나(5 s 간격 표본)
    chance = 0; taken = 0
    for i, rows in by_agent.items():
        for r in rows[::10]:
            if r["state"] not in ("moving", "waiting"): continue
            cand = [(k, fld.route_exposure(k, r["x"], r["y"], r["t"])) for k in range(len(fld.exits))
                    if fld.dist(k, r["x"], r["y"]) is not None]
            if len(cand) < 2: continue
            mine = dict(cand).get(r["target_exit"]); best_k, best = min(cand, key=lambda z: z[1])
            if mine is None or mine <= best + 1.0: continue
            chance += 1
            j = rows.index(r)
            if any(x["target_exit"] == best_k for x in rows[j:j + 40]): taken += 1        # 이후 20 s 안에 갈아탐
    out["A4_bad_exit_situations"] = chance
    out["A4_switch_to_better_rate"] = round(taken / chance, 3) if chance else None

    # ── B 경로 계획 ────────────────────────────────────────────────────────────
    det = []; expo = []
    for i, rows in by_agent.items():
        moved = [r for r in rows if r["state"] in ("moving", "waiting")]
        if len(moved) < 3: continue
        L = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(moved, moved[1:]))
        d0 = fld.dist(moved[0]["target_exit"], moved[0]["x"], moved[0]["y"])
        if d0 and d0 > 1.0: det.append(L / d0)
        e = 0.0
        for a, b in zip(moved, moved[1:]):
            ks, _ = fld.hazard(a["t"], a["x"], a["y"]); e += ks * (b["t"] - a["t"])
        expo.append(e)
    out["B1_detour_ratio_p50"] = round(float(np.median(det)), 2) if det else None
    out["B2_smoke_exposure_p50"] = round(float(np.median(expo)), 2) if expo else None
    out["B2_smoke_exposure_p90"] = round(float(np.percentile(expo, 90)), 2) if expo else None

    # ── C 역할 수행 ────────────────────────────────────────────────────────────
    pairs = [(i, r["guardian_of"]) for i, r in enumerate(roster) if r.get("guardian_of") is not None] if roster else []
    sep = []; co_exit = 0; dep_ok = 0
    for g, c in pairs:
        gr = {r["t"]: r for r in by_agent.get(g, [])}; cr = {r["t"]: r for r in by_agent.get(c, [])}
        ts = [t for t in gr if t in cr and gr[t]["state"] in ("moving", "waiting")]
        if ts:
            d = [math.hypot(gr[t]["x"] - cr[t]["x"], gr[t]["y"] - cr[t]["y"]) for t in ts]
            sep.append(float(np.mean([x > 3.0 for x in d])))
        ge = next((r["t"] for r in by_agent.get(g, []) if r["state"] == "exited"), None)
        ce = next((r["t"] for r in by_agent.get(c, []) if r["state"] == "exited"), None)
        if ce is not None: dep_ok += 1
        if ge is not None and ce is not None and abs(ge - ce) <= 10.0: co_exit += 1
    out["C1_guardian_pairs"] = len(pairs)
    out["C1_separation_time_frac"] = round(float(np.mean(sep)), 3) if sep else None
    out["C1_dependent_exit_rate"] = round(dep_ok / len(pairs), 3) if pairs else None
    out["C1_co_exit_rate"] = round(co_exit / len(pairs), 3) if pairs else None

    ff = [i for i, r in enumerate(roster or []) if r["role"] == "firefighter"]
    back = []; inside = []
    for i in ff:
        rows = by_agent.get(i, [])
        act = [r for r in rows if r["state"] in ("moving", "waiting")]
        if not act: continue
        d = [fld.dist(0, r["x"], r["y"]) for r in act]
        d = [x for x in d if x is not None]
        if len(d) > 3: back.append(round(max(d) - d[0], 1))       # 출구에서 멀어진 최대 깊이(역주행)
        inside.append(act[-1]["t"] - act[0]["t"])
    out["C2_ff_penetration_m"] = round(float(np.mean(back)), 1) if back else None
    out["C2_ff_inside_s"] = round(float(np.mean(inside)), 1) if inside else None

    inj = [i for i, r in enumerate(roster or []) if r["role"] == "injured"]
    imm = [i for i, r in enumerate(roster or []) if r.get("cannot_walk")]
    def final(i, key):
        rows = by_agent.get(i, [])
        return rows[-1]["state"] == key if rows else False
    out["C3_injured_exit_rate"] = round(float(np.mean([final(i, "exited") for i in inj])), 3) if inj else None
    out["C3_immobile_exit_rate"] = round(float(np.mean([final(i, "exited") for i in imm])), 3) if imm else None
    out["C3_immobile_dead_rate"] = round(float(np.mean([final(i, "dead") for i in imm])), 3) if imm else None

    # ── D 자율성(획일성의 반대) ────────────────────────────────────────────────
    ent_t = []
    for t in times[::10]:
        rows = [r for r in by_t[t] if r["state"] in ("moving", "waiting")]
        if len(rows) < 8: continue
        for key in ("mode", "target_exit"):
            c = defaultdict(int)
            for r in rows: c[r[key]] += 1
            p = np.array(list(c.values()), float); p /= p.sum()
            ent_t.append((key, float(-(p * np.log(p + 1e-12)).sum())))
    for key in ("mode", "target_exit"):
        v = [e for k, e in ent_t if k == key]
        out["D1_entropy_%s" % key] = round(float(np.mean(v)), 3) if v else None
    if decs:
        n = sum(l["n"] for l in decs); sec = sum(l["seconds"] for l in decs)
        out["decisions"] = n; out["decision_s"] = round(sec / max(n, 1), 3)
        modes = defaultdict(int)
        for l in decs:
            for d in l["decisions"]: modes[d["mode"]] += 1
        out["mode_counts"] = dict(modes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--macro", default="rule,llama,typesafe")
    ap.add_argument("--field", default="/home/work/BULC_DATA/evac_big/fields/HALL_30x30_2MW.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fld = Field(a.field)
    rp = os.path.join(a.run, "roster.json")
    roster = json.load(open(rp)) if os.path.exists(rp) else None
    res = {}
    for m in a.macro.split(","):
        r = eval_macro(a.run, m, fld, roster)
        if r: res[m] = r
    keys = sorted({k for v in res.values() for k in v if not isinstance(v[k], dict)})
    print("%-32s %s" % ("지표", "  ".join("%14s" % m for m in res)))
    for k in keys:
        print("%-32s %s" % (k, "  ".join("%14s" % ("-" if res[m].get(k) is None else res[m][k]) for m in res)))
    for m in res:
        if "mode_counts" in res[m]: print("%-32s %s: %s" % ("mode_counts", m, res[m]["mode_counts"]))
    json.dump(res, open(a.out or os.path.join(a.run, "behavior_eval.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
