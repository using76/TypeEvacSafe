# -*- coding: utf-8 -*-
"""판단층 저작 벤치마크(openjev 의 authored144 에 해당) — 피난 상황 20건, 기대 답이 규칙에서 자명한 것만.
프롬프트를 '깎을' 때 이 점수로 판정한다. 사용: python jev_bench.py --backend llama [--lang en]
"""
import argparse
import copy
import json
import time

import numpy as np

from jev_decide import LlamaBackend, HFBackend, TypeSafeBackend, build_questions, messages_for, softmax, perm_jobs, merge_perms, ask_letters

X = lambda i, d, dr, smoke="clear", heat="cool", fire=False, wait=0, w=1.2, blocked=False: dict(
    id="X%d" % i, usable="no, fire at the exit" if fire else ("no, blocked" if blocked else "yes"), distance_m=d, direction=dr, smoke=smoke, heat=heat, fire_at_exit=fire, people_waiting=wait,
    queue_s=int(round(wait / (1.3 * w))), width_m=w, blocked=blocked)
ME = lambda role="adult", exp="none", smoke="clear", heat="cool", see=30: dict(
    role=role, walking_speed="normal" if role in ("adult", "guardian", "firefighter") else "slow", exposure_so_far=exp,
    here=dict(smoke=smoke, heat=heat), can_see_m=see, **({"equipment": "breathing apparatus and protective suit"} if role == "firefighter" else {}))
FIRE = lambda d, dr, kw: dict(direction=dr, distance_m=d, size="small" if kw < 500 else ("medium" if kw < 2000 else "large"), heat_release_kW=kw, growing=True)

CASES = [
    dict(name="single_exit_clear", ev=dict(time_since_ignition_s=40, fire=FIRE(12, "north", 800), me=ME(), exits=[X(1, 10, "south")]),
         expect=dict(mode="evacuate")),
    dict(name="near_exit_on_fire", ev=dict(time_since_ignition_s=75, fire=FIRE(6, "west", 1400), me=ME(exp="light", smoke="light smoke", heat="warm", see=8),
         exits=[X(1, 9, "west", "thick smoke", "hot", fire=True), X(2, 27, "east"), X(3, 31, "north", "light smoke")]),
         expect=dict(mode="evacuate", target_exit="X2")),
    dict(name="near_exit_crowded", ev=dict(time_since_ignition_s=60, fire=FIRE(30, "north", 900), me=ME(), exits=[X(1, 8, "south", wait=24, w=1.2), X(2, 20, "east", wait=0, w=1.8)]),
         expect=dict(mode="evacuate", target_exit="X2")),
    dict(name="guardian_child_far", ev=dict(time_since_ignition_s=50, fire=FIRE(15, "north", 900), me=ME("guardian"), exits=[X(1, 12, "south")],
         my_child=dict(status="walking", distance_m=8.0, direction="north", exposure="light")), expect=dict(mode="escort")),
    dict(name="guardian_child_collapsed", ev=dict(time_since_ignition_s=90, fire=FIRE(15, "north", 1500), me=ME("guardian", "light"), exits=[X(1, 12, "south")],
         my_child=dict(status="collapsed", distance_m=5.0, direction="north", exposure="heavy")), expect=dict(mode="escort")),
    dict(name="guardian_child_outside", ev=dict(time_since_ignition_s=90, fire=FIRE(15, "north", 1500), me=ME("guardian"), exits=[X(1, 12, "south")],
         my_child=dict(status="already outside", distance_m=14.0, direction="south", exposure="light")), expect=dict(mode="evacuate")),
    dict(name="firefighter_victim", ev=dict(time_since_ignition_s=100, fire=FIRE(18, "north", 2500), me=ME("firefighter", smoke="thick smoke", heat="warm", see=2),
         exits=[X(1, 6, "south")], people_in_danger_nearby=[dict(id="V1", who="elderly", distance_m=9.0, direction="north", condition="collapsed, cannot walk", smoke_there="thick smoke")]),
         expect=dict(mode="rescue", rescue_target="V1", rescue_feasible="yes")),
    dict(name="firefighter_no_victim", ev=dict(time_since_ignition_s=140, fire=FIRE(18, "north", 2500), me=ME("firefighter", smoke="thick smoke", see=2), exits=[X(1, 6, "south")]),
         expect=dict(mode="evacuate")),
    dict(name="adult_helps_collapsed", ev=dict(time_since_ignition_s=70, fire=FIRE(20, "north", 1000), me=ME(exp="none"), exits=[X(1, 7, "south")],
         people_in_danger_nearby=[dict(id="V1", who="elderly", distance_m=2.5, direction="east", condition="collapsed, cannot walk", smoke_there="light smoke")]),
         expect=dict(mode="rescue", rescue_target="V1")),
    dict(name="adult_near_collapse_self", ev=dict(time_since_ignition_s=120, fire=FIRE(8, "north", 2500), me=ME(exp="near collapse", smoke="thick smoke", heat="hot", see=2), exits=[X(1, 7, "south")],
         people_in_danger_nearby=[dict(id="V1", who="adult", distance_m=3.0, direction="east", condition="collapsed, cannot walk", smoke_there="thick smoke")]),
         expect=dict(mode="evacuate")),
    dict(name="all_exits_fire", ev=dict(time_since_ignition_s=150, fire=FIRE(10, "north", 4000), me=ME(exp="light", smoke="light smoke"),
         exits=[X(1, 12, "west", "thick smoke", "burning", fire=True), X(2, 25, "east", "smoke too thick to see", "hot", fire=True)]), expect=dict(mode="shelter")),
    dict(name="breakthrough_small_fire", ev=dict(time_since_ignition_s=25, fire=FIRE(3, "south", 250), me=ME(), exits=[X(1, 4, "south", "light smoke", "hot", fire=True)]),
         expect=dict(mode="breakthrough")),
    dict(name="no_breakthrough_big_fire", ev=dict(time_since_ignition_s=150, fire=FIRE(6, "south", 3500), me=ME(smoke="light smoke", heat="warm"), exits=[X(1, 15, "south", "smoke too thick to see", "burning", fire=True)]),
         expect=dict(mode="shelter")),
    dict(name="elderly_two_exits", ev=dict(time_since_ignition_s=60, fire=FIRE(20, "north", 1000), me=ME("elderly"), exits=[X(1, 15, "west", "thick smoke", "warm"), X(2, 16, "east")]),
         expect=dict(mode="evacuate", target_exit="X2")),
    dict(name="smoke_vs_distance", ev=dict(time_since_ignition_s=80, fire=FIRE(15, "north", 1500), me=ME(smoke="light smoke", see=8), exits=[X(1, 9, "north", "smoke too thick to see", "hot"), X(2, 22, "south")]),
         expect=dict(mode="evacuate", target_exit="X2")),
    dict(name="blocked_exit", ev=dict(time_since_ignition_s=40, fire=FIRE(25, "north", 700), me=ME(), exits=[X(1, 5, "west", blocked=True), X(2, 18, "east")]),
         expect=dict(mode="evacuate", target_exit="X2")),
    dict(name="crowd_but_only_exit", ev=dict(time_since_ignition_s=50, fire=FIRE(25, "north", 700), me=ME(), exits=[X(1, 8, "south", wait=30, w=1.2)]),
         expect=dict(mode="evacuate")),
    dict(name="firefighter_victim_too_far_in_fire", ev=dict(time_since_ignition_s=200, fire=FIRE(4, "north", 6000), me=ME("firefighter", exp="heavy", smoke="smoke too thick to see", heat="burning", see=1),
         exits=[X(1, 5, "south")], people_in_danger_nearby=[dict(id="V1", who="adult", distance_m=14.0, direction="north", condition="collapsed, cannot walk", smoke_there="smoke too thick to see")]),
         expect=dict(rescue_feasible="no")),
    dict(name="two_clear_exits_pick_near", ev=dict(time_since_ignition_s=30, fire=FIRE(30, "north", 500), me=ME(), exits=[X(1, 6, "west"), X(2, 25, "east")]),
         expect=dict(mode="evacuate", target_exit="X1")),
    dict(name="queue_small_diff_keep_near", ev=dict(time_since_ignition_s=30, fire=FIRE(30, "north", 500), me=ME(), exits=[X(1, 6, "west", wait=3), X(2, 25, "east", wait=0)]),
         expect=dict(mode="evacuate", target_exit="X1")),
]


def run(be, lang="en", verbose=True, n_perm=1):
    hits = tot = 0; rows = []; t0 = time.time(); items = []
    for c in CASES:
        ev = c["ev"]; vic = ev.get("people_in_danger_nearby", [])
        items.append((ev, build_questions(ev, ev["exits"], vic, ev["me"]["role"])))
    answers = be.ask_many(items, lang) if isinstance(be, TypeSafeBackend) else ask_letters(be, items, lang, n_perm)
    res = {}; jobs = [None] * sum(len(q) for _, q in items)
    for c, pr in zip(CASES, answers):
        res[c["name"]] = {h: (max(p, key=p.get), {k: round(v, 2) for k, v in p.items()}) for h, p in pr.items()}
    for c in CASES:
        r = res[c["name"]]; ok = []
        for h, want in c["expect"].items():
            got = r.get(h, ("-", {}))[0]; ok.append(got == want); tot += 1; hits += (got == want)
        if verbose:
            print("%-34s %s | %s" % (c["name"], "OK " if all(ok) else "XX ", "  ".join("%s=%s%s" % (h, r[h][0], "" if h not in c["expect"] else ("" if r[h][0] == c["expect"][h] else "(want %s)" % c["expect"][h])) for h in r)))
        rows.append(dict(name=c["name"], result={h: v[0] for h, v in r.items()}, probs={h: v[1] for h, v in r.items()}, ok=all(ok)))
    dt = time.time() - t0
    print("정확도 %d/%d = %.2f  (%d 판정, %.1fs, stats %s)" % (hits, tot, hits / tot, len(jobs), dt, {k: v for k, v in getattr(be, "stats", {}).items() if k != "usage"}))
    return hits / tot, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama"); ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--lang", default="en"); ap.add_argument("--out", default=None)
    ap.add_argument("--n_perm", type=int, default=3); ap.add_argument("--think", action="store_true")
    a = ap.parse_args()
    be = TypeSafeBackend() if a.backend == "typesafe" else (LlamaBackend(a.url, think=a.think) if a.backend == "llama" else HFBackend(a.model))
    acc, rows = run(be, a.lang, n_perm=a.n_perm)
    if a.out:
        json.dump(dict(acc=acc, rows=rows), open(a.out, "w"), indent=1, ensure_ascii=False)
