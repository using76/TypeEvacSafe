# -*- coding: utf-8 -*-
"""피난 판단층(Jev 형식) — 상태 텍스트 + 질문 + 타입 옵션 → 한 번의 forward 로 옵션 확률 판독.

참조: TheoLeeCJ/openjev (동결 Qwen, 답 글자 토큰 A..P 의 마지막 위치 로짓만 softmax, 상태 프리필 공유),
      browser-use/jev-ultrafast (한 상태에 operation + 조건부 target 질문을 동시에 던지고 선택된 operation 의 head 만 실행).
백엔드
  · llama : 우리 llama-server(Qwen3.8-Flash-Next UD-Q4_K_XL, qwen4exp 패치 빌드) — /v1/chat/completions, max_tokens 1, top_logprobs 로 글자 로짓 판독
  · hf    : transformers 인과 LM(예: Qwen/Qwen3.5-4B bf16) — openjev 와 같은 직접 로짓 판독, 상태 프리필 공유 배치
사용 (env2 와 결합): env = EvacEnv2(..., macro="jev"); env.decider = JevDecider(backend="llama", url="http://127.0.0.1:8081")
단독 시험: python jev_decide.py --backend llama --demo
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import time
import urllib.error
import urllib.request

import numpy as np

from jev_prompts import (SYSTEM, RULES_EN, RULES_KO, Q_MODE, MODE_OPTIONS, Q_EXIT, Q_RESCUE, Q_FEASIBLE,
                         FEASIBLE_OPTIONS, Q_SURVIVE, SURVIVE_OPTIONS, Q_PACE, PACE_OPTIONS)

LETTERS = "ABCDEFGHIJKLMNOP"
DIRS8 = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def compass(dx, dy):
    a = math.degrees(math.atan2(dy, dx)) % 360
    return DIRS8[int((a + 22.5) // 45) % 8]


def smoke_word(ks):
    return "clear" if ks < 0.1 else ("light smoke" if ks < 0.5 else ("thick smoke" if ks < 1.5 else "smoke too thick to see"))


def heat_word(t):
    return "cool" if t < 40 else ("warm" if t < 80 else ("hot" if t < 150 else "burning"))


# ---------------------------------------------------------------- 상태 텍스트(evidence)
def build_evidence(env, i):
    """env2 의 에이전트 i 상태 → evidence dict (짧게, 판단에 쓰이는 관계만)."""
    import torch
    from env2 import ROLES, R_GUARD, R_FF, MAX_EXITS
    b = int(env.ca[i]); t = float(env.t[i])
    pos = env.pos[i]
    role = ROLES[int(env.role[i])]
    fed = float(env.fed_tox[i] + env.fed_heat[i])
    T_here = float(env._sample_field(env.T, env.pos, env.t)[i]); K_here = float(env.ks_avg[i])
    fi = int(env._frame(env.t)[i].round()); hrr = float(env.HRR[b, min(fi, env.NT - 1)])
    fx, fy = env.fire_xy[b].tolist()
    ex = []
    rf = env.route_fire()[i] if hasattr(env, "FDIST") else None
    for k in range(MAX_EXITS):
        if not bool(env.ex_valid[b, k]): continue
        dk = env._lookup_exit(env.DIST_E, env.pos, torch.full((env.A,), k, device=env.dev, dtype=torch.long))[i]
        dk = float(dk) if math.isfinite(float(dk)) else None
        c = env.ex_center[b, k]
        fire_here = bool(env.ex_safe[b, k] < 0.05 and env.ex_open[b, k]); blk = (not bool(env.ex_open[b, k])) or dk is None
        ex.append(dict(id="X%d" % (k + 1), usable="no, fire at the exit" if fire_here else ("no, blocked" if blk else "yes"),
                       distance_m=None if dk is None else round(dk, 1),
                       direction=compass(float(c[0] - pos[0]), float(c[1] - pos[1])),
                       smoke=smoke_word(float(env.ex_K[b, k])), heat=heat_word(float(env.ex_T[b, k])),
                       fire_at_exit=fire_here,
                       route=("passes right next to the fire" if (rf is not None and bool(rf[k]) and hrr > 200) else "clear of the fire"),
                       people_waiting=int(round(float(env.ex_q[b, k] * 1.3 * env.ex_width[b, k]))),
                       queue_s=int(round(float(env.ex_q[b, k]))), width_m=round(float(env.ex_width[b, k]), 1),
                       blocked=blk))
    ev = dict(
        time_since_ignition_s=int(t),
        fire=dict(direction=compass(fx - float(pos[0]), fy - float(pos[1])), distance_m=round(float((env.fire_xy[b] - pos).norm()), 1),
                  size="small" if hrr < 500 else ("medium" if hrr < 2000 else "large"), heat_release_kW=int(hrr),
                  growing=bool(env.HRR[b, min(fi + 10, env.NT - 1)] > hrr * 1.05)),
        me=dict(role=role, sex="female" if int(getattr(env, "sex", torch.zeros(1))[i] if hasattr(env, "sex") else 0) == 1 else "male",
                age=int(float(env.age[i])) if hasattr(env, "age") else 35,
                mobility=("cannot walk, needs to be carried" if (hasattr(env, "cannot_walk") and bool(env.cannot_walk[i])) else
                          ("collapsed" if bool(env.down[i]) else ("injured, walks slowly" if role == "injured" else
                           ("slow" if role in ("child", "elderly") else "normal")))),
                exposure_so_far="none" if fed < 0.05 else ("light" if fed < 0.3 else ("heavy" if fed < 0.7 else "near collapse")),
                here=dict(smoke=smoke_word(K_here), heat=heat_word(T_here)),
                can_see_m=int(min(30, 3.0 / max(K_here, 0.1))), knows_building="yes"),
        exits=ex)
    if role == "firefighter":
        ev["me"]["equipment"] = "breathing apparatus and protective suit"
    w = int(env.ward[i])
    if role == "guardian" and w >= 0:
        wf = float(env.fed_tox[w] + env.fed_heat[w]); dwn = bool(env.down[w]); exd = bool(env.exited[w])
        dist = float((env.pos[w] - pos).norm())
        ev["my_child"] = dict(status="already outside" if exd else ("collapsed" if dwn else "walking"),
                              distance_m=round(dist, 1), direction=compass(float(env.pos[w, 0] - pos[0]), float(env.pos[w, 1] - pos[1])),
                              exposure="heavy" if wf >= 0.3 else "light")
    # 위험 피난자 목록(15 m)
    vic = []
    N = env.N; base = b * N
    fedv = env.fed_tox + env.fed_heat
    for j in range(base, base + N):
        if j == i or env.done[j] or not env.active[j] or env.attached[j] >= 0: continue
        dj = float((env.pos[j] - pos).norm())
        if dj > 15.0: continue
        f = float(fedv[j]); down = bool(env.down[j]); stuck = float(env.stuck_t[j]) >= 5.0
        if not (down or f >= 0.3 or stuck): continue
        vic.append(dict(id="V%d" % (len(vic) + 1), agent=j, who=ROLES[int(env.role[j])], distance_m=round(dj, 1),
                        direction=compass(float(env.pos[j, 0] - pos[0]), float(env.pos[j, 1] - pos[1])),
                        condition="collapsed, cannot walk" if down else ("heavily exposed, still walking" if f >= 0.3 else "stuck in smoke"),
                        smoke_there=smoke_word(float(env.ks_avg[j]))))
        if len(vic) >= 6: break
    if vic:
        ev["people_in_danger_nearby"] = [{k: v for k, v in x.items() if k != "agent"} for x in vic]
    return ev, ex, vic


def build_questions(ev, exits, victims, role):
    """jev-ultrafast 식 fan-out: mode + 조건부 head. 반환 {head: (question, [(id, description)])}"""
    qs = {}
    opts = [(m, d) for m, d in MODE_OPTIONS if not (m == "escort" and "my_child" not in ev)
            and not (m == "rescue" and not victims)]
    qs["mode"] = (Q_MODE, opts)
    def _exo(e):
        # 위험을 문장 맨 앞에 — 거리보다 먼저 읽히게(실측: 뒤에 붙이면 9 m 화재 출구를 27 m 안전 출구보다 골랐다)
        if e["fire_at_exit"]: head = "DANGER, FIRE AT THIS EXIT: "
        elif e["blocked"]: head = "BLOCKED, cannot be used: "
        elif e["smoke"] == "smoke too thick to see": head = "DANGER, smoke too thick to see: "
        elif e.get("route", "").startswith("passes"): head = "WARNING, the way there passes right next to the fire: "
        else: head = "usable: "
        return (e["id"], head + "%s, %s m %s, %s, %s, %d people waiting (about %d s queue)" % (
            e["id"], e["distance_m"] if e["distance_m"] is not None else "unreachable", e["direction"], e["smoke"], e["heat"], e["people_waiting"], e["queue_s"]))
    exo = [_exo(e) for e in exits]
    if len(exo) >= 2:
        qs["target_exit"] = (Q_EXIT, exo)
    if victims:
        qs["rescue_target"] = (Q_RESCUE, [(v["id"], "%s %s, %s m %s, %s" % (v["who"], v["id"], v["distance_m"], v["direction"], v["condition"])) for v in victims]
                               + [("none", "Nobody; keep evacuating.")])
        qs["rescue_feasible"] = (Q_FEASIBLE, FEASIBLE_OPTIONS)
    qs["pace"] = (Q_PACE, PACE_OPTIONS)
    qs["survive"] = (Q_SURVIVE, SURVIVE_OPTIONS)
    return qs


def messages_for(ev, question, options, lang="en"):
    payload = {"situation": ev, "question": question,
               "options": [{"letter": LETTERS[k], "description": d} for k, (_, d) in enumerate(options)]}
    rules = RULES_EN if lang == "en" else RULES_KO
    return [{"role": "system", "content": SYSTEM + "\n\n" + rules},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


# ---------------------------------------------------------------- 백엔드
class LlamaBackend:
    """llama-server /v1/chat/completions → 글자 로짓. 슬롯 병렬(threads).
    think=False: max_tokens=1, 답 위치 top_logprobs 판독(openjev 직접 판독).
    think=True : 생각(reasoning) 허용 후 content 첫 글자 토큰의 top_logprobs 판독 — Qwen3.8 은 생각 없이 읽으면 순열에 불안정했다(벤치 3/30)."""

    def __init__(self, url="http://127.0.0.1:8081", threads=4, top=20, think=False, think_tokens=700):
        self.url, self.threads, self.top, self.think, self.think_tokens = url, threads, top, think, think_tokens
        self.stats = dict(calls=0, gen_tokens=0)

    def _one(self, msgs, n_opt):
        body = dict(messages=msgs, max_tokens=self.think_tokens if self.think else 1, temperature=0.0, logprobs=True,
                    top_logprobs=self.top, cache_prompt=True, chat_template_kwargs={"enable_thinking": bool(self.think)})
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        c = r["choices"][0]
        self.stats["calls"] += 1; self.stats["gen_tokens"] += int(r.get("usage", {}).get("completion_tokens", 0))
        toks = c["logprobs"]["content"] or []
        out = np.full(n_opt, -30.0)
        # 생각 모드: logprobs 에는 reasoning 토큰도 들어온다 — '</think>' 뒤의 첫 글자 토큰만 답이다
        start = 0
        for j, tk in enumerate(toks):
            if tk["token"].strip() == "</think>": start = j + 1
        for tk in toks[start:]:
            tok = tk["token"].strip()
            if tok in LETTERS[:n_opt]:
                lp = {t["token"].strip(): float(t["logprob"]) for t in tk["top_logprobs"]}
                for k in range(n_opt):
                    if LETTERS[k] in lp: out[k] = lp[LETTERS[k]]
                return out
        if toks:                                                    # 글자를 못 찾음 — 첫 토큰 분포로 대체
            lp = {t["token"].strip(): float(t["logprob"]) for t in toks[0]["top_logprobs"]}
            for k in range(n_opt):
                if LETTERS[k] in lp: out[k] = lp[LETTERS[k]]
        return out

    def score(self, jobs):
        """jobs: [(msgs, n_opt)] → [logits np.array]"""
        with cf.ThreadPoolExecutor(self.threads) as ex:
            return list(ex.map(lambda j: self._one(*j), jobs))


class HFBackend:
    """openjev direct.py 와 같은 판독: 마지막 위치 로짓에서 A..P 토큰만. (배치 패딩 좌측)"""

    def __init__(self, model="Qwen/Qwen3.5-4B", device="cuda", dtype="bfloat16"):
        import torch, transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map={"": device})
        self.model.eval(); self.dev = device
        self.slots = [self.tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def score(self, jobs, bs=16):
        import torch
        out = []
        for s in range(0, len(jobs), bs):
            chunk = jobs[s:s + bs]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m, _ in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            with torch.inference_mode():
                lg = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            for r, (_, n) in enumerate(chunk):
                out.append(lg[r, self.slots[:n]].cpu().numpy())
        return out


class TypeSafeBackend:
    """TypeSafe Jev API(jev-ultrafast model.py 의 호출 형식): 한 state 에 여러 choice 질문을 한 요청으로.
    키는 환경변수 TYPESAFE_API_KEY 또는 ~/.typesafe_key 파일 — 코드·저장소에 넣지 않는다."""

    def __init__(self, model="jev-latest", url="https://api.typesafe.ai/v1/systemone", threads=4):
        import os
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            kp = os.path.expanduser("~/.typesafe_key")
            key = open(kp).read().strip() if os.path.exists(kp) else None
        if not key: raise RuntimeError("TYPESAFE_API_KEY 없음")
        self.key, self.model, self.url, self.threads = key, model, url, threads
        self.stats = dict(calls=0, seconds=0.0, usage=[])

    def ask(self, ev, qs, lang="en"):
        """qs: {head: (question, [(id, desc)])} → {head: {id: prob}}"""
        rules = RULES_EN if lang == "en" else RULES_KO
        questions = {h: {"type": "choice", "criteria": {oid: desc for oid, desc in opts},
                         "instructions": {"question": q, "rules": rules}} for h, (q, opts) in qs.items()}
        body = {"model": self.model, "state": ev, "questions": questions}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        t0 = time.time(); r = None
        for attempt in range(3):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=60).read()); break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")[:300]
                if e.code in (429, 503, 529) and attempt < 2: time.sleep(1.0 * 2 ** attempt); continue
                raise RuntimeError("TypeSafe HTTP %d: %s" % (e.code, msg))
        self.stats["calls"] += 1; self.stats["seconds"] += time.time() - t0; self.stats["usage"].append(r.get("usage", {}))
        out = {}
        for h, (q, opts) in qs.items():
            a = r["answers"].get(h, {}); pr = a.get("probabilities", {})
            out[h] = {oid: float(pr.get(oid, 0.0)) for oid, _ in opts}
            if not any(out[h].values()) and a.get("choice") in out[h]: out[h][a["choice"]] = 1.0
        return out

    def ask_many(self, items, lang="en"):
        with cf.ThreadPoolExecutor(self.threads) as ex:
            return list(ex.map(lambda it: self.ask(it[0], it[1], lang), items))


def softmax(x):
    x = np.asarray(x, dtype=np.float64); x = x - x.max(); e = np.exp(x); return e / e.sum()


def ask_letters(be, items, lang="en", n_perm=1):
    """글자 로짓 백엔드(llama/hf)로 items=[(ev, qs)] 를 한꺼번에 → [{head: {id: prob}}]"""
    jobs, meta = [], []
    for ev, qs in items:
        for h, (q, opts) in qs.items():
            jb, pm = perm_jobs(ev, q, opts, lang, n_perm); jobs += jb; meta.append((h, opts, pm))
    lg = be.score(jobs); out = []; p = 0; k = 0
    for ev, qs in items:
        res = {}
        for _ in qs:
            h, opts, pm = meta[k]; k += 1
            pr = merge_perms(lg[p:p + len(pm)], pm, len(opts)); p += len(pm)
            res[h] = {opts[j][0]: float(pr[j]) for j in range(len(opts))}
        out.append(res)
    return out


def perm_jobs(ev, q, opts, lang, n_perm):
    """글자 위치 편향 완화(openjev 의 perturbation 시험이 잰 그 편향): 옵션을 순환 이동한 n_perm 개 순열로 묻고
    옵션 id 기준으로 로그확률을 평균한다. 실측(27B Q4): 원순서 A=0.95 → 역순 0.35 — 첫 글자 선호가 의미를 눌렀다."""
    n = len(opts); k = max(1, min(n_perm, n))
    perms = [[(i + r) % n for i in range(n)] for r in range(k)]          # 순환 이동
    jobs = [(messages_for(ev, q, [opts[i] for i in pm], lang), n) for pm in perms]
    return jobs, perms


def merge_perms(logits_list, perms, n):
    lp = np.zeros(n)
    for lg, pm in zip(logits_list, perms):
        l = np.log(softmax(lg) + 1e-9)
        for pos, i in enumerate(pm): lp[i] += l[pos]
    return softmax(lp / len(perms))


# ---------------------------------------------------------------- 결정기
class JevDecider:
    def __init__(self, backend="llama", lang="en", n_perm=1, think=False, **kw):
        """think=False 가 기본 — openjev 식 즉답 판독. 27B Q4 처럼 즉답이 무너지는 모델에서만 think=True.
        (실측: 생각 모드는 head 당 수백 토큰을 생성해 결정당 수십 초가 된다)"""
        if backend == "typesafe": self.be = TypeSafeBackend(**kw)
        elif backend == "llama": self.be = LlamaBackend(think=think, **kw)
        else: self.be = HFBackend(**kw)
        self.backend, self.lang, self.n_perm = backend, lang, n_perm
        self.log = []

    def ask(self, items):
        if self.backend == "typesafe": return self.be.ask_many(items, self.lang)
        return ask_letters(self.be, items, self.lang, self.n_perm)

    def decide(self, env, idx):
        """idx: 결정할 에이전트 인덱스 목록 → [{mode, target_exit_k, rescue_agent, p_survive, probs}]"""
        t0 = time.time()
        specs, items = [], []
        for i in idx:
            ev, exits, vic = build_evidence(env, i)
            qs = build_questions(ev, exits, vic, ev["me"]["role"])
            items.append((ev, qs)); specs.append((i, ev, exits, vic, qs))
        answers = self.ask(items)
        out = []
        for (i, ev, exits, vic, qs), pr in zip(specs, answers):
            res = dict(agent=int(i), probs=pr)
            for h in qs:
                res[h] = max(pr[h], key=pr[h].get)
            # 실행 규칙(jev-ultrafast): mode 가 고른 head 만 쓴다
            res["target_exit_k"] = None
            if "target_exit" in res and res["mode"] in ("evacuate", "breakthrough", "follow_crowd", "escort"):
                res["target_exit_k"] = int(res["target_exit"][1:]) - 1
            elif len(exits) == 1:
                res["target_exit_k"] = 0
            res["rescue_agent"] = -1
            # 실행 가드(jev-ultrafast 의 validate 와 같은 취지): 구조가 불가능하다고 답했으면 rescue 를 실행하지 않는다
            if res["mode"] == "rescue" and (not vic or res.get("rescue_target", "none") == "none" or res.get("rescue_feasible") == "no"):
                res["mode"] = "evacuate"
                if "target_exit" in res: res["target_exit_k"] = int(res["target_exit"][1:]) - 1
                elif len(exits) == 1: res["target_exit_k"] = 0
            if res["mode"] == "rescue":
                res["rescue_agent"] = int(next(v["agent"] for v in vic if v["id"] == res["rescue_target"]))
            res["p_survive"] = pr["survive"].get("likely", 0.5)
            res["pace_mult"] = {"run": 1.4, "walk": 1.0, "slow": 0.6, "stop": 0.0}.get(res.get("pace", "walk"), 1.0)
            out.append(res)
        self.log.append(dict(n=len(idx), t=float(env.t[idx[0]]) if idx else 0, seconds=round(time.time() - t0, 2),
                             decisions=[{k: v for k, v in r.items() if k != "probs"} for r in out]))
        return out


# ---------------------------------------------------------------- 단독 시험
DEMO_EV = {
    "time_since_ignition_s": 75,
    "fire": {"direction": "west", "distance_m": 6.5, "size": "medium", "heat_release_kW": 1400, "growing": True},
    "me": {"role": "guardian", "walking_speed": "normal", "exposure_so_far": "light", "here": {"smoke": "light smoke", "heat": "warm"}, "can_see_m": 8},
    "exits": [
        {"id": "X1", "distance_m": 9.0, "direction": "west", "smoke": "thick smoke", "heat": "hot", "fire_at_exit": True, "people_waiting": 0, "queue_s": 0, "width_m": 1.8, "blocked": False},
        {"id": "X2", "distance_m": 27.0, "direction": "east", "smoke": "clear", "heat": "cool", "fire_at_exit": False, "people_waiting": 14, "queue_s": 9, "width_m": 1.2, "blocked": False},
        {"id": "X3", "distance_m": 31.0, "direction": "north", "smoke": "light smoke", "heat": "cool", "fire_at_exit": False, "people_waiting": 2, "queue_s": 1, "width_m": 1.2, "blocked": False}],
    "my_child": {"status": "walking", "distance_m": 4.5, "direction": "south", "exposure": "light"},
}
DEMO_EXITS = DEMO_EV["exits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama"); ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--lang", default="en")
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--n_perm", type=int, default=3)
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args()
    be = TypeSafeBackend() if a.backend == "typesafe" else (LlamaBackend(a.url, think=a.think) if a.backend == "llama" else HFBackend(a.model))
    if a.demo:
        for e in DEMO_EV["exits"]: e.setdefault("usable", "no, fire at the exit" if e["fire_at_exit"] else "yes")
        qs = build_questions(DEMO_EV, DEMO_EXITS, [], "guardian")
        t0 = time.time()
        pr = be.ask_many([(DEMO_EV, qs)], a.lang)[0] if a.backend == "typesafe" else ask_letters(be, [(DEMO_EV, qs)], a.lang, a.n_perm)[0]
        for h in qs:
            print("%-14s %s" % (h, "  ".join("%s=%.2f" % kv for kv in pr[h].items())))
        print("%.2fs, %d heads, stats %s" % (time.time() - t0, len(qs), {k: v for k, v in getattr(be, "stats", {}).items() if k != "usage"}))


if __name__ == "__main__":
    main()
