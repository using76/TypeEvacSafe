# -*- coding: utf-8 -*-
"""피난 RL v2 환경 — 다중 에이전트·역할·가족·구조·출구 선택 (EVAC_RL_PLAN_v2 §2·§3).

배치 = B 케이스 × N 에이전트(평탄화 A=B·N). 동역학은 v1(env.py, rust_evac BR 이식)을 유지하고 다음을 더한다.
  · 에이전트 간 반발·마찰(rust_evac sfm_force, A=2000 B=0.08 k=1.2e5 κ=2.4e5, cutoff 2.5 m)
  · 역할 5종: adult / child / elderly / guardian / firefighter — v0·반경·FED 계수·복사 임계
  · 가족: guardian ↔ child(ward) 링크. child 는 정책 없이 보호자 추종(rust_evac Follow), 보호자는 동행 중 v0 를 child 속도로 제한
  · 구조: 위험 피난자(FED ≥ 0.3 / 5 s 정체 / 무력화)를 구조자가 1 m 안에서 확보(attach) → 함께 이동(0.6/0.5 m/s) → 출구 = 구조 완료
  · 출구 목록(≤8) 요약: 거리·방향·안전 점수 s_e(온도·연기·화염)·대기 인원 q_e — 매크로(목표 출구)는 규칙 v1(5 s 주기), 판단층은 Phase D
  · 무력화(FED ≥ 1)는 종료가 아니라 '쓰러짐'(이동 불가, 구조 가능). FED ≥ 2 → 사망(종료).
보상은 R dict 노브 — 계획서 §3 표 그대로.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from env_v1 import P, SUB, SUB_DT, STEP_DT, N_DIR, DIRS, MULTS, N_ACT, FIRE_THR, PATCH, R_HOLD, fed_rates, speed_factor

# ── 역할 ────────────────────────────────────────────────────────────────────────
ROLES = ["adult", "child", "elderly", "guardian", "firefighter", "injured"]
R_ADULT, R_CHILD, R_ELDER, R_GUARD, R_FF, R_INJ = range(6)
N_ROLE = 6
ROLE_V0 = torch.tensor([1.2, 0.8, 0.7, 1.2, 1.2, 0.5])
ROLE_RAD = torch.tensor([0.20, 0.15, 0.25, 0.20, 0.25, 0.25])
ROLE_TOX = torch.tensor([1.0, 1.3, 1.0, 1.0, 0.0, 1.0])          # SCBA → 독성 0
ROLE_HEAT = torch.tensor([1.0, 1.0, 1.2, 1.0, 0.2, 1.0])         # 방화복
ROLE_QTH = torch.tensor([2.5, 2.5, 2.5, 2.5, 7.0, 2.5])          # 복사 임계 kW/m²
ROLE_KSLOPE = torch.tensor([0.081, 0.081, 0.12, 0.081, 0.081, 0.10])
SEX_V0 = {"male": 1.0, "female": 0.92}                             # IMO 1533: 성인 남 1.11~1.85, 여 0.93~1.55 → 상대 배율
MAX_EXITS = 8
MACRO_DT = 5.0
VICTIM_FED = 0.3
STUCK_T = 5.0
ATTACH_R = 1.0
SPEED_ASSIST, SPEED_CARRY = 0.6, 0.5
DEAD_FED = 2.0                        # FED 2 는 즉시 사망(보조 기준)
DEAD_T = 10.0                         # 사용자 규칙(2026-09-18): ASET(FED≥1) 넘은 뒤 위험(열·연기)에 10 s 이상 노출 → 사망. 시신은 제자리에 남는다
VIS_C = 3.0                           # FDS VISIBILITY = C / K_s (반사 표지 C=3) — 우리 KS(=8700·soot) 로 같은 값
VIS_FULL, VIS_NONE = 10.0, 1.0        # 시야 ≥10 m 이면 길찾기 정확, ≤1 m 이면 방향 오차 σ=90°
AA_CUTOFF = 2.5
A_AGENT, B_AGENT = 2000.0, 0.08

MODES = ["evacuate", "wait", "escort", "rescue", "follow_crowd", "breakthrough", "shelter"]
M_EVAC, M_WAIT, M_ESCORT, M_RESCUE, M_FOLLOW, M_BREAK, M_SHELTER = range(7)

R_DEFAULT = dict(exit=100.0, clean=50.0, safe_exit=30.0, time=-0.1, fed=-20.0, down=-100.0, dead=-50.0, trap=-50.0,
                 contact=-0.01, crowd=-0.05, switch=20.0, ward_exit=60.0, ward_down=-80.0, separation=-0.05,
                 solo_exit=0.5, attach_ff=30.0, attach_adult=20.0, rescue_ff=100.0, rescue_adult=60.0,
                 approach=0.5, ff_idle=-0.1, carried_exit=50.0, shaping=0.5)

OBS_PATCH_C = 5                      # T, KS, FIRE, WALL, DENSITY
OBS_VEC = 13 + 7 + MAX_EXITS * 8 + 5 + 6 + 3 + 8 + 1   # = 107 (역할 one-hot 6 + v0, 출구 8×8, 모드 7 + p_survive, 시야/10)


class EvacEnv2:
    def __init__(self, field_dir, n_cases, n_agents, device="cuda", seed=0, reward=None, case_filter=None,
                 no_fire=False, time_scale=1.0, roles=None, n_ff=1, ff_enter=(60.0, 120.0), max_cells=None,
                 macro="rule", fp16=True, macro_dt=MACRO_DT, premove=5.0, roster=None, spawn=None):
        """roster: 케이스마다 같은 N 명 프로필 목록 [{role, sex, age, cannot_walk, guardian_of(인덱스)}] — 있으면 역할 비율 대신 사용.
        spawn: (cx, cy, r_min, r_max) — 출발 위치를 이 원환 안 자유 셀로 제한(예: 화원 주변 밀집)."""
        self.roster = roster; self.spawn = spawn
        self.dev = torch.device(device)
        self.macro_dt = float(macro_dt); self.premove = float(premove)          # 출발 지연 상한(s): 인지·망설임. 화재 성장을 겪게 하려면 30~90
        self.B, self.N = n_cases, n_agents
        self.A = n_cases * n_agents
        self.rng = random.Random(seed)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.files = sorted(glob.glob(os.path.join(field_dir, "*.npz")))
        if case_filter:
            self.files = [f for f in self.files if case_filter(os.path.basename(f))]
        assert self.files, "위험장 없음: " + field_dir
        self.max_cells = max_cells
        self.R = dict(R_DEFAULT)
        if reward: self.R.update(reward)
        self.no_fire = no_fire; self.time_scale = float(time_scale); self.fp16 = fp16
        # 역할 비율(adult, child, elderly, guardian) — child 수 = guardian 수(1:1 링크)
        self.roles = roles or dict(adult=0.6, child=0.15, elderly=0.1, guardian=0.15)
        self.n_ff = n_ff; self.ff_enter = ff_enter
        self.macro_mode = macro
        self.ca = torch.arange(self.B, device=self.dev).repeat_interleave(self.N)      # 에이전트 → 케이스
        self._load_batch()

    # ── 케이스 적재 ────────────────────────────────────────────────────────────
    def _pick_files(self):
        """셀수가 비슷한 케이스끼리 배치(패딩 낭비 방지): 무작위 앵커 하나를 뽑고 크기 순위 이웃에서 B 개."""
        if not hasattr(self, "_sizes"):
            self._sizes = []
            for f in self.files:
                z = np.load(f); self._sizes.append(int(z["WALL"].size)); z.close()
            order = np.argsort(self._sizes); self._order = [self.files[i] for i in order]
            if self.max_cells:
                self._order = [f for f, i in zip(self._order, order) if self._sizes[i] <= self.max_cells]
        k = self.rng.randrange(len(self._order))
        lo = max(0, min(k - self.B // 2, len(self._order) - 4 * self.B))
        pool = self._order[lo:lo + 4 * self.B]
        return [self.rng.choice(pool) for _ in range(self.B)]

    def _load_batch(self):
        picks = self._pick_files()
        zs = [np.load(p) for p in picks]
        NX = max(int(z["T"].shape[1]) for z in zs); NY = max(int(z["T"].shape[2]) for z in zs)
        NT = max(int(z["T"].shape[0]) for z in zs)
        B, d = self.B, self.dev
        dt_ = torch.float16 if self.fp16 else torch.float32
        T = torch.full((B, NT, NX, NY), 20.0, dtype=dt_); KS = torch.zeros((B, NT, NX, NY), dtype=dt_)
        CO = torch.zeros_like(KS); O2 = torch.full_like(KS, 20.9); FIRE = torch.zeros_like(KS)
        WALL = torch.zeros((B, NX, NY), dtype=torch.bool); FREE = torch.zeros_like(WALL); OUT = torch.zeros_like(WALL)
        EXIT = torch.zeros_like(WALL)
        DIST_E = torch.full((B, MAX_EXITS, NX, NY), float("inf"))
        self.dx = float(zs[0]["dx"])
        self.x0 = torch.zeros(B); self.y0 = torch.zeros(B); self.times = torch.zeros((B, NT)); self.case = []
        self.ex_valid = torch.zeros((B, MAX_EXITS), dtype=torch.bool); self.ex_open = torch.zeros_like(self.ex_valid)
        self.ex_center = torch.zeros((B, MAX_EXITS, 2)); self.ex_width = torch.ones((B, MAX_EXITS))
        self.ex_cells = torch.zeros((B, MAX_EXITS, NX, NY), dtype=torch.bool)
        for b, z in enumerate(zs):
            nt, nx, ny = z["T"].shape
            for A_, k in ((T, "T"), (KS, "KS"), (CO, "CO"), (O2, "O2"), (FIRE, "FIRE")):
                A_[b, :nt, :nx, :ny] = torch.from_numpy(np.nan_to_num(z[k].astype(np.float32)).clip(-6e4, 6e4)).to(dt_)
                if nt < NT: A_[b, nt:] = A_[b, nt - 1:nt]
            WALL[b, :nx, :ny] = torch.from_numpy(z["WALL"]); FREE[b, :nx, :ny] = torch.from_numpy(z["FREE"])
            if "OUTSIDE" in z.files: OUT[b, :nx, :ny] = torch.from_numpy(z["OUTSIDE"])
            EXIT[b, :nx, :ny] = torch.from_numpy(z["EXIT"])
            ex = json.loads(str(z["exits_json"])) if "exits_json" in z.files else [dict(id="E1", center=[0, 0], width=1.0, blocked=False)]
            de = z["DIST_E"] if "DIST_E" in z.files else z["DIST"][None]
            eid = z["EXIT_ID"] if "EXIT_ID" in z.files else z["EXIT"].astype(np.int8)
            for k, e in enumerate(ex[:MAX_EXITS]):
                self.ex_valid[b, k] = True; self.ex_open[b, k] = not e["blocked"]
                self.ex_center[b, k] = torch.tensor(e["center"], dtype=torch.float32); self.ex_width[b, k] = float(e["width"])
                DIST_E[b, k, :nx, :ny] = torch.from_numpy(de[k].astype(np.float32))
                self.ex_cells[b, k, :nx, :ny] = torch.from_numpy(eid == (k + 1))
            # 패딩(격자 밖)은 벽 — v2 는 출구가 명시적이라 경계=출구 규약이 없다
            WALL[b, nx:, :] = True; WALL[b, :, ny:] = True
            tt = torch.from_numpy(z["times"].astype(np.float32)); self.times[b, :nt] = tt
            if nt < NT: self.times[b, nt:] = tt[-1] + torch.arange(1, NT - nt + 1) * (tt[-1] - tt[-2])
            self.x0[b] = float(z["x0"]); self.y0[b] = float(z["y0"]); self.case.append(str(z["case"]))
            z.close()
        if self.no_fire:
            T.fill_(20.0); KS.zero_(); CO.zero_(); O2.fill_(20.9); FIRE.zero_()
        self.T, self.KS, self.CO, self.O2, self.FIRE = (A_.to(d) for A_ in (T, KS, CO, O2, FIRE))
        self.WALL, self.FREE, self.OUT, self.EXIT = WALL.to(d), FREE.to(d), OUT.to(d), EXIT.to(d)
        self.DIST_E = DIST_E.to(d); self.ex_cells = self.ex_cells.to(d)
        for k in ("ex_valid", "ex_open", "ex_center", "ex_width", "times", "x0", "y0"):
            setattr(self, k, getattr(self, k).to(d))
        self.NX, self.NY, self.NT = NX, NY, NT
        self.DIST = torch.where(self.ex_open[:, :, None, None], self.DIST_E, torch.full_like(self.DIST_E, float("inf"))).amin(1)
        self.t_end = self.times[:, -1] * self.time_scale + R_HOLD
        self.WDIST = self._wall_dist(self.WALL)
        gx, gy = self._grad(self.WDIST)
        self.WN = torch.stack([gx, gy], -1); self.WN = self.WN / self.WN.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # 출구별 방향장 −∇DIST_E [B,K,X,Y,2]
        Dm = torch.where(torch.isfinite(self.DIST_E), self.DIST_E, torch.full_like(self.DIST_E, 1e3)).flatten(0, 1)
        gx, gy = self._grad(Dm)
        g = -torch.stack([gx, gy], -1); g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.EXIT_DIR_E = g.view(self.B, MAX_EXITS, NX, NY, 2)
        fmax = self.FIRE.float().amax(dim=1); idx = fmax.flatten(1).argmax(1)
        self.fire_xy = torch.stack([idx // NY, idx % NY], -1).float() * self.dx + torch.stack([self.x0, self.y0], -1)
        # 화원 2 m 권역에서의 측지 거리장(벽 우회) — 경로가 화원 곁을 지나는지 판정: FDIST(a) + DEF[k] ≈ DIST_E[k](a)
        fire_cell = torch.zeros_like(self.WALL); fire_cell[torch.arange(B, device=d), idx // NY, idx % NY] = True
        r2 = int(round(2.0 / self.dx)); fire_zone = F.max_pool2d(fire_cell.float().unsqueeze(1), 2 * r2 + 1, 1, r2).squeeze(1) > 0
        self.FDIST = self._geo_dist(fire_zone & ~self.WALL)
        self.DEF = self._def()
        HRR = torch.zeros((B, NT))
        for b, p in enumerate(picks):
            z = np.load(p)
            if "HRR" in z.files:
                h = torch.from_numpy(z["HRR"].astype(np.float32)); HRR[b, :len(h)] = h; HRR[b, len(h):] = h[-1]
            else:
                HRR[b] = float(z["fire_Q"])
            z.close()
        self.HRR = HRR.to(d)                                               # [B,NT] kW — 복사력 점광원 세기 Q(t)
        if self.no_fire: self.HRR.zero_()
        self.frame_dt = (self.times[:, 1] - self.times[:, 0]).clamp_min(1e-3)
        # 출구 대기열 계수용: 출구 중심 3 m 반경
        self.main_exit = torch.where(self.ex_open, self.ex_width, torch.zeros_like(self.ex_width)).argmax(1)   # 가장 넓은 열린 출구

    def _geo_dist(self, src):
        """src(bool [B,X,Y]) 에서 자유 셀을 따라 4-이웃 측지 거리(m). 벽은 inf."""
        B, NX, NY = src.shape
        inf = torch.full((), float("inf"), device=src.device)
        d = torch.where(src & ~self.WALL, torch.zeros((), device=src.device), inf).expand(B, NX, NY).clone()
        for _ in range(NX + NY):
            p = F.pad(d.unsqueeze(1), (1, 1, 1, 1), value=float("inf")).squeeze(1)
            n = torch.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]], 0).amin(0) + self.dx
            n = torch.where(self.WALL, inf, n)
            nd = torch.minimum(d, n)
            if torch.equal(nd, d): break
            d = nd
        return d

    def _def(self):
        """출구별 '화원 권역까지의 거리' [B,K] = 화원 2 m 권역 자유 셀 위 DIST_E[k] 의 최소."""
        out = torch.full((self.B, MAX_EXITS), float("inf"), device=self.dev)
        near = torch.isfinite(self.FDIST) & (self.FDIST <= self.dx * 0.5)
        for b in range(self.B):
            m = near[b]
            if m.any():
                out[b] = self.DIST_E[b][:, m].amin(1)
        return out

    def _wall_dist(self, wall):
        B, NX, NY = wall.shape
        d = torch.where(wall, torch.zeros((), device=wall.device), torch.full((), float("inf"), device=wall.device)).expand(B, NX, NY).clone()
        for _ in range(max(NX, NY)):
            p = F.pad(d.unsqueeze(1), (1, 1, 1, 1), value=float("inf")).squeeze(1)
            n = torch.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]], 0).amin(0) + self.dx
            nd = torch.minimum(d, n)
            if torch.equal(nd, d): break
            d = nd
        return d

    @staticmethod
    def _grad(f):
        gx = torch.zeros_like(f); gy = torch.zeros_like(f)
        gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2; gy[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / 2
        return torch.nan_to_num(gx, 0.0, 0.0, 0.0), torch.nan_to_num(gy, 0.0, 0.0, 0.0)

    # ── 조회(에이전트 단위, 케이스 인덱스 ca) ──────────────────────────────────
    def _cell(self, pos):
        ix = ((pos[:, 0] - self.x0[self.ca]) / self.dx).round().long().clamp(0, self.NX - 1)
        iy = ((pos[:, 1] - self.y0[self.ca]) / self.dx).round().long().clamp(0, self.NY - 1)
        return ix, iy

    def _frame(self, t):
        return (t / self.time_scale / self.frame_dt[self.ca]).clamp(0, self.NT - 1.001)

    def _sample_field(self, A_, pos, t):
        ix, iy = self._cell(pos); fi = self._frame(t); f0 = fi.floor().long(); w = fi - f0
        v0 = A_[self.ca, f0, ix, iy].float(); v1 = A_[self.ca, (f0 + 1).clamp(max=self.NT - 1), ix, iy].float()
        return v0 * (1 - w) + v1 * w

    def _lookup2d(self, A_, pos, idx=None):
        """A_ 의 첫 축을 idx(기본 self.ca)로, 좌표 원점은 항상 에이전트의 케이스(self.ca)."""
        idx = self.ca if idx is None else idx
        if A_.dtype == torch.bool:
            ix, iy = self._cell(pos); return A_[idx, ix, iy]
        fx = ((pos[:, 0] - self.x0[self.ca]) / self.dx).clamp(0, self.NX - 1.001)
        fy = ((pos[:, 1] - self.y0[self.ca]) / self.dx).clamp(0, self.NY - 1.001)
        x0 = fx.floor().long(); y0 = fy.floor().long(); wx = fx - x0; wy = fy - y0
        if A_.dim() == 4: wx = wx.unsqueeze(-1); wy = wy.unsqueeze(-1)
        return (A_[idx, x0, y0] * (1 - wx) * (1 - wy) + A_[idx, x0 + 1, y0] * wx * (1 - wy)
                + A_[idx, x0, y0 + 1] * (1 - wx) * wy + A_[idx, x0 + 1, y0 + 1] * wx * wy)

    def _lookup_exit(self, A_, pos, k):
        """A_ [B,K,X,Y(,2)] 를 에이전트의 목표 출구 k 에서 조회."""
        Bk = self.ca * MAX_EXITS + k
        return self._lookup2d(A_.flatten(0, 1), pos, idx=Bk)

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    def reset(self, reload=True, mask_cases=None):
        """mask_cases [B] bool: 그 케이스만 재시작(에이전트 전부). 없으면 전체."""
        if reload and mask_cases is None:
            self._load_batch()
            if hasattr(self, "vel"): del self.vel
        A, B, N, d = self.A, self.B, self.N, self.dev
        if mask_cases is None: mask_cases = torch.ones(B, dtype=torch.bool, device=d)
        mask = mask_cases[self.ca]
        first = not hasattr(self, "vel")
        if first:
            self.pos = torch.zeros((A, 2), device=d); self.vel = torch.zeros((A, 2), device=d)
            self.orient = torch.tensor([[1.0, 0.0]], device=d).expand(A, 2).clone()
            z = torch.zeros(A, device=d)
            for k in ("fed_tox", "fed_heat", "ks_avg", "t", "step_n", "stuck_t", "t_macro", "t_enter", "idle_t", "t_exit", "down_t", "wf_theta"):
                setattr(self, k, z.clone())
            self.haz_speed = torch.ones(A, device=d)
            for k in ("done", "exited", "down", "dead", "trapped", "active", "rescued", "switched"):
                setattr(self, k, torch.zeros(A, dtype=torch.bool, device=d))
            self.role = torch.zeros(A, dtype=torch.long, device=d)
            self.ward = torch.full((A,), -1, dtype=torch.long, device=d)          # guardian → child 인덱스
            self.guard = torch.full((A,), -1, dtype=torch.long, device=d)         # child → guardian
            self.attached = torch.full((A,), -1, dtype=torch.long, device=d)      # 피구조자 → 구조자
            self.carrying = torch.full((A,), -1, dtype=torch.long, device=d)      # 구조자 → 피구조자
            self.tgt_exit = torch.zeros(A, dtype=torch.long, device=d)
            self.rescue_tgt = torch.full((A,), -1, dtype=torch.long, device=d)
            self.mode = torch.zeros(A, dtype=torch.long, device=d)                # MODES 인덱스(판단층/규칙 매크로가 씀)
            self.pace = torch.ones(A, device=d)                                   # 판단층의 속도 배율(run 1.4 / walk 1.0 / slow 0.6 / stop 0)
            self.p_survive = torch.full((A,), 0.5, device=d)
            self.q_at_switch = torch.zeros(A, device=d)
        # 역할 배정(케이스마다 같은 구성): roster 가 있으면 그대로, 없으면 비율
        if first:
            self.sex = torch.zeros(A, dtype=torch.long, device=d); self.age = torch.full((A,), 35.0, device=d)
            self.cannot_walk = torch.zeros(A, dtype=torch.bool, device=d); self.v0mult = torch.ones(A, device=d)
        if self.roster is not None:
            assert len(self.roster) == N, "roster 길이는 에이전트 수와 같아야"
            role_b = torch.tensor([ROLES.index(r["role"]) for r in self.roster], device=d)
            ward_b = torch.full((N,), -1, dtype=torch.long, device=d); guard_b = ward_b.clone()
            for i, r in enumerate(self.roster):
                if r.get("guardian_of") is not None:
                    ward_b[i] = int(r["guardian_of"]); guard_b[int(r["guardian_of"])] = i
            n_g = int((role_b == R_GUARD).sum()); n_c = 0                         # child 자리 지정은 roster 가 한다
        else:
            n_g = int(round(N * self.roles["guardian"])); n_c = n_g; n_e = int(round(N * self.roles["elderly"]))
            n_f = min(self.n_ff, max(0, N - n_g - n_c - n_e - 4))
            roles = [R_GUARD] * n_g + [R_CHILD] * n_c + [R_ELDER] * n_e + [R_FF] * n_f
            roles += [R_ADULT] * (N - len(roles))
            role_b = torch.tensor(roles, device=d)
            ward_b = torch.full((N,), -1, dtype=torch.long, device=d); guard_b = ward_b.clone()
            for i in range(n_g):
                ward_b[i] = n_g + i; guard_b[n_g + i] = i
        # 출발 위치: 실내 자유·도달 가능·벽 0.4 m·화원 2 m (spawn 이 있으면 그 원환 안)
        ok = self.FREE & torch.isfinite(self.DIST) & (self.DIST >= 2.0) & (self.WDIST >= 0.4)
        for b in torch.nonzero(mask_cases).flatten().tolist():
            xs, ys = torch.nonzero(ok[b], as_tuple=True)
            cand = torch.stack([xs, ys], -1).float() * self.dx + torch.stack([self.x0[b], self.y0[b]])
            far = (cand - self.fire_xy[b]).norm(dim=-1) >= 2.0
            cand = cand[far] if far.any() else cand
            if self.spawn is not None:
                cx, cy, r0, r1 = self.spawn; rr = (cand - torch.tensor([cx, cy], device=d)).norm(dim=-1)
                ring = (rr >= r0) & (rr <= r1)
                if ring.any(): cand = cand[ring]
            sel = cand[torch.randint(0, cand.shape[0], (N,), generator=self.g, device=d)]
            sl = slice(b * N, (b + 1) * N)
            # child 는 보호자 옆 0.5 m
            if self.roster is None:
                sel[n_g:n_g + n_c] = sel[:n_g] + torch.randn((n_g, 2), generator=self.g, device=d) * 0.3
            else:
                for i, r in enumerate(self.roster):
                    if r.get("guardian_of") is not None:
                        sel[int(r["guardian_of"])] = sel[i] + torch.randn((2,), generator=self.g, device=d) * 0.3
            self.pos[sl] = sel
            self.role[sl] = role_b; base = b * N
            self.ward[sl] = torch.where(ward_b >= 0, ward_b + base, ward_b); self.guard[sl] = torch.where(guard_b >= 0, guard_b + base, guard_b)
            if self.roster is not None:
                self.sex[sl] = torch.tensor([1 if r.get("sex") == "female" else 0 for r in self.roster], device=d)
                self.age[sl] = torch.tensor([float(r.get("age", 35)) for r in self.roster], device=d)
                self.cannot_walk[sl] = torch.tensor([bool(r.get("cannot_walk", False)) for r in self.roster], device=d)
                self.v0mult[sl] = torch.tensor([SEX_V0.get(r.get("sex", "male"), 1.0) * float(r.get("v0mult", 1.0)) for r in self.roster], device=d)
            # 소방관: 주 출구 밖에서 t_enter 에 진입
            ff = torch.nonzero(role_b == R_FF).flatten() + base
            if len(ff):
                k = self.main_exit[b]
                dk = self.DIST_E[b, k]; near = self.FREE[b] & (dk >= 1.0) & (dk <= 2.5)
                if not near.any(): near = self.FREE[b] & (dk <= 4.0)
                xs, ys = torch.nonzero(near, as_tuple=True)
                cn = torch.stack([xs, ys], -1).float() * self.dx + torch.stack([self.x0[b], self.y0[b]])
                self.pos[ff] = cn[torch.randint(0, cn.shape[0], (len(ff),), generator=self.g, device=d)]
                self.t_enter[ff] = self.ff_enter[0] + torch.rand(len(ff), generator=self.g, device=d) * (self.ff_enter[1] - self.ff_enter[0])
        z1 = torch.zeros(A, device=d); m2 = mask.unsqueeze(-1)
        self.vel = torch.where(m2, torch.zeros_like(self.vel), self.vel)
        self.orient = torch.where(m2, torch.tensor([[1.0, 0.0]], device=d).expand(A, 2), self.orient)
        for k in ("fed_tox", "fed_heat", "ks_avg", "step_n", "stuck_t", "t_macro", "idle_t", "q_at_switch", "t_exit", "down_t", "wf_theta"):
            setattr(self, k, torch.where(mask, z1, getattr(self, k)))
        self.haz_speed = torch.where(mask, torch.ones_like(z1), self.haz_speed)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)                # 케이스 공용 시계(에이전트마다 같은 값)
        t_start = torch.rand(A, generator=self.g, device=d) * self.premove         # 망설임(출발 지연) 0~premove
        self.t_enter = torch.where(mask & (self.role != R_FF), t_start, self.t_enter)
        for k in ("done", "exited", "down", "dead", "trapped", "rescued", "switched"):
            setattr(self, k, torch.where(mask, torch.zeros_like(self.done), getattr(self, k)))
        self.active = torch.where(mask, torch.zeros_like(self.active), self.active)  # t ≥ t_enter 에 활성화(소방관 60~120 s)
        self.attached = torch.where(mask, torch.full_like(self.attached, -1), self.attached)
        self.carrying = torch.where(mask, torch.full_like(self.carrying, -1), self.carrying)
        self.rescue_tgt = torch.where(mask, torch.full_like(self.rescue_tgt, -1), self.rescue_tgt)
        self.mode = torch.where(mask, torch.zeros_like(self.mode), self.mode)
        self.pace = torch.where(mask, torch.ones_like(self.pace), self.pace)
        self.v0 = ROLE_V0.to(d)[self.role] * self.v0mult; self.rad = ROLE_RAD.to(d)[self.role]
        # 보행 불능 부상자: 처음부터 쓰러진 상태(구조 대상). 위험 노출 10 s 사망 규칙은 열·연기가 닿을 때만 돈다.
        self.down = torch.where(mask & self.cannot_walk, torch.ones_like(self.down), self.down)
        self._exit_summary(); self._macro(force=mask)
        return self._obs()

    def snapshot(self):
        """모든 에이전트의 (t, x, y, 상태) — 0.5 s 마다 CSV 로 내보내는 용도."""
        st = torch.where(self.exited, 3, torch.where(self.dead, 4, torch.where(self.down, 2, torch.where(self.active, 1, 0))))
        return float(self.t[0]), self.pos.detach().cpu().numpy(), st.cpu().numpy(), self.mode.cpu().numpy(), self.tgt_exit.cpu().numpy()

    def ctrl_mask(self):
        """정책이 제어하는 에이전트(활성·미종료·child 아님·피구조 아님·안 쓰러짐)."""
        return self.active & ~self.done & (self.role != R_CHILD) & (self.attached < 0) & ~self.down

    def stats(self):
        """케이스 배치 집계(역할별 탈출·쓰러짐·사망, 구조, 가족 분리, 탈출 시각)."""
        out = {}
        for r, nm in enumerate(ROLES):
            m = self.role == r
            if m.any():
                out[nm] = dict(n=int(m.sum()), exit=float(self.exited[m].float().mean()), down=float(self.down[m].float().mean()),
                               dead=float(self.dead[m].float().mean()))
        out["rescued"] = int(self.rescued.sum()); out["exit_all"] = float(self.exited.float().mean())
        out["down_all"] = float(self.down.float().mean()); out["dead_all"] = float(self.dead.float().mean())
        ex = self.exited
        out["t_exit_p50"] = float(self.t_exit[ex].median()) if ex.any() else float("nan")
        out["fed_p50"] = float((self.fed_tox + self.fed_heat).median())
        return out

    # ── 출구 요약·매크로 ───────────────────────────────────────────────────────
    def _exit_summary(self):
        """s_e(안전), q_e(대기 s), 각 케이스 출구별 — 현재 t 의 물리장(케이스 시각은 에이전트 평균)."""
        B, K, d = self.B, MAX_EXITS, self.dev
        c = self.ex_center                                                # [B,K,2]
        cflat = c.flatten(0, 1); cb = torch.arange(B, device=d).repeat_interleave(K)
        tb = torch.zeros(B, device=d).index_add_(0, self.ca, self.t) / self.N
        fi = (tb[cb] / self.time_scale / self.frame_dt[cb]).clamp(0, self.NT - 1.001).round().long()
        ix = ((cflat[:, 0] - self.x0[cb]) / self.dx).round().long().clamp(0, self.NX - 1)
        iy = ((cflat[:, 1] - self.y0[cb]) / self.dx).round().long().clamp(0, self.NY - 1)
        Te = self.T[cb, fi, ix, iy].float(); Ke = self.KS[cb, fi, ix, iy].float()
        # 화염 2 m 내: 출구 중심 주변 ±10셀 최대
        r = int(round(2.0 / self.dx)); ar = torch.arange(-r, r + 1, device=d)
        gx = (ix[:, None, None] + ar[None, :, None]).clamp(0, self.NX - 1); gy = (iy[:, None, None] + ar[None, None, :]).clamp(0, self.NY - 1)
        Fe = self.FIRE[cb[:, None, None], fi[:, None, None], gx, gy].float().amax((1, 2))
        s = torch.exp(-(Te - 40).clamp_min(0) / 60.0) * torch.exp(-Ke) * (Fe < FIRE_THR).float()
        self.ex_safe = s.view(B, K) * self.ex_open.float()
        # 대기 인원: 출구 중심 3 m 안 활성·미탈출 에이전트
        dpos = self.pos.view(B, self.N, 1, 2) - c.view(B, 1, K, 2)
        near = (dpos.norm(dim=-1) < 3.0) & (self.active & ~self.done).view(B, self.N, 1)
        n_e = near.sum(1).float()                                          # [B,K]
        self.ex_q = n_e / (1.3 * self.ex_width)
        self.ex_T, self.ex_K = Te.view(B, K), Ke.view(B, K)

    def _nav_dir(self, kidx):
        """목표 출구 kidx[A] 거리장의 8-이웃 내리막 방향(단위벡터). 벽 셀은 inf 라 못 고르고, 대각선은 양옆이 자유일 때만.
        ±3셀 창 최소값 방식은 1셀 두께 파티션 너머 셀을 골라 벽에 박혀 정체했다(사무실 40×25 실측: 절반 미탈출)."""
        A, d = self.A, self.dev
        ix, iy = self._cell(self.pos)
        Dk = self.DIST_E.flatten(0, 1); idx = self.ca * MAX_EXITS + kidx
        here = Dk[idx, ix, iy]
        best = here.clone(); bx = torch.zeros(A, device=d); by = torch.zeros(A, device=d)
        def at(ox, oy):
            jx = (ix + ox).clamp(0, self.NX - 1); jy = (iy + oy).clamp(0, self.NY - 1)
            return Dk[idx, jx, jy]
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            dv = at(ox, oy)
            if ox and oy:                                                        # 코너 끼임 방지
                dv = torch.where(torch.isfinite(at(ox, 0)) & torch.isfinite(at(0, oy)), dv, torch.full_like(dv, float("inf")))
            better = dv < best - 1e-6
            best = torch.where(better, dv, best)
            n = math.sqrt(ox * ox + oy * oy)
            bx = torch.where(better, torch.full_like(bx, ox / n), bx); by = torch.where(better, torch.full_like(by, oy / n), by)
        return torch.stack([bx, by], -1)

    def route_fire(self):
        """[A,K] bool: 목표 출구 k 로 가는 측지 경로가 화원 2 m 권역을 지나는가."""
        fa = self._lookup2d(self.FDIST, self.pos)
        De = torch.stack([self._lookup_exit(self.DIST_E, self.pos, torch.full((self.A,), k, device=self.dev, dtype=torch.long)) for k in range(MAX_EXITS)], 1)
        via = fa.unsqueeze(1) + self.DEF[self.ca]
        return torch.isfinite(De) & torch.isfinite(via) & (via - De < 3.0) & (fa.unsqueeze(1) > 0.5)

    def _macro(self, force=None):
        """규칙 매크로(v1): 목표 출구 = argmin DIST_E/max(s_e,0.05) + 1.2·q_e. 소방관·구조 대상 규칙 포함. 5 s 주기."""
        A, d = self.A, self.dev
        due = (self.t - self.t_macro >= self.macro_dt)
        if force is not None: due = due | force
        due &= self.active & ~self.done
        if not due.any(): return
        # 에이전트별 출구 비용 [A,K]
        De = torch.stack([self._lookup_exit(self.DIST_E, self.pos, torch.full((A,), k, device=d, dtype=torch.long)) for k in range(MAX_EXITS)], 1)
        s = self.ex_safe[self.ca]; q = self.ex_q[self.ca]; openm = self.ex_open[self.ca]
        # 경로가 화원 곁을 지나면 화원 세기에 비례한 벌점(미터 환산, 최대 60 m) — 출구 셀만 보던 s_e 의 사각지대
        fi_ = self._frame(self.t).round().long(); Qn = self.HRR[self.ca, fi_.clamp(max=self.NT - 1)]
        pen = self.route_fire().float() * (Qn / 50.0).clamp(0, 60.0).unsqueeze(1)
        cost = De / s.clamp_min(0.05) + 1.2 * q + pen
        cost = torch.where(openm & torch.isfinite(De), cost, torch.full_like(cost, 1e9))
        new = cost.argmin(1)
        # 소방관: 자기 진입 출구로 복귀(구조 후) — 기본 목표
        ff = self.role == R_FF
        new = torch.where(ff, self.main_exit[self.ca], new)
        changed = due & (new != self.tgt_exit) & (self.step_n > 0)
        self.switched |= changed
        self.q_at_switch = torch.where(changed, q.gather(1, self.tgt_exit[:, None]).squeeze(1) - q.gather(1, new[:, None]).squeeze(1), self.q_at_switch)
        self.tgt_exit = torch.where(due, new, self.tgt_exit)
        # 구조 대상: 위험 피난자(활성·미탈출·미확보, FED≥0.3 또는 쓰러짐 또는 정체) — 소방관 15 m / 성인(FED<0.3) 5 m
        fedv = self.fed_tox + self.fed_heat
        victim = self.active & ~self.done & (self.attached < 0) & (self.role != R_FF) & (
            (fedv >= VICTIM_FED) | self.down | ((self.stuck_t >= STUCK_T) & (self.ks_avg > 0.3)))
        victim_ad = victim & self.down                                     # 성인은 쓰러진 사람만
        pv = self.pos.view(self.B, self.N, 2)
        dd = (pv[:, :, None] - pv[:, None]).norm(dim=-1)                   # [B,N,N]
        dd.diagonal(dim1=1, dim2=2).fill_(1e9)
        dff = torch.where(victim.view(self.B, 1, self.N), dd, torch.full_like(dd, 1e9))
        dad = torch.where(victim_ad.view(self.B, 1, self.N), dd, torch.full_like(dd, 1e9))
        jff = dff.argmin(2).flatten(); mff = dff.amin(2).flatten()
        jad = dad.argmin(2).flatten(); mad = dad.amin(2).flatten()
        can_ff = ff & (mff < 15.0)
        can_ad = (self.role == R_ADULT) & (fedv < 0.3) & (mad < 5.0)
        jmin = torch.where(ff, jff, jad)
        rt = torch.where((can_ff | can_ad) & (self.carrying < 0), jmin + self.ca * self.N, torch.full_like(jmin, -1))
        self.rescue_tgt = torch.where(due, rt, self.rescue_tgt)
        # 규칙 모드: 구조 대상 있으면 rescue, 보호자가 아이와 3 m 밖이면 escort, 아니면 evacuate
        gd = (self.role == R_GUARD) & (self.ward >= 0)
        w = self.ward.clamp_min(0)
        far = gd & self.active[w] & ~self.done[w] & ((self.pos[w] - self.pos).norm(dim=-1) > 3.0)
        rule_mode = torch.where(self.rescue_tgt >= 0, torch.full_like(self.mode, M_RESCUE),
                                torch.where(far, torch.full_like(self.mode, M_ESCORT), torch.full_like(self.mode, M_EVAC)))
        self.mode = torch.where(due, rule_mode, self.mode)
        self.t_macro = torch.where(due, self.t, self.t_macro)
        # 판단층(Jev 형식): 규칙 결과 위에 LLM 결정을 덮어쓴다 — 제어 가능한 에이전트만
        if self.macro_mode == "jev" and getattr(self, "decider", None) is not None:
            idx = torch.nonzero(due & (self.role != R_CHILD) & (self.attached < 0) & ~self.down).flatten().tolist()
            if idx:
                for r in self.decider.decide(self, idx):
                    i = r["agent"]
                    self.mode[i] = MODES.index(r["mode"])
                    if r["target_exit_k"] is not None and bool(self.ex_open[self.ca[i], r["target_exit_k"]]):
                        k_new = r["target_exit_k"]
                        if k_new != int(self.tgt_exit[i]):
                            self.switched[i] = True
                            self.q_at_switch[i] = float(self.ex_q[self.ca[i], self.tgt_exit[i]] - self.ex_q[self.ca[i], k_new])
                        self.tgt_exit[i] = k_new
                    self.rescue_tgt[i] = r["rescue_agent"] if r["mode"] == "rescue" else -1
                    self.p_survive[i] = r["p_survive"]
                    self.pace[i] = r.get("pace_mult", 1.0)

    # ── 힘 ───────────────────────────────────────────────────────────────────────
    def _wayfind_noise(self):
        """시야(=VIS_C/K_s, FDS VISIBILITY)가 짧을수록 출구 방향 인지 오차가 커진다. σ(vis) 는 10 m 에서 0, 1 m 에서 90°.
        오차각 wf_theta 는 스텝마다 AR(1) 로 갱신(상관 ~3 s)해 '헤매는' 궤적이 나오게 한다. 반환: 회전 [A,2,2] 용 cos/sin."""
        vis = VIS_C / self.ks_avg.clamp_min(1e-3)
        sig = ((VIS_FULL - vis) / (VIS_FULL - VIS_NONE)).clamp(0.0, 1.0) * (math.pi / 2)
        sig = torch.where(self.role == R_FF, sig * 0.3, sig)                           # 훈련·장비(열화상)
        a = math.exp(-STEP_DT / 3.0)
        noise = torch.randn(self.A, generator=self.g, device=self.dev) * sig * math.sqrt(1 - a * a)
        self.wf_theta = a * self.wf_theta + noise
        self.wf_theta = torch.where(sig <= 0, torch.zeros_like(self.wf_theta), self.wf_theta)
        return torch.cos(self.wf_theta), torch.sin(self.wf_theta)

    @staticmethod
    def _rot(v, c, s_):
        return torch.stack([v[:, 0] * c - v[:, 1] * s_, v[:, 0] * s_ + v[:, 1] * c], -1)

    def _wall_force(self, pos, vel, R=4):
        A, d = self.A, self.dev
        ix, iy = self._cell(pos)
        ar = torch.arange(-R, R + 1, device=d)
        gx = (ix[:, None, None] + ar[None, :, None]).clamp(0, self.NX - 1).expand(-1, -1, 2 * R + 1)
        gy = (iy[:, None, None] + ar[None, None, :]).clamp(0, self.NY - 1).expand(-1, 2 * R + 1, -1)
        iswall = self.WALL[self.ca[:, None, None], gx, gy]
        cx = gx.float() * self.dx + self.x0[self.ca][:, None, None]; cy = gy.float() * self.dx + self.y0[self.ca][:, None, None]
        dvx_c = pos[:, 0, None, None] - cx; dvy_c = pos[:, 1, None, None] - cy
        half = self.dx * 0.5
        dvx = torch.sign(dvx_c) * (dvx_c.abs() - half).clamp_min(0.0); dvy = torch.sign(dvy_c) * (dvy_c.abs() - half).clamp_min(0.0)
        dist = torch.sqrt(dvx ** 2 + dvy ** 2)
        inside = dist < 1e-6
        dvx = torch.where(inside, dvx_c, dvx); dvy = torch.where(inside, dvy_c, dvy)
        dist = torch.where(inside, torch.full_like(dist, 1e-3), dist)
        corner = (dvx.abs() > 1e-6) & (dvy.abs() > 1e-6) & ~inside
        sect = [(dvx_c < 0) & (dvx_c.abs() >= dvy_c.abs()), (dvx_c > 0) & (dvx_c.abs() >= dvy_c.abs()),
                (dvy_c < 0) & (dvy_c.abs() > dvx_c.abs()), (dvy_c > 0) & (dvy_c.abs() > dvx_c.abs())]
        rad = self.rad
        fx = torch.zeros(A, device=d); fy = torch.zeros(A, device=d); push_sum = torch.zeros(A, device=d)
        best_d = torch.full((A,), 1e9, device=d); wn = torch.zeros((A, 2), device=d); ar_e = torch.arange(A, device=d)
        for sm in sect:
            dm = torch.where(iswall & sm & ~corner, dist, torch.full_like(dist, 1e9)).flatten(1)
            j = dm.argmin(1); dj = dm[ar_e, j]; has = dj < 1e8
            nx_ = dvx.flatten(1)[ar_e, j] / dj.clamp_min(1e-6); ny_ = dvy.flatten(1)[ar_e, j] / dj.clamp_min(1e-6)
            push = P["a_obst"] * torch.exp((rad - dj) / P["b"]); close = dj < rad
            push = push + torch.where(close, P["k"] * (rad - dj), torch.zeros_like(dj))
            tx, ty = -ny_, nx_; vt = vel[:, 0] * tx + vel[:, 1] * ty
            fric = torch.where(close, P["kappa"] * (rad - dj) * vt, torch.zeros_like(dj)); h = has.float()
            fx = fx + (nx_ * push + tx * fric) * h; fy = fy + (ny_ * push + ty * fric) * h
            push_sum = push_sum + torch.where(close, push, torch.zeros_like(push)) * h
            better = has & (dj < best_d); best_d = torch.where(better, dj, best_d)
            wn = torch.where(better.unsqueeze(-1), torch.stack([nx_, ny_], -1), wn)
        return torch.stack([fx, fy], -1), push_sum, wn

    def _agent_force(self, pos, vel, alive):
        """rust_evac sfm_force 에이전트 쌍 — 케이스 안 N×N 밀집. 반환 (힘/질량 [A,2], 접촉 push 합, 이웃수 2.5 m)."""
        B, N = self.B, self.N
        p = pos.view(B, N, 2); v = vel.view(B, N, 2); al = alive.view(B, N)
        dvec = p[:, :, None] - p[:, None]                                  # i ← j
        dist = dvec.norm(dim=-1).clamp_min(1e-6)
        n = dvec / dist.unsqueeze(-1)
        rsum = self.rad.view(B, N, 1) + self.rad.view(B, 1, N)
        valid = al[:, :, None] & al[:, None] & (dist < AA_CUTOFF)
        valid.diagonal(dim1=1, dim2=2).fill_(False)
        over = (rsum - dist).clamp_min(0)
        push = A_AGENT * torch.exp((rsum - dist) / B_AGENT) + P["k"] * over
        t = torch.stack([-n[..., 1], n[..., 0]], -1)
        dv = v[:, None] - v[:, :, None]                                    # v_j − v_i
        fric = P["kappa"] * over * (dv * t).sum(-1)
        f = (n * push.unsqueeze(-1) + t * fric.unsqueeze(-1)) * valid.unsqueeze(-1).float()
        contact = (over > 0).float() * push * valid.float()
        return f.sum(2).view(-1, 2) / P["mass"], contact.sum(2).flatten(), valid.sum(2).flatten().float()

    def _hazard(self, dt):
        T = self._sample_field(self.T, self.pos, self.t); KS = self._sample_field(self.KS, self.pos, self.t)
        CO = self._sample_field(self.CO, self.pos, self.t); O2 = self._sample_field(self.O2, self.pos, self.t)
        tox, heat = fed_rates(T, CO, torch.zeros_like(T), O2)
        live = (~self.done & (self.active | (self.role != R_FF))).float()          # 출발 전에도 실내라 노출된다
        self.fed_tox = self.fed_tox + tox * ROLE_TOX.to(self.dev)[self.role] * dt / 60.0 * live
        self.fed_heat = self.fed_heat + heat * ROLE_HEAT.to(self.dev)[self.role] * dt / 60.0 * live
        self.ks_avg = self.ks_avg + (KS - self.ks_avg) * min(dt / P["exposure_tau"], 1.0)
        slope = ROLE_KSLOPE.to(self.dev)[self.role]
        self.haz_speed = (1.0 - slope * self.ks_avg.clamp_min(0)).clamp(P["speed_min"], 1.0)
        dpx = P["perception"]
        def H(p):
            Tq = self._sample_field(self.T, p, self.t); Kq = self._sample_field(self.KS, p, self.t)
            return (Tq - P["temp_ref"]).clamp_min(0) / P["temp_ref"] + Kq.clamp_min(0) / P["ks_ref"]
        H0 = H(self.pos)
        def Hlos(dirv):
            # 시선 제한: 1·2·3 m 표본 중 벽을 만나기 전 가장 먼 것. 벽 너머 화재실의 열이 복도 보행자를 밀지 않게.
            out = H0.clone(); blocked = torch.zeros_like(H0, dtype=torch.bool)
            for r_ in (1.0, 2.0, dpx):
                p_ = self.pos + dirv * r_
                ix_, iy_ = self._cell(p_); w_ = self.WALL[self.ca, ix_, iy_]
                blocked |= w_
                out = torch.where(blocked, out, H(p_))
            return out
        ex = torch.tensor([1.0, 0.0], device=self.dev); ey = torch.tensor([0.0, 1.0], device=self.dev)
        g = torch.stack([(Hlos(ex) - Hlos(-ex)) / (2 * dpx), (Hlos(ey) - Hlos(-ey)) / (2 * dpx)], -1)
        push = g * (-P["repulsion"] * dpx); m = push.norm(dim=-1, keepdim=True)
        return torch.where(m > P["push_cap"], push * (P["push_cap"] / m.clamp_min(1e-9)), push), T, KS

    def _los(self, p, q, r_max=None, n=8):
        """p→q 선분 위 n 점이 전부 비벽이면 True. r_max 를 넘는 거리는 False."""
        ok = torch.ones(p.shape[0], dtype=torch.bool, device=self.dev)
        for k in range(1, n + 1):
            s_ = p + (q - p) * (k / (n + 1.0))
            ix, iy = self._cell(s_); ok &= ~self.WALL[self.ca, ix, iy]
        if r_max is not None: ok &= (q - p).norm(dim=-1) <= r_max
        return ok

    # ── 스텝 ────────────────────────────────────────────────────────────────────
    def _desired(self, action):
        """정책 행동 → (e0, mult). child·피구조자·쓰러짐은 환경 규칙이 덮어쓴다."""
        A, d = self.A, self.dev
        e0 = DIRS.to(d)[action // 4]; mult = MULTS.to(d)[action % 4]
        # child: 보호자(없으면 최근접 성인) 추종 — 목표점 = 보호자 위치 − 0.6·보호자 진행방향
        ch = self.role == R_CHILD
        if ch.any():
            g = self.guard.clamp_min(0)
            g_ok = (self.guard >= 0) & self.active[g] & ~self.done[g] & ~self.down[g]
            tgt = self.pos[g] - 0.6 * self.orient[g]
            dv = tgt - self.pos; dist = dv.norm(dim=-1)
            e_f = dv / dist.clamp_min(1e-6).unsqueeze(-1)
            m_f = torch.where(dist > 2.0, torch.full_like(dist, 1.4), torch.where(dist > 0.5, torch.ones_like(dist), torch.zeros_like(dist)))
            # 보호자가 안 보이거나 4 m 밖: 보호자의 목표 출구 거리장을 따라간다(직선 추종은 벽에 걸린다).
            # 보호자보다 출구에 가까우면(앞서면) 기다린다.
            self.tgt_exit = torch.where(ch & (self.guard >= 0), self.tgt_exit[g], self.tgt_exit)
            see = self._los(self.pos, self.pos[g]) & (dist < 4.0)
            exd = self._nav_dir(self.tgt_exit)
            d_me = self._lookup_exit(self.DIST_E, self.pos, self.tgt_exit); d_g = self._lookup_exit(self.DIST_E, self.pos[g], self.tgt_exit)
            ahead = d_me < d_g - 0.5
            e_path = exd; m_path = torch.where(ahead, torch.zeros_like(dist), torch.ones_like(dist))
            e_f = torch.where(see.unsqueeze(-1), e_f, e_path); m_f = torch.where(see, m_f, m_path)
            # 보호자 없음: 목표 출구 방향으로 느리게(0.6) — 최근접 성인 추종의 근사
            e_f = torch.where(g_ok.unsqueeze(-1), e_f, exd); m_f = torch.where(g_ok, m_f, torch.full_like(m_f, 0.6))
            e0 = torch.where(ch.unsqueeze(-1), e_f, e0); mult = torch.where(ch, m_f, mult)
        # 보호자: 피보호자와 동행 — 피보호자가 3 m 밖이면 되돌아간다(속도 1.0), 아니면 정책대로(v0 는 child 속도로 제한)
        gd = (self.role == R_GUARD) & (self.ward >= 0)
        if gd.any():
            w = self.ward.clamp_min(0)
            w_ok = gd & self.active[w] & ~self.done[w] & (self.attached[w] < 0)
            dv = self.pos[w] - self.pos; dist = dv.norm(dim=-1)
            back = w_ok & (dist > 3.0)
            e0 = torch.where(back.unsqueeze(-1), dv / dist.clamp_min(1e-6).unsqueeze(-1), e0)
            mult = torch.where(back, torch.ones_like(mult), mult)
        # 판단층이 준 속도(pace)를 기본으로 — 규칙 매크로면 pace 가 1.0 이라 영향 없음
        if self.macro_mode == "jev":
            mult = self.pace
        # 모드 효과: wait/shelter 정지, breakthrough 달림(1.4), escort 는 아이 쪽으로(3 m 밖일 때)
        hold = (self.mode == M_WAIT) | (self.mode == M_SHELTER)
        mult = torch.where(hold, torch.zeros_like(mult), mult)
        mult = torch.where(self.mode == M_BREAK, torch.full_like(mult, 1.4), mult)
        esc = (self.mode == M_ESCORT) & (self.role == R_GUARD) & (self.ward >= 0)
        if esc.any():
            w = self.ward.clamp_min(0); dv = self.pos[w] - self.pos; dist = dv.norm(dim=-1)
            go = esc & self.active[w] & ~self.done[w] & (dist > 1.5)
            e0 = torch.where(go.unsqueeze(-1), dv / dist.clamp_min(1e-6).unsqueeze(-1), e0)
            mult = torch.where(go, torch.ones_like(mult), mult)
        # 쓰러짐: 정지
        mult = torch.where(self.down, torch.zeros_like(mult), mult)
        return e0, mult

    def step(self, action):
        A, d = self.A, self.dev
        e0, mult = self._desired(action)
        rew = torch.zeros(A, device=d); contact = torch.zeros(A, device=d)
        fed0 = self.fed_tox + self.fed_heat
        # 활성화(출발 지연 경과·소방관 진입)
        self.active |= (self.t >= self.t_enter) & ~self.done
        alive = self.active & ~self.done
        d_tgt0 = self._lookup_exit(self.DIST_E, self.pos, self.tgt_exit)
        # 구조 대상 접근 셰이핑용 거리
        rt = self.rescue_tgt.clamp_min(0); has_rt = self.rescue_tgt >= 0
        d_rt0 = torch.where(has_rt, (self.pos[rt] - self.pos).norm(dim=-1), torch.zeros(A, device=d))
        # 동행·구조 속도 제한
        v0 = self.v0.clone()
        gd = (self.role == R_GUARD) & (self.ward >= 0)
        w = self.ward.clamp_min(0)
        escort = gd & self.active[w] & ~self.done[w] & ((self.pos[w] - self.pos).norm(dim=-1) < 3.0)
        v0 = torch.where(escort, torch.minimum(v0, self.v0[w]), v0)
        cr = self.carrying >= 0; ci = self.carrying.clamp_min(0)
        v0 = torch.where(cr, torch.where(self.down[ci], torch.full_like(v0, SPEED_CARRY), torch.full_like(v0, SPEED_ASSIST)), v0)
        inv_m = 1.0 / P["mass"]
        los = self._los(self.pos, self.fire_xy[self.ca], r_max=12.0)          # 화원이 보일 때만 복사력(벽 너머 화재실은 안 민다)
        # 위험장 표본·FED 는 스텝당 1회(물리장은 1 s 프레임), 벽 반발은 스텝당 2회 — 서브스텝마다 하면 GPU 스모크가 스텝당 0.3 s
        push, T, KS = self._hazard(STEP_DT)
        hit_exit = torch.zeros(A, dtype=torch.bool, device=d)
        for k_ in range(SUB):
            alive = self.active & ~self.done
            vdes = v0 * mult * self.haz_speed
            f = (e0 * vdes.unsqueeze(-1) - self.vel) / P["tau"]
            if k_ % (SUB // 2) == 0:
                fw, push_w, wn = self._wall_force(self.pos, self.vel)
            present = self.active & ~self.exited & (self.attached < 0)                 # 쓰러진 사람·시신도 자리를 차지한다
            fa, push_a, _ = self._agent_force(self.pos, self.vel, present)
            contact = contact + (push_w + push_a) * SUB_DT
            f = f + fw * inv_m + fa
            dv = self.pos - self.fire_xy[self.ca]; r = dv.norm(dim=-1).clamp_min(0.3); n = dv / r.unsqueeze(-1)
            fi_ = self._frame(self.t); f0_ = fi_.floor().long(); w_ = fi_ - f0_
            Qt = self.HRR[self.ca, f0_] * (1 - w_) + self.HRR[self.ca, (f0_ + 1).clamp(max=self.NT - 1)] * w_
            q = (P["chi_r"] * Qt) / (4 * math.pi * r * r)
            over = (q - ROLE_QTH.to(d)[self.role]).clamp_min(0)
            mag = P["sr"] * (over / P["qref"]).clamp_max(3.0) * (0.4 + 0.6 * (self.orient * (-n)).sum(-1).clamp_min(0))
            mag = mag * los.float()
            tot = n * mag.unsqueeze(-1) + push * (vdes / P["tau"]).unsqueeze(-1)          # 복사력 + 위험장 밀침
            cap = (P["push_cap"] * vdes / P["tau"]).unsqueeze(-1); tn = tot.norm(dim=-1, keepdim=True)
            tot = torch.where(tn > cap, tot * (cap / tn.clamp_min(1e-9)), tot)             # 합이 희망(1.0)을 못 넘게 — 각각 0.9 면 합 1.8 로 영구 정체(실측: 사무실 우측 절반 미탈출)
            f = f + tot
            nv = self.vel + f * SUB_DT
            sp = nv.norm(dim=-1, keepdim=True)
            nv = torch.where(sp > P["clamp_v"], nv * (P["clamp_v"] / sp.clamp_min(1e-9)), nv)
            nv = torch.where(alive.unsqueeze(-1), nv, torch.zeros_like(nv))
            npos = self.pos + nv * SUB_DT
            ix, iy = self._cell(npos); blocked = self.WALL[self.ca, ix, iy]
            if blocked.any():
                # 축 분리 슬라이딩(격자 벽): 전체 이동이 벽이면 x 만 → y 만 → 정지. 군중이 문기둥으로 밀 때
                # 법선 기반 슬라이딩은 최근접 벽 법선과 막힌 셀이 어긋나 정지를 남발했다(실측: 문 앞 교착).
                px = torch.stack([npos[:, 0], self.pos[:, 1]], -1); py = torch.stack([self.pos[:, 0], npos[:, 1]], -1)
                bx = self.WALL[self.ca, *self._cell(px)]; by = self.WALL[self.ca, *self._cell(py)]
                vx = torch.stack([nv[:, 0], torch.zeros_like(nv[:, 0])], -1); vy = torch.stack([torch.zeros_like(nv[:, 1]), nv[:, 1]], -1)
                use_x = blocked & ~bx; use_y = blocked & bx & ~by; stop = blocked & bx & by
                npos = torch.where(use_x.unsqueeze(-1), px, torch.where(use_y.unsqueeze(-1), py, torch.where(stop.unsqueeze(-1), self.pos, npos)))
                nv = torch.where(use_x.unsqueeze(-1), vx, torch.where(use_y.unsqueeze(-1), vy, torch.where(stop.unsqueeze(-1), torch.zeros_like(nv), nv)))
            # 피구조자는 구조자 뒤 0.5 m 에 붙어 이동
            att = self.attached >= 0; ai = self.attached.clamp_min(0)
            npos = torch.where(att.unsqueeze(-1), npos[ai] - 0.5 * self.orient[ai], npos)
            nv = torch.where(att.unsqueeze(-1), nv[ai], nv)
            self.pos = torch.where(alive.unsqueeze(-1), npos, self.pos); self.vel = nv
            hit_exit = hit_exit | self._lookup2d(self.EXIT, self.pos)           # 서브스텝마다 — 밀려서 0.3 m 뛰면 1셀 출구선을 건너뛴다
            ori = nv / nv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.orient = torch.where((nv.norm(dim=-1) > 1e-3).unsqueeze(-1), ori, self.orient)
            self.t = self.t + SUB_DT
        alive = self.active & ~self.done
        fed = self.fed_tox + self.fed_heat
        self._wf_cs = self._wayfind_noise()
        # 정체 타이머(구조 대상 판정): 속도 < 0.2 & 이동 의사 있음
        moving_intent = (mult > 0) & ~self.down
        self.stuck_t = torch.where(alive & moving_intent & (self.vel.norm(dim=-1) < 0.2), self.stuck_t + STEP_DT, torch.zeros_like(self.stuck_t))
        # 구조 확보: 구조자가 대상 1 m 안 → attach
        rt = self.rescue_tgt.clamp_min(0)
        can = alive & (self.rescue_tgt >= 0) & (self.carrying < 0) & (self.attached < 0) & (self.attached[rt] < 0) & alive[rt] & ~self.exited[rt]
        near = can & ((self.pos[rt] - self.pos).norm(dim=-1) < ATTACH_R)
        newly_attach = near.clone()
        if near.any():
            idx = torch.nonzero(near).flatten()
            for i in idx.tolist():                                     # 같은 대상 중복 확보 방지(순차)
                j = int(self.rescue_tgt[i])
                if int(self.attached[j]) < 0 and int(self.carrying[i]) < 0:
                    self.attached[j] = i; self.carrying[i] = j
                else:
                    newly_attach[i] = False
            self.rescue_tgt = torch.where(newly_attach, torch.full_like(self.rescue_tgt, -1), self.rescue_tgt)
        ctrl = alive & (self.role != R_CHILD) & (self.attached < 0) & ~self.down       # 이번 스텝에 정책 행동이 유효했던 에이전트
        # 종료 판정
        at_exit = hit_exit
        ff_leave = (self.carrying >= 0) | ((self.rescue_tgt < 0) & (self.t - self.t_enter > 30.0))
        at_exit = at_exit & ((self.role != R_FF) | ff_leave)
        at_exit = at_exit | (self._lookup2d(self.OUT, self.pos) & (self.role == R_FF) & (self.t - self.t_enter > 5.0))   # 밖으로 나간 소방관
        newly_exit = at_exit & alive
        # 피구조자: 구조자가 나가면 함께
        ci = self.carrying.clamp_min(0)
        newly_rescued = newly_exit & (self.carrying >= 0)
        victim_idx = ci[newly_rescued]
        newly_down = (fed >= P["fed_threshold"]) & alive & ~self.down & ~newly_exit
        # 쓰러진 뒤 위험 노출 시간: 열(T>40)·연기(K_s>0.2)·FED 증가 중 하나라도 있으면 누적
        T_here = self._sample_field(self.T, self.pos, self.t); K_here = self._sample_field(self.KS, self.pos, self.t)
        hazard_here = (T_here > 40.0) | (K_here > 0.2) | ((fed - fed0) > 1e-4)
        self.down_t = torch.where((self.down | newly_down) & alive & hazard_here, self.down_t + STEP_DT, self.down_t)
        newly_dead = ((fed >= DEAD_FED) | (self.down_t >= DEAD_T)) & alive & ~newly_exit
        newly_trap = (self.t >= self.t_end[self.ca]) & alive & ~newly_exit & ~newly_dead
        # ── 보상 ──
        R = self.R
        rew += R["time"] * STEP_DT * alive.float()
        rew += R["fed"] * (fed - fed0) * alive.float()
        # 접촉(압박): rust_evac k=1.2e5 N/m 라 5 cm 겹침이 6,000 N — 원값에 −0.005 를 곱하면 −30/스텝으로 모든 항을 압도해
        # 정책이 '제자리 정지'로 붕괴했다(M1_nofire 실측: mult 0/0.6 두 행동만). 100 N·s 단위로 정규화하고 −0.1/스텝에서 캡.
        rew += R["contact"] * (contact / 100.0).clamp(0, 10.0)
        # 잠재 셰이핑: Φ = −shaping·DIST_target
        d_tgt1 = self._lookup_exit(self.DIST_E, self.pos, self.tgt_exit)
        fin = torch.isfinite(d_tgt0) & torch.isfinite(d_tgt1)
        rew += torch.where(fin & alive, -R["shaping"] * (d_tgt1 - d_tgt0), torch.zeros_like(rew))
        # 탈출: 기본 + 청정 보너스 + 안전 출구 점수
        s_here = self.ex_safe[self.ca].gather(1, self.tgt_exit[:, None]).squeeze(1)
        ex_r = R["exit"] + R["clean"] * (1 - fed).clamp_min(0) + R["safe_exit"] * s_here
        # 보호자가 피보호자를 두고 혼자 나가면 절반
        w = self.ward.clamp_min(0)
        solo = (self.role == R_GUARD) & (self.ward >= 0) & ~self.exited[w] & ~self.dead[w]
        ex_r = torch.where(solo, ex_r * R["solo_exit"], ex_r)
        rew += ex_r * newly_exit.float()
        # 출구 변경 보상: 바꾼 뒤 실제 그 출구로 탈출, 변경 시 대기 차 ≥ 5 s
        rew += R["switch"] * (newly_exit & self.switched & (self.q_at_switch >= 5.0)).float()
        rew += R["down"] * newly_down.float() + R["dead"] * newly_dead.float() + R["trap"] * newly_trap.float()
        # 군중 밀집 체류
        _, _, nnb = self._agent_force(self.pos, self.vel, alive)
        rew += R["crowd"] * STEP_DT * (nnb > 3 * math.pi * AA_CUTOFF ** 2 / 4).float() * alive.float()   # 반경 2.5 m 안 >~15명 ≈ 3 인/m² 이상
        # 가족
        gd = (self.role == R_GUARD) & (self.ward >= 0)
        rew += R["ward_exit"] * (gd & newly_exit[w]).float() + R["ward_down"] * (gd & newly_down[w]).float()
        sep = gd & alive & alive[w] & ((self.pos[w] - self.pos).norm(dim=-1) > 3.0)
        rew += R["separation"] * STEP_DT * sep.float()
        # 구조
        ff = self.role == R_FF
        rew += torch.where(ff, R["attach_ff"], R["attach_adult"]) * newly_attach.float()
        rew += torch.where(ff, R["rescue_ff"], R["rescue_adult"]) * newly_rescued.float()
        has_rt = self.rescue_tgt >= 0; rt = self.rescue_tgt.clamp_min(0)
        d_rt1 = (self.pos[rt] - self.pos).norm(dim=-1)
        rew += torch.where(has_rt & alive, R["approach"] * (d_rt0 - d_rt1), torch.zeros_like(rew))
        idle = ff & alive & (self.rescue_tgt < 0) & (self.carrying < 0)
        rew += R["ff_idle"] * STEP_DT * idle.float()
        rew += R["fed"] * 0.0                                            # (소방관 자기 FED 는 위 fed 항이 이미 heat×0.2 로 계산)
        # 피구조자 탈출 처리
        if newly_rescued.any():
            self.rescued[victim_idx] = True; self.exited[victim_idx] = True; self.done[victim_idx] = True
            rew[victim_idx] += R["carried_exit"]
            self.attached[victim_idx] = -1; self.carrying[newly_rescued] = -1
        # 구조자가 쓰러지면 확보 해제(둘 다 위험)
        drop = newly_down & (self.carrying >= 0)
        if drop.any():
            vi = self.carrying[drop]; self.attached[vi] = -1; self.carrying[drop] = -1
        self.t_exit = torch.where(newly_exit, self.t, self.t_exit)
        if newly_rescued.any(): self.t_exit[victim_idx] = self.t[victim_idx]
        self.exited |= newly_exit; self.down |= newly_down; self.dead |= newly_dead; self.trapped |= newly_trap
        self.done |= newly_exit | newly_dead | newly_trap
        self.step_n += 1
        # 5 s 주기 매크로·출구 요약
        if int(self.step_n[0]) % max(1, int(round(min(self.macro_dt, MACRO_DT) / STEP_DT))) == 0:
            self._exit_summary()
        self._macro()
        info = dict(exited=self.exited.clone(), down=self.down.clone(), dead=self.dead.clone(), rescued=self.rescued.clone(),
                    t=self.t.clone(), t_exit=self.t_exit.clone(), fed=fed.clone(), ctrl=ctrl)
        case_done = self.done.view(self.B, self.N).all(1) | (self.done | ~self.active).view(self.B, self.N).all(1) & (self.t.view(self.B, self.N).amax(1) >= self.t_end)
        return self._obs(), rew, self.done.clone(), case_done, info

    # ── 관측 ────────────────────────────────────────────────────────────────────
    def _density(self):
        """활성 에이전트 밀도맵 [B,NX/2,NY/2] (0.4 m 셀 기준 인/셀)."""
        ds = 2
        ix, iy = self._cell(self.pos)
        al = (self.active & ~self.done)
        Hx, Hy = (self.NX + ds - 1) // ds, (self.NY + ds - 1) // ds
        idx = (self.ca * Hx + ix // ds) * Hy + iy // ds
        dens = torch.zeros(self.B * Hx * Hy, device=self.dev).index_add_(0, idx[al], torch.ones(int(al.sum()), device=self.dev))
        return dens.view(self.B, Hx, Hy), ds

    def _obs(self):
        A, d = self.A, self.dev
        DS = max(1, int(round(0.4 / self.dx)))
        R = PATCH * DS // 2
        ix, iy = self._cell(self.pos)
        ar = torch.arange(-R, R, DS, device=d)
        gx = (ix.unsqueeze(-1) + ar).clamp(0, self.NX - 1); gy = (iy.unsqueeze(-1) + ar).clamp(0, self.NY - 1)
        fi = self._frame(self.t).round().long()
        cb = self.ca[:, None, None]
        def take(A_): return A_[cb, fi[:, None, None], gx[:, :, None], gy[:, None, :]].float()
        def take2(A_): return A_[cb, gx[:, :, None], gy[:, None, :]]
        vis = 3.0 / self.ks_avg.clamp_min(3e-3)
        rr = torch.sqrt((ar.float() ** 2)[:, None] + (ar.float() ** 2)[None, :]) * self.dx
        vmask = (rr[None] <= vis[:, None, None]).float()
        dens, ds = self._density()
        dpatch = dens[cb, (gx // ds).clamp(max=dens.shape[1] - 1)[:, :, None], (gy // ds).clamp(max=dens.shape[2] - 1)[:, None, :]]
        patch = torch.stack([take(self.T) / 1000.0 * vmask, take(self.KS) / 5.0 * vmask, (take(self.FIRE) > FIRE_THR).float() * vmask,
                             take2(self.WALL).float(), (dpatch / 4.0).clamp_max(2.0) * vmask], 1)
        exdir = self._nav_dir(self.tgt_exit)
        if not hasattr(self, "_wf_cs"): self._wf_cs = self._wayfind_noise()
        exdir = self._rot(exdir, *self._wf_cs)                                         # 연기 속에서는 출구 방향을 틀리게 안다
        dist = self._lookup_exit(self.DIST_E, self.pos, self.tgt_exit); dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, 50.0))
        wd = self._lookup2d(self.WDIST, self.pos)
        base = torch.cat([self.vel / 2.0, exdir, (dist / 30.0).unsqueeze(-1), (wd / 3.0).clamp_max(1).unsqueeze(-1),
                          self.fed_tox.unsqueeze(-1), self.fed_heat.unsqueeze(-1), (self.ks_avg / 5).unsqueeze(-1),
                          (self.t / 60.0).unsqueeze(-1), self.haz_speed.unsqueeze(-1), self.orient], -1)          # 13
        role = F.one_hot(self.role, N_ROLE).float(); rv = torch.cat([role, (self.v0 / 1.5).unsqueeze(-1)], -1)     # 7
        # 출구 요약 8×7: valid, dist/50, dir(2), s_e, q_e/30, is_target
        De = torch.stack([self._lookup_exit(self.DIST_E, self.pos, torch.full((A,), k, device=d, dtype=torch.long)) for k in range(MAX_EXITS)], 1)
        De = torch.where(torch.isfinite(De), De, torch.full_like(De, 60.0))
        dvec = self.ex_center[self.ca] - self.pos.unsqueeze(1); dvec = dvec / dvec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        exs = torch.cat([self.ex_valid[self.ca].float().unsqueeze(-1), (De / 50.0).unsqueeze(-1), dvec,
                         self.ex_safe[self.ca].unsqueeze(-1), (self.ex_q[self.ca] / 30.0).unsqueeze(-1),
                         F.one_hot(self.tgt_exit, MAX_EXITS).float().unsqueeze(-1),
                         self.route_fire().float().unsqueeze(-1)], -1).flatten(1)                                    # 64
        # 피보호자 5, 구조 대상 6, 이웃 3
        w = self.ward.clamp_min(0); has_w = (self.ward >= 0) & ~self.done[w]
        rel = (self.pos[w] - self.pos) / 10.0
        wardv = torch.cat([has_w.float().unsqueeze(-1), rel * has_w.unsqueeze(-1), (rel.norm(dim=-1) * has_w).unsqueeze(-1),
                           ((self.fed_tox + self.fed_heat)[w] * has_w).unsqueeze(-1)], -1)
        rt = self.rescue_tgt.clamp_min(0); has_rt = self.rescue_tgt >= 0
        relr = (self.pos[rt] - self.pos) / 15.0
        resv = torch.cat([has_rt.float().unsqueeze(-1), relr * has_rt.unsqueeze(-1), (relr.norm(dim=-1) * has_rt).unsqueeze(-1),
                          ((self.fed_tox + self.fed_heat)[rt] * has_rt).unsqueeze(-1), (self.carrying >= 0).float().unsqueeze(-1)], -1)
        _, _, nnb = self._agent_force(self.pos, self.vel, self.active & ~self.done)
        p = self.pos.view(self.B, self.N, 2); v = self.vel.view(self.B, self.N, 2)
        near = ((p[:, :, None] - p[:, None]).norm(dim=-1) < AA_CUTOFF).float(); near.diagonal(dim1=1, dim2=2).fill_(0)
        vmean = (near.unsqueeze(-1) * v[:, None]).sum(2) / near.sum(2, keepdim=True).clamp_min(1)
        nbv = torch.cat([(nnb / 10.0).unsqueeze(-1), vmean.view(-1, 2) / 2.0], -1)
        modev = torch.cat([F.one_hot(self.mode, 7).float(), self.p_survive.unsqueeze(-1)], -1)
        visv = (VIS_C / self.ks_avg.clamp_min(1e-3)).clamp_max(30.0).unsqueeze(-1) / 10.0
        vec = torch.cat([base, rv, exs, wardv, resv, nbv, modev, visv], -1)
        return patch, vec

    # ── 기준선 ──────────────────────────────────────────────────────────────────
    def baseline_action(self):
        """rust_evac nav 등가: 목표 출구 거리장의 8-이웃 내리막 방향, m=1.0 — 매크로 규칙은 그대로 쓴다."""
        A, d = self.A, self.dev
        nav = self._nav_dir(self.tgt_exit)
        tx = self.pos[:, 0] + nav[:, 0]; ty = self.pos[:, 1] + nav[:, 1]
        # 소방관·구조 대상이 있으면 대상 쪽으로
        rt = self.rescue_tgt.clamp_min(0); has_rt = (self.rescue_tgt >= 0) & (self.carrying < 0)
        tx = torch.where(has_rt, self.pos[rt, 0], tx); ty = torch.where(has_rt, self.pos[rt, 1], ty)
        exdir = torch.stack([tx, ty], -1) - self.pos
        if not hasattr(self, "_wf_cs"): self._wf_cs = self._wayfind_noise()
        exdir = self._rot(exdir, *self._wf_cs)                                         # 기준선도 같은 시야 제약을 받는다
        ang = torch.atan2(exdir[:, 1], exdir[:, 0]) % (2 * math.pi)
        k = (ang / (2 * math.pi) * N_DIR).round().long() % N_DIR
        m = torch.full_like(k, 2)
        m = torch.where((self.mode == M_WAIT) | (self.mode == M_SHELTER), torch.zeros_like(m), m)
        m = torch.where(self.mode == M_BREAK, torch.full_like(m, 3), m)
        return k * 4 + m
