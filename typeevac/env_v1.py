# -*- coding: utf-8 -*-
"""Phase 1 — 피난 RL 환경 (torch 벡터화, 단일 에이전트).

동역학은 rust_evac(evac/rust_jupedsim/rust_evac/src) 의 BR 모델을 그대로 옮겼다:
  추진 (e0·v0_eff − v)/τ · 벽 반발 sfm_force · 복사력(점광원, facing 가중, 상한 3) ·
  위험장 밀침 −∇H·perception (상한 0.9 < 희망 1) · Frantzich–Nilsson 감속(K_s EMA τ=2s) ·
  Purser FED(tox+heat), FED ≥ 1 무력화.
정책이 주는 것은 rust_evac 의 nav 가 주던 **희망방향 e0 와 속도배율 m** 뿐이다.
행동 = 16방향 × {0, 0.6, 1.0, 1.4} = 64-way. m=1.4 가 밀침 상한 0.9 를 이겨 '뛰어넘기'가 된다.

보상(노브): 탈출 +100 · 시간 −0.1/s · FED −20·ΔFED · 무력화 −100 · 고립 −50 · 벽 접촉 −0.005·|f|
"""
from __future__ import annotations

import glob
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

# ── rust_evac params.rs (Model::Br) ───────────────────────────────────────────
# radius 만 0.3→0.2 (params.rs 의 CFSM/AVM 값). 이유: .fds 의 문 0.9 m 가 0.2 m 격자에서 4셀=0.8 m 로
# 잘리고, SFM 은 문 양쪽 기둥 모서리가 진행 반대 방향으로 미는 성분이 합산되므로 반경 0.3 이면
# 문 여유가 0.2 m 뿐이라 단독 보행자가 문 앞에서 멈춘다(실측: 벽 반발 1.9~2.3 vs 추진 2.0).
# 0.2 면 여유 0.4 m 로 rust_evac 의 연속 기하(문 0.9, r 0.3, 여유 0.3)와 같은 수준이 된다.
P = dict(v0=1.0, radius=0.2, mass=80.0, tau=0.5, a_obst=2000.0, b=0.08, k=120_000.0,
         kappa=240_000.0, clamp_v=3.2, chi_r=0.3, qth=2.5, qref=5.0, sr=4.0,
         repulsion=1.0, perception=3.0, temp_ref=40.0, ks_ref=1.0, exposure_tau=2.0,
         fed_threshold=1.0, speed_min=0.15, push_cap=0.9)
SUB, SUB_DT = 8, 0.0125            # model.rs: BR sub=8, dt=0.0125 → 0.1 s / step
STEP_DT = SUB * SUB_DT
N_DIR = 16
DIRS = torch.tensor([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, N_DIR, endpoint=False)],
                    dtype=torch.float32)
MULTS = torch.tensor([0.0, 0.6, 1.0, 1.4])
N_ACT = N_DIR * 4
FIRE_THR = 50.0                    # kW/m³ — 화염 셀
PATCH = 20                         # 시야 패치 20×20, 0.4 m 셀 → 반경 4 m
PATCH_DS = 2                       # 0.2 m 격자를 2× 다운샘플
R_HOLD = 90.0                      # 물리장 마지막 프레임 고정 시간(s)


def fed_rates(temp_c, co_ppm, co2_pct, o2_pct):
    """hazard.rs::fed_rates — Purser. 반환 [1/min]."""
    co = co_ppm.clamp_min(0.0)
    co2 = co2_pct.clamp_min(0.0)
    o2 = o2_pct.clamp(0.0, 20.9)
    f_co = 2.764e-5 * co.pow(1.036)
    hv = torch.exp(0.1930 * co2 + 2.0004) / 7.1
    f_o2 = 1.0 / torch.exp(8.13 - 0.54 * (20.9 - o2))
    heat = torch.where(temp_c > 0, 1.0 / (5.0e7 * temp_c.clamp_min(1.0).pow(-3.4)), torch.zeros_like(temp_c))
    return f_co * hv + f_o2, heat


def speed_factor(ks, min_frac=P["speed_min"]):
    """hazard.rs::speed_factor — Frantzich & Nilsson."""
    return (1.0 + (-0.057 / 0.706) * ks.clamp_min(0.0)).clamp(min_frac, 1.0)


class EvacEnv:
    def __init__(self, field_dir: str, n_envs: int, device="cuda", seed=0, max_time=None,
                 reward=None, case_filter=None, no_fire=False, situ_emb=None, time_scale=1.0):
        self.dev = torch.device(device)
        self.E = n_envs
        self.rng = random.Random(seed)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.files = sorted(glob.glob(os.path.join(field_dir, "*.npz")))
        if case_filter:
            self.files = [f for f in self.files if case_filter(os.path.basename(f))]
        assert self.files, "위험장 없음: " + field_dir
        self.R = dict(exit=100.0, time=-0.1, fed=-20.0, down=-100.0, trap=-50.0, contact=-0.005)
        if reward:
            self.R.update(reward)
        self.max_time = max_time
        self.no_fire = no_fire
        # 화재 성장 시간 배율: 에이전트 시간 t 에서 물리장은 t/time_scale 프레임을 본다.
        # 캐시는 램프 1 s 에 3.3 MW 최대라 30 s 안에 방이 300°C 를 넘고 기준선도 19% 만 탈출한다 —
        # 학습 가능한 커리큘럼(3 → 2 → 1)을 만들기 위한 노브. 물리적으로는 "성장이 k 배 느린 화원"에 근사.
        self.time_scale = float(time_scale)
        # 상황 텍스트 임베딩(Qwen3.8-Flash-Next 동결, 케이스당 1개). 없으면 emb_dim=0.
        self.situ = None
        if situ_emb and os.path.exists(situ_emb):
            z = np.load(situ_emb); self.situ = {k: z[k] for k in z.files}
            self.emb_dim = int(next(iter(self.situ.values())).shape[0])
        else:
            self.emb_dim = 0
        self._load_batch()

    # ── 케이스 적재(에피소드마다 E 개 샘플, 최대 격자로 패딩) ──────────────────
    def _load_batch(self):
        picks = [self.rng.choice(self.files) for _ in range(self.E)]
        zs = [np.load(p) for p in picks]
        NX = max(int(z["T"].shape[1]) for z in zs)
        NY = max(int(z["T"].shape[2]) for z in zs)
        NT = max(int(z["T"].shape[0]) for z in zs)
        E = self.E
        T = torch.full((E, NT, NX, NY), 20.0)
        KS = torch.zeros((E, NT, NX, NY)); CO = torch.zeros_like(KS); O2 = torch.full_like(KS, 20.9)
        FIRE = torch.zeros_like(KS)
        # 패딩 영역(케이스 격자 밖)은 벽이 아니라 **외부·출구**로 둔다 — 벽이면 작은 케이스의 도메인
        # 경계(=출구)에 접근할 때 패딩 벽이 밀어내 출구 직전에서 멈춘다(실측 DIST 0.3 정지).
        WALL = torch.zeros((E, NX, NY), dtype=torch.bool); EXIT = torch.ones_like(WALL); FREE = torch.zeros_like(WALL)
        DIST = torch.full((E, NX, NY), float("inf"))
        self.dx = float(zs[0]["dx"]); self.x0 = torch.zeros(E); self.y0 = torch.zeros(E)
        self.times = torch.zeros((E, NT)); self.case = []
        for e, z in enumerate(zs):
            nt, nx, ny = z["T"].shape
            T[e, :nt, :nx, :ny] = torch.from_numpy(z["T"].astype(np.float32))
            KS[e, :nt, :nx, :ny] = torch.from_numpy(z["KS"].astype(np.float32))
            CO[e, :nt, :nx, :ny] = torch.from_numpy(z["CO"].astype(np.float32))
            O2[e, :nt, :nx, :ny] = torch.from_numpy(z["O2"].astype(np.float32))
            FIRE[e, :nt, :nx, :ny] = torch.from_numpy(z["FIRE"].astype(np.float32))
            if nt < NT:
                for A in (T, KS, CO, O2, FIRE):
                    A[e, nt:] = A[e, nt - 1:nt]
            WALL[e, :nx, :ny] = torch.from_numpy(z["WALL"]); EXIT[e, :nx, :ny] = torch.from_numpy(z["EXIT"])
            FREE[e, :nx, :ny] = torch.from_numpy(z["FREE"]); DIST[e, :nx, :ny] = torch.from_numpy(z["DIST"])
            tt = torch.from_numpy(z["times"].astype(np.float32)); self.times[e, :nt] = tt
            if nt < NT:
                self.times[e, nt:] = tt[-1] + torch.arange(1, NT - nt + 1) * (tt[-1] - tt[-2])
            self.x0[e] = float(z["x0"]); self.y0[e] = float(z["y0"]); self.case.append(str(z["case"]))
        d = self.dev
        if self.situ is not None:
            self.EMB = torch.from_numpy(np.stack([self.situ.get(c, np.zeros(self.emb_dim, np.float32)) for c in self.case])).to(d)
            self.EMB = self.EMB / self.EMB.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # 코사인 정규화
        if self.no_fire:                                    # 커리큘럼 1단계 — 화재 없는 방에서 경로만
            T.fill_(20.0); KS.zero_(); CO.zero_(); O2.fill_(20.9); FIRE.zero_()
        self.T, self.KS, self.CO, self.O2, self.FIRE = (A.to(d) for A in (T, KS, CO, O2, FIRE))
        self.WALL, self.EXIT, self.FREE, self.DIST = WALL.to(d), EXIT.to(d), FREE.to(d), DIST.to(d)
        self.times = self.times.to(d); self.x0 = self.x0.to(d); self.y0 = self.y0.to(d)
        self.NX, self.NY, self.NT = NX, NY, NT
        self.t_end = (self.times[:, -1] * self.time_scale + R_HOLD) if self.max_time is None else torch.full((E,), float(self.max_time), device=d)
        # 벽 거리장(m)과 그 기울기 — 벽 반발용. 거리변환은 BFS 로 근사(4-이웃).
        self.WDIST = self._wall_dist(self.WALL)
        gx, gy = self._grad(self.WDIST)
        self.WN = torch.stack([gx, gy], -1)              # 벽에서 멀어지는 단위방향
        self.WN = self.WN / self.WN.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # 출구 방향 = −∇DIST
        dxg, dyg = self._grad(torch.where(torch.isfinite(self.DIST), self.DIST, torch.full_like(self.DIST, 1e3)))
        self.EXIT_DIR = -torch.stack([dxg, dyg], -1)
        self.EXIT_DIR = self.EXIT_DIR / self.EXIT_DIR.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # 화원 점광원(케이스당 1개): 전 시간 FIRE 최대 셀. Q[kW] = 컬럼 HRRPUV 합 근사 → FIRE 최대 × 셀부피 × 활성셀수
        fmax = self.FIRE.amax(dim=1)                       # [E,X,Y]
        idx = fmax.flatten(1).argmax(1)
        self.fire_xy = torch.stack([idx // NY, idx % NY], -1).float() * self.dx + torch.stack([self.x0, self.y0], -1)
        self.fire_Q = torch.tensor([float(z["fire_Q"]) if "fire_Q" in z.files else 3286.4 for z in zs], device=self.dev)
        if self.no_fire:
            self.fire_Q = torch.zeros_like(self.fire_Q)
        self.frame_dt = (self.times[:, 1] - self.times[:, 0]).clamp_min(1e-3)

    def _wall_dist(self, wall):
        E, NX, NY = wall.shape
        d = torch.where(wall, torch.zeros(()), torch.full((), float("inf"))).to(wall.device).expand(E, NX, NY).clone()
        # 반복 완화(Chamfer 근사): 충분히 큰 횟수
        for _ in range(max(NX, NY)):
            p = F.pad(d.unsqueeze(1), (1, 1, 1, 1), value=float("inf")).squeeze(1)
            n = torch.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]], 0).amin(0) + self.dx
            nd = torch.minimum(d, n)
            if torch.equal(nd, d):
                break
            d = nd
        return d

    @staticmethod
    def _grad(f):
        gx = torch.zeros_like(f); gy = torch.zeros_like(f)
        gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2; gy[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / 2
        gx = torch.nan_to_num(gx, 0.0, 0.0, 0.0); gy = torch.nan_to_num(gy, 0.0, 0.0, 0.0)
        return gx, gy

    # ── 격자 조회 ────────────────────────────────────────────────────────────────
    def _cell(self, pos):
        ix = ((pos[:, 0] - self.x0) / self.dx).round().long().clamp(0, self.NX - 1)
        iy = ((pos[:, 1] - self.y0) / self.dx).round().long().clamp(0, self.NY - 1)
        return ix, iy

    def _sample_field(self, A, pos, t):
        """A [E,T,X,Y] 를 위치·시각에서 표본(시간 선형보간, 공간 최근접)."""
        ix, iy = self._cell(pos)
        e = torch.arange(self.E, device=self.dev)
        fi = (t / self.time_scale / self.frame_dt).clamp(0, self.NT - 1.001)
        f0 = fi.floor().long(); w = (fi - f0).unsqueeze(-1) if A.dim() > 3 else fi - f0
        v0 = A[e, f0, ix, iy]; v1 = A[e, (f0 + 1).clamp(max=self.NT - 1), ix, iy]
        return v0 * (1 - (fi - f0)) + v1 * (fi - f0)

    def _lookup2d(self, A, pos):
        """[E,X,Y] 또는 [E,X,Y,2] 를 위치에서 bilinear 보간 (bool 은 최근접)."""
        if A.dtype == torch.bool:
            ix, iy = self._cell(pos)
            return A[torch.arange(self.E, device=self.dev), ix, iy]
        fx = ((pos[:, 0] - self.x0) / self.dx).clamp(0, self.NX - 1.001)
        fy = ((pos[:, 1] - self.y0) / self.dx).clamp(0, self.NY - 1.001)
        x0 = fx.floor().long(); y0 = fy.floor().long(); wx = (fx - x0); wy = (fy - y0)
        e = torch.arange(self.E, device=self.dev)
        if A.dim() == 4:
            wx = wx.unsqueeze(-1); wy = wy.unsqueeze(-1)
        v = (A[e, x0, y0] * (1 - wx) * (1 - wy) + A[e, x0 + 1, y0] * wx * (1 - wy)
             + A[e, x0, y0 + 1] * (1 - wx) * wy + A[e, x0 + 1, y0 + 1] * wx * wy)
        return v

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    def reset(self, reload=True, mask=None):
        """mask 가 있으면 그 env 만 재시작(케이스 유지). 없으면 전체."""
        if reload and mask is None:
            self._load_batch()
            if hasattr(self, "vel"):
                del self.vel
        E, d = self.E, self.dev
        if mask is None:
            mask = torch.ones(E, dtype=torch.bool, device=d)
        # 출발 위치: FREE ∧ DIST 유한 ∧ 방 내부(DIST ≥ 3 m) ∧ 벽거리 ≥ 0.4 ∧ 화원에서 ≥ 1.5 m
        ok = self.FREE & torch.isfinite(self.DIST) & (self.DIST >= 3.0) & (self.WDIST >= 0.4)
        pos = self.pos.clone() if hasattr(self, "vel") else torch.zeros((E, 2), device=d)
        for e in torch.nonzero(mask).flatten().tolist():
            xs, ys = torch.nonzero(ok[e], as_tuple=True)
            cand = torch.stack([xs, ys], -1).float() * self.dx + torch.stack([self.x0[e], self.y0[e]])
            far = (cand - self.fire_xy[e]).norm(dim=-1) >= 1.5
            cand = cand[far] if far.any() else cand
            pos[e] = cand[torch.randint(0, cand.shape[0], (1,), generator=self.g, device=d)]
        z2 = torch.zeros((E, 2), device=d); z1 = torch.zeros(E, device=d); o1 = torch.ones(E, device=d)
        t_new = torch.rand(E, generator=self.g, device=d) * 5.0      # 출발 지연 0~5 s (망설임)
        ori0 = torch.tensor([[1.0, 0.0]], device=d).expand(E, 2)
        m2 = mask.unsqueeze(-1)
        if not hasattr(self, "vel"):
            self.vel = z2.clone(); self.orient = ori0.clone(); self.fed_tox = z1.clone(); self.fed_heat = z1.clone()
            self.ks_avg = z1.clone(); self.haz_speed = o1.clone(); self.t = t_new.clone()
            self.done = torch.zeros(E, dtype=torch.bool, device=d); self.exited = self.done.clone(); self.down = self.done.clone()
            self.step_n = z1.clone()
        self.pos = pos
        self.vel = torch.where(m2, z2, self.vel); self.orient = torch.where(m2, ori0, self.orient)
        self.fed_tox = torch.where(mask, z1, self.fed_tox); self.fed_heat = torch.where(mask, z1, self.fed_heat)
        self.ks_avg = torch.where(mask, z1, self.ks_avg); self.haz_speed = torch.where(mask, o1, self.haz_speed)
        self.t = torch.where(mask, t_new, self.t)
        self.done = torch.where(mask, torch.zeros_like(self.done), self.done)
        self.exited = torch.where(mask, torch.zeros_like(self.exited), self.exited)
        self.down = torch.where(mask, torch.zeros_like(self.down), self.down)
        self.step_n = torch.where(mask, z1, self.step_n)
        return self._obs()

    def _wall_force(self, pos, vel, R=4):
        """rust_evac NearWall 등가: 4방향(±x,±y) 각각에서 **가장 가까운 벽 셀 하나**에 sfm_force.
        벽 한 면의 셀을 전부 합산하면 셀 수만큼 과대(실측 60~80 m/s²)해지므로 면당 하나만 센다.
        반환 (힘[E,2], 접촉 push 합[E], 최근접 벽 법선[E,2])."""
        E, d = self.E, self.dev
        ix, iy = self._cell(pos)
        ar = torch.arange(-R, R + 1, device=d)
        gx = (ix[:, None, None] + ar[None, :, None]).clamp(0, self.NX - 1).expand(-1, -1, 2 * R + 1)
        gy = (iy[:, None, None] + ar[None, None, :]).clamp(0, self.NY - 1).expand(-1, 2 * R + 1, -1)
        e = torch.arange(E, device=d)[:, None, None]
        iswall = self.WALL[e, gx, gy]                                   # [E,W,W]
        cx = gx.float() * self.dx + self.x0[:, None, None]; cy = gy.float() * self.dx + self.y0[:, None, None]
        dvx_c = pos[:, 0, None, None] - cx; dvy_c = pos[:, 1, None, None] - cy
        # 벽 셀은 한 변 dx 의 정사각형 — 셀 중심이 아니라 **셀 면까지**의 최단거리로 잰다.
        # (중심 거리 + 반셀 보정은 문 폭을 0.8→0 으로 잠식해 통과가 막혔다. 실측: 문 앞 정지)
        half = self.dx * 0.5
        dvx = torch.sign(dvx_c) * (dvx_c.abs() - half).clamp_min(0.0)
        dvy = torch.sign(dvy_c) * (dvy_c.abs() - half).clamp_min(0.0)
        dist = torch.sqrt(dvx ** 2 + dvy ** 2)
        inside = dist < 1e-6                                               # 셀 안(관통) — 중심 방향으로 밀어낸다
        dvx = torch.where(inside, dvx_c, dvx); dvy = torch.where(inside, dvy_c, dvy)
        dist = torch.where(inside, torch.full_like(dist, 1e-3), dist)
        # 면 법선 힘만: 모서리(대각) 셀은 제외한다. 문 기둥 모서리가 진행 반대 방향으로 미는 성분이
        # 합산되어 단독 보행자의 문 통과를 막는 것이 SFM 의 알려진 결함이고(실측: 반발 1.9~2.3 vs
        # 추진 2.0), 실무 보행 시뮬레이터는 벽 세그먼트 법선 방향 힘만 쓴다. 관통은 슬라이딩이 막는다.
        corner = (dvx.abs() > 1e-6) & (dvy.abs() > 1e-6) & ~inside
        # 4 섹터(셀 중심 기준): 벽이 에이전트의 +x 쪽이면 dvx_c < 0
        sect = [(dvx_c < 0) & (dvx_c.abs() >= dvy_c.abs()), (dvx_c > 0) & (dvx_c.abs() >= dvy_c.abs()),
                (dvy_c < 0) & (dvy_c.abs() > dvx_c.abs()), (dvy_c > 0) & (dvy_c.abs() > dvx_c.abs())]
        rad = P["radius"]
        fx = torch.zeros(E, device=d); fy = torch.zeros(E, device=d); push_sum = torch.zeros(E, device=d)
        best_d = torch.full((E,), 1e9, device=d); wn = torch.zeros((E, 2), device=d)
        ar_e = torch.arange(E, device=d)
        for sm in sect:
            dm = torch.where(iswall & sm & ~corner, dist, torch.full_like(dist, 1e9)).flatten(1)
            j = dm.argmin(1); dj = dm[ar_e, j]
            has = dj < 1e8
            nx_ = (dvx.flatten(1)[ar_e, j] / dj.clamp_min(1e-6)); ny_ = (dvy.flatten(1)[ar_e, j] / dj.clamp_min(1e-6))
            push = P["a_obst"] * torch.exp((rad - dj) / P["b"])
            close = dj < rad
            push = push + torch.where(close, P["k"] * (rad - dj), torch.zeros_like(dj))
            tx, ty = -ny_, nx_
            vt = vel[:, 0] * tx + vel[:, 1] * ty
            fric = torch.where(close, P["kappa"] * (rad - dj) * vt, torch.zeros_like(dj))
            h = has.float()
            fx = fx + (nx_ * push + tx * fric) * h; fy = fy + (ny_ * push + ty * fric) * h
            push_sum = push_sum + torch.where(close, push, torch.zeros_like(push)) * h
            better = has & (dj < best_d)
            best_d = torch.where(better, dj, best_d)
            wn = torch.where(better.unsqueeze(-1), torch.stack([nx_, ny_], -1), wn)
        return torch.stack([fx, fy], -1), push_sum, wn

    def _hazard(self, dt):
        """sim.rs::hazard_pass — 표본 → FED → K_s EMA → 속도계수 → 밀침."""
        T = self._sample_field(self.T, self.pos, self.t)
        KS = self._sample_field(self.KS, self.pos, self.t)
        CO = self._sample_field(self.CO, self.pos, self.t)
        O2 = self._sample_field(self.O2, self.pos, self.t)
        tox, heat = fed_rates(T, CO, torch.zeros_like(T), O2)
        live = (~self.done).float()
        self.fed_tox = self.fed_tox + tox * dt / 60.0 * live
        self.fed_heat = self.fed_heat + heat * dt / 60.0 * live
        blend = min(dt / P["exposure_tau"], 1.0)
        self.ks_avg = self.ks_avg + (KS - self.ks_avg) * blend
        self.haz_speed = speed_factor(self.ks_avg)
        # 위험장 밀침: H = max(0,T−40)/40 + KS/1, 인지거리 d 앞뒤 차분
        dpx = P["perception"]
        def H(p):
            Tq = self._sample_field(self.T, p, self.t); Kq = self._sample_field(self.KS, p, self.t)
            return (Tq - P["temp_ref"]).clamp_min(0) / P["temp_ref"] + Kq.clamp_min(0) / P["ks_ref"]
        ex = torch.tensor([dpx, 0.0], device=self.dev); ey = torch.tensor([0.0, dpx], device=self.dev)
        g = torch.stack([(H(self.pos + ex) - H(self.pos - ex)) / (2 * dpx),
                         (H(self.pos + ey) - H(self.pos - ey)) / (2 * dpx)], -1)
        push = g * (-P["repulsion"] * dpx)
        m = push.norm(dim=-1, keepdim=True)
        push = torch.where(m > P["push_cap"], push * (P["push_cap"] / m.clamp_min(1e-9)), push)
        return push, T, KS

    def step(self, action):
        """action [E] long in [0, 32). 반환 obs, reward, done, info."""
        E, d = self.E, self.dev
        e0 = DIRS.to(d)[action // 4]                    # 희망방향 (16방향)
        mult = MULTS.to(d)[action % 4]                  # 속도배율
        rew = torch.zeros(E, device=d)
        contact = torch.zeros(E, device=d)
        inv_m = 1.0 / P["mass"]
        fed0 = self.fed_tox + self.fed_heat
        for _ in range(SUB):
            active = ~self.done
            push, T, KS = self._hazard(SUB_DT)
            v0 = P["v0"] * mult * self.haz_speed
            f = (e0 * v0.unsqueeze(-1) - self.vel) / P["tau"]
            # 벽 반발 — rust_evac NearWall 방식: 주변 벽 셀 각각을 벽점으로 sfm_force 합산
            #   (문 양쪽 벽이 상쇄되어 통과가 가능해진다. 단일 법선으로는 안 된다.)
            fw, push_sum, wn = self._wall_force(self.pos, self.vel)
            contact = contact + push_sum * SUB_DT
            f = f + fw * inv_m
            # 복사력 — 점광원
            dv = self.pos - self.fire_xy; r = dv.norm(dim=-1).clamp_min(0.3); n = dv / r.unsqueeze(-1)
            q = (P["chi_r"] * self.fire_Q) / (4 * math.pi * r * r)
            over = (q - P["qth"]).clamp_min(0)
            mag = P["sr"] * (over / P["qref"]).clamp_max(3.0)
            facing = (self.orient * (-n)).sum(-1).clamp_min(0)
            mag = mag * (0.4 + 0.6 * facing)
            # rust_evac 원식은 상한이 12 m/s² 로 추진(2)을 압도해 3.3 MW 화원 3 m 안은 접근 불가가 된다.
            # 여기서는 위험장 밀침과 같은 원칙 — 밀침 < 희망 — 을 적용해 push_cap·v0/τ 로 캡한다.
            # 통과의 실제 대가는 대류열 FED 가 매기고, 정책은 m=1.4 로 이 캡을 이겨 '뛰어넘기'를 택할 수 있다.
            mag = torch.minimum(mag, P["push_cap"] * v0 / P["tau"] * torch.ones_like(mag))
            f = f + n * mag.unsqueeze(-1)
            # 위험장 밀침
            f = f + push * (v0 / P["tau"]).unsqueeze(-1)
            nv = self.vel + f * SUB_DT
            sp = nv.norm(dim=-1, keepdim=True)
            nv = torch.where(sp > P["clamp_v"], nv * (P["clamp_v"] / sp.clamp_min(1e-9)), nv)
            nv = torch.where(active.unsqueeze(-1), nv, torch.zeros_like(nv))
            npos = self.pos + nv * SUB_DT
            # 벽 진입 금지: 새 위치가 벽이면 법선 성분을 제거하고 접선으로 미끄러진다
            ix, iy = self._cell(npos)
            blocked = self.WALL[torch.arange(E, device=d), ix, iy]
            if blocked.any():
                vn = (nv * wn).sum(-1, keepdim=True)
                slide = nv - wn * vn.clamp_max(0.0)          # 벽 쪽(음의 법선) 성분만 제거
                npos2 = self.pos + slide * SUB_DT
                ix2, iy2 = self._cell(npos2)
                still = self.WALL[torch.arange(E, device=d), ix2, iy2]
                npos = torch.where(blocked.unsqueeze(-1), torch.where(still.unsqueeze(-1), self.pos, npos2), npos)
                nv = torch.where(blocked.unsqueeze(-1), torch.where(still.unsqueeze(-1), torch.zeros_like(nv), slide), nv)
            self.pos = torch.where(active.unsqueeze(-1), npos, self.pos)
            self.vel = nv
            ori = nv / nv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.orient = torch.where((nv.norm(dim=-1) > 1e-3).unsqueeze(-1), ori, self.orient)
            self.t = self.t + SUB_DT * active.float()
        # ── 종료·보상 ──
        fed = self.fed_tox + self.fed_heat
        at_exit = self._lookup2d(self.EXIT, self.pos)
        newly_exit = at_exit & ~self.done
        newly_down = (fed >= P["fed_threshold"]) & ~self.done & ~newly_exit
        newly_trap = (self.t >= self.t_end) & ~self.done & ~newly_exit & ~newly_down
        rew = rew + self.R["time"] * STEP_DT * (~self.done).float()
        rew = rew + self.R["fed"] * (fed - fed0) * (~self.done).float()
        rew = rew + self.R["contact"] * contact
        rew = rew + self.R["exit"] * newly_exit.float() + self.R["down"] * newly_down.float() + self.R["trap"] * newly_trap.float()
        self.exited |= newly_exit; self.down |= newly_down
        self.done |= newly_exit | newly_down | newly_trap
        self.step_n += 1
        info = dict(exited=self.exited.clone(), down=self.down.clone(), t=self.t.clone(), fed=fed.clone())
        return self._obs(), rew, self.done.clone(), info

    # ── 관측 ────────────────────────────────────────────────────────────────────
    def _obs(self):
        E, d = self.E, self.dev
        # 시야 패치: 에이전트 중심 PATCH×PATCH(0.4 m), 채널 [T/1000, KS/5, FIRE>thr, WALL], 가시거리 밖 0
        R = PATCH * PATCH_DS // 2
        ix, iy = self._cell(self.pos)
        ar = torch.arange(-R, R, PATCH_DS, device=d)
        gx = (ix.unsqueeze(-1) + ar).clamp(0, self.NX - 1)               # [E,P]
        gy = (iy.unsqueeze(-1) + ar).clamp(0, self.NY - 1)
        e = torch.arange(E, device=d)
        fi = (self.t / self.time_scale / self.frame_dt).round().long().clamp(0, self.NT - 1)
        def take(A):
            return A[e[:, None, None], fi[:, None, None], gx[:, :, None], gy[:, None, :]]
        def take2(A):
            return A[e[:, None, None], gx[:, :, None], gy[:, None, :]]
        vis = 3.0 / self.ks_avg.clamp_min(3e-3)
        rr = torch.sqrt((ar.float() ** 2)[:, None] + (ar.float() ** 2)[None, :]) * self.dx
        vmask = (rr[None] <= vis[:, None, None]).float()
        patch = torch.stack([take(self.T) / 1000.0 * vmask, take(self.KS) / 5.0 * vmask,
                             (take(self.FIRE) > FIRE_THR).float() * vmask, take2(self.WALL).float()], 1)
        exdir = self._lookup2d(self.EXIT_DIR, self.pos)
        dist = self._lookup2d(self.DIST, self.pos); dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, 50.0))
        wd = self._lookup2d(self.WDIST, self.pos)
        vec = torch.cat([self.vel / 2.0, exdir, (dist / 30.0).unsqueeze(-1), (wd / 3.0).clamp_max(1).unsqueeze(-1),
                         self.fed_tox.unsqueeze(-1), self.fed_heat.unsqueeze(-1), (self.ks_avg / 5).unsqueeze(-1),
                         (self.t / 60.0).unsqueeze(-1), self.haz_speed.unsqueeze(-1), self.orient], -1)
        if self.situ is not None:
            return patch, vec, self.EMB
        return patch, vec

    # ── 기준선: rust_evac nav — 최근접 출구 거리장 방향, m=1.0 ───────────────────
    def baseline_action(self):
        """rust_evac nav 등가: 주변 셀 중 DIST 가 가장 작은 곳(반경 3셀)의 방향, m=1.0."""
        E, d = self.E, self.dev
        ix, iy = self._cell(self.pos)
        ar = torch.arange(-3, 4, device=d)
        gx = (ix[:, None, None] + ar[None, :, None]).clamp(0, self.NX - 1)
        gy = (iy[:, None, None] + ar[None, None, :]).clamp(0, self.NY - 1)
        e = torch.arange(E, device=d)[:, None, None]
        D = self.DIST[e, gx, gy]
        D = torch.where(self.FREE[e, gx, gy], D, torch.full_like(D, float("inf")))
        rr = torch.sqrt((ar.float() ** 2)[None, :, None] + (ar.float() ** 2)[None, None, :])
        D = D + 0.05 * rr * self.dx                   # 동점이면 가까운 셀(문 중앙 쪽)을 고른다
        j = D.flatten(1).argmin(1)
        gxf = gx.expand(-1, -1, gy.shape[2]).flatten(1); gyf = gy.expand(-1, gx.shape[1], -1).flatten(1)
        tx = gxf[torch.arange(E, device=d), j].float() * self.dx + self.x0
        ty = gyf[torch.arange(E, device=d), j].float() * self.dx + self.y0
        exdir = torch.stack([tx, ty], -1) - self.pos
        ang = torch.atan2(exdir[:, 1], exdir[:, 0]) % (2 * math.pi)
        k = (ang / (2 * math.pi) * N_DIR).round().long() % N_DIR
        return k * 4 + 2          # mult index 2 = 1.0


OBS_VEC = 12
OBS_PATCH_C = 4
