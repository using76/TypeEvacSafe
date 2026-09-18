# -*- coding: utf-8 -*-
"""피난 RL v2 — 대공간 파라메트릭 생성기 (Tier D, EVAC_RL_PLAN_v2 §1.2).

9 용도(마트·사무실·학교·병원·지하상가·식당·공연장·주차장·창고) × 크기 3단계 × 출구 배치 → 64 형상.
형상마다 시나리오: 화원 위치 3(near_exit / center / far) × 성장 2(medium / fast) + 다출구는 blocked_1.

설계 규칙
  · 격자: 면적 ≤400 m² dx 0.2 / ≤2,000 dx 0.25 / 초과 dx 0.4. 정육면체 셀(dz = dx), 높이 nz·dx ≈ 3 m.
  · 모든 좌표는 dx 배수로 스냅(슬리버 장애물 → 수치불안정, gen_big_fds_v2 실측).
  · 벽 두께 1셀, 문 HOLE z 0~2.1. 외부 출구는 건물 밖 여유폭(1 m)으로 나가고 도메인 4면 OPEN.
  · 화원: t² 성장 후 평탄 유지(α medium 0.01172 / fast 0.04689 kW/s²), 상한은 용도별. 버너 한 변 = √(Q/2500) 스냅.
  · 출력: 2D 슬라이스 z 0.5/1.5/2.4 m × 9채널(U V W P T ρ O2 SOOT HRRPUV), DT 1 s. 3D 캐시 없음.
  · 케이스마다 <name>.fds + <name>_meta.json(출구 사각형·폭·blocked, 화원, 용도, 밀도) — convert_sf2d 가 읽는다.
사용: python gen_bigspace_v2.py <out_dir> [--only D1,D5] [--dry]
"""
import argparse
import json
import math
import os

import numpy as np

ALPHA = {"medium": 0.01172, "fast": 0.04689}
Q_MAX = {"mart": 10000, "office": 5000, "school": 5000, "hospital": 5000, "umall": 10000,
         "restaurant": 3000, "theater": 10000, "parking": 8000, "warehouse": 20000, "hall": 10000}
DENSITY = {"mart": 0.3, "office": 0.1, "school": 0.5, "hospital": 0.15, "umall": 0.4,
           "restaurant": 0.7, "theater": 1.2, "parking": 0.02, "warehouse": 0.03, "hall": 0.5}     # 인/m² (환경이 N 결정)
DOOR_Z = 2.1
MARGIN = 1.0


def grid_dx(lx, ly):
    a = lx * ly
    return 0.2 if a <= 400 else (0.25 if a <= 2000 else 0.4)


class Building:
    """벽·문·가구·출구를 모아 .fds 텍스트와 메타를 만든다. 좌표는 건물 내부 원점(0,0)~(lx,ly)."""

    def __init__(self, kind, lx, ly, tag):
        self.kind, self.lx, self.ly, self.tag = kind, float(lx), float(ly), tag
        self.dx = grid_dx(lx, ly)
        self.walls, self.holes, self.furn, self.exits = [], [], [], []

    def s(self, v):
        return round(round(v / self.dx) * self.dx, 4)

    # ---- 요소 ----
    def wall(self, x0, x1, y0, y1, h=None):
        h = h or self.lz()
        self.walls.append((self.s(x0), self.s(x1), self.s(y0), self.s(y1), h))

    def hwall(self, y, x0, x1):          # y 위치, x0~x1 로 뻗는 벽(두께 1셀)
        y = self.s(y); self.wall(x0, x1, y, y + self.dx)

    def vwall(self, x, y0, y1):
        x = self.s(x); self.wall(x, x + self.dx, y0, y1)

    def door(self, x0, x1, y0, y1):     # 내부 문(HOLE)
        self.holes.append((self.s(x0), self.s(x1), self.s(y0), self.s(y1)))

    def door_h(self, y, xc, w=0.9):     # 가로벽(y)에 난 문, 중심 xc
        self.door(xc - w / 2, xc + w / 2, y - self.dx, y + 2 * self.dx)

    def door_v(self, x, yc, w=0.9):
        self.door(x - self.dx, x + 2 * self.dx, yc - w / 2, yc + w / 2)

    def block(self, x0, x1, y0, y1, h, name="F"):
        self.furn.append((self.s(x0), self.s(x1), self.s(y0), self.s(y1), h, name))

    def exit(self, side, pos, w, eid=None):
        """외벽 side(n/s/e/w)의 pos(0~1) 위치에 폭 w 출구. HOLE 은 외벽을 관통해 밖 여유폭까지."""
        eid = eid or "X%d" % (len(self.exits) + 1)
        w = self.s(w); t = self.dx
        if side in ("s", "n"):
            xc = self.s(self.lx * pos); x0, x1 = xc - w / 2, xc + w / 2
            y0, y1 = (-t, 2 * t) if side == "s" else (self.ly - 2 * t, self.ly + t)
            nrm = (0, 1) if side == "s" else (0, -1)
        else:
            yc = self.s(self.ly * pos); y0, y1 = yc - w / 2, yc + w / 2
            x0, x1 = (-t, 2 * t) if side == "w" else (self.lx - 2 * t, self.lx + t)
            nrm = (1, 0) if side == "w" else (-1, 0)
        self.exits.append(dict(id=eid, side=side, xb=[self.s(x0), self.s(x1), self.s(y0), self.s(y1)],
                               width=w, normal=nrm, blocked=False))

    def lz(self):
        return round(int(round(3.0 / self.dx)) * self.dx, 4)

    # ---- 래스터(화원 배치용) ----
    def occupancy(self, z=0.6):
        dx = self.dx; nx, ny = int(round(self.lx / dx)), int(round(self.ly / dx))
        occ = np.zeros((nx, ny), bool)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True          # 외벽 1셀
        def fill(x0, x1, y0, y1, v=True):
            i0, i1 = max(0, int(round(x0 / dx))), min(nx, int(round(x1 / dx)))
            j0, j1 = max(0, int(round(y0 / dx))), min(ny, int(round(y1 / dx)))
            occ[i0:i1, j0:j1] = v
        for x0, x1, y0, y1, h in self.walls:
            fill(x0, x1, y0, y1)
        for x0, x1, y0, y1, h, _ in self.furn:
            if h > z: fill(x0, x1, y0, y1)
        for x0, x1, y0, y1 in self.holes:
            fill(x0, x1, y0, y1, False)
        return occ

    def place_fire(self, where, side_len):
        """버너 위치 — near_exit: 주 출구에서 4 m 안쪽 / center / far: 주 출구에서 가장 먼 자유 셀. 장애물 여유 0.2 m."""
        occ = self.occupancy(); dx = self.dx; nx, ny = occ.shape
        half = side_len / 2 + 0.2
        r = int(math.ceil(half / dx))
        # 창(2r+1)² 안이 전부 자유인 셀 — 누적합으로 O(1) 창 검사
        c = np.zeros((nx + 1, ny + 1), np.int64); c[1:, 1:] = np.cumsum(np.cumsum(occ, 0), 1)
        ok = np.zeros((nx, ny), bool)
        for i in range(r, nx - r):
            for j in range(r, ny - r):
                i0, i1, j0, j1 = i - r, i + r + 1, j - r, j + r + 1
                ok[i, j] = (c[i1, j1] - c[i0, j1] - c[i1, j0] + c[i0, j0]) == 0
        if not ok.any():
            raise RuntimeError("%s: 버너 자리 없음(side %.1f)" % (self.tag, side_len))
        ex = self.exits[0]; xb = ex["xb"]; n = ex["normal"]
        exc = np.array([(xb[0] + xb[1]) / 2, (xb[2] + xb[3]) / 2])
        II, JJ = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        P = np.stack([(II + 0.5) * dx, (JJ + 0.5) * dx], -1)
        if where == "near_exit":
            tgt = exc + np.array(n) * 4.0
            d = np.linalg.norm(P - tgt, axis=-1); d[~ok] = np.inf
        elif where == "center":
            d = np.linalg.norm(P - np.array([self.lx / 2, self.ly / 2]), axis=-1); d[~ok] = np.inf
        else:
            d = -np.linalg.norm(P - exc, axis=-1); d[~ok] = np.inf
        k = int(np.argmin(d)); i, j = divmod(k, ny)
        return self.s((i + 0.5) * dx), self.s((j + 0.5) * dx)

    # ---- .fds ----
    def fds(self, name, fire_where, growth, blocked=None, t_end=None, fire_xy=None, q_max=None, t_peak=None):
        dx = self.dx; lz = self.lz(); nz = int(round(lz / dx))
        mx0, mx1 = -MARGIN, self.lx + MARGIN
        my0, my1 = -MARGIN, self.ly + MARGIN
        nx, ny = int(round((mx1 - mx0) / dx)), int(round((my1 - my0) / dx))
        t_end = t_end or (150.0 if self.lx * self.ly <= 400 else 240.0)
        qmax = q_max or max(1000.0, min(Q_MAX[self.kind], 25.0 * self.lx * self.ly)); a = ALPHA[growth]   # 소형 공간은 면적당 25 kW/m² 상한
        tp = t_peak or math.sqrt(qmax / a)              # 상한 도달 시각
        side = max(1.2, self.s(math.sqrt(qmax / 2500.0)))          # HRRPUA ≤ 2,500 kW/m²
        hrrpua = qmax / (side * side)
        fx, fy = (self.s(fire_xy[0]), self.s(fire_xy[1])) if fire_xy else self.place_fire(fire_where, side)
        L = ["&HEAD CHID='%s' /" % name, "&TIME T_END=%.0f /" % t_end, "&DUMP DT_SLCF=1.0, DT_HRR=1.0 /",
             "&MESH IJK=%d,%d,%d, XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f /" % (nx, ny, nz, mx0, mx1, my0, my1, lz),
             "&REAC ID='SFPE_WOOD_OAK', FUEL='SFPE WOOD_OAK_fuel', CO_YIELD=0.004, SOOT_YIELD=0.015, RADIATIVE_FRACTION=0.371 /",
             "&SPEC ID='SFPE WOOD_OAK_fuel', FORMULA='C1.0H1.7O0.72N0.001' /",
             "&SURF ID='BURNER', HRRPUA=%.1f, RAMP_Q='RQ', TMP_FRONT=300.0 /" % hrrpua]
        ts = [i * min(tp, t_end) / 8.0 for i in range(9)]
        if tp < t_end:
            ts += [tp + (t_end - tp) * i / 3.0 for i in range(1, 4)]
        for t in ts:
            L.append("&RAMP ID='RQ', T=%.1f, F=%.4f /" % (t, min(1.0, (t / tp) ** 2)))
        t = dx
        for wid, xb in (("W_S", (0, self.lx, 0, t)), ("W_N", (0, self.lx, self.ly - t, self.ly)),
                        ("W_W", (0, t, 0, self.ly)), ("W_E", (self.lx - t, self.lx, 0, self.ly))):
            L.append("&OBST ID='%s', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f, SURF_ID='INERT' /" % ((wid,) + xb + (lz,)))
        for i, (x0, x1, y0, y1, h) in enumerate(self.walls, 1):
            L.append("&OBST ID='IW%d', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f, SURF_ID='INERT' /" % (i, x0, x1, y0, y1, h))
        for i, (x0, x1, y0, y1, h, nm) in enumerate(self.furn, 1):
            L.append("&OBST ID='%s%d', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f, SURF_ID='INERT' /" % (nm, i, x0, x1, y0, y1, h))
        for i, (x0, x1, y0, y1) in enumerate(self.holes, 1):
            L.append("&HOLE ID='D%d', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.1f /" % (i, x0, x1, y0, y1, DOOR_Z))
        exits = []
        for e in self.exits:
            e = dict(e); e["blocked"] = (e["id"] == blocked)
            if not e["blocked"]:
                L.append("&HOLE ID='%s', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.1f /" % ((e["id"],) + tuple(e["xb"]) + (DOOR_Z,)))
            exits.append(e)
        h = side / 2
        L.append("&OBST ID='FIRE', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f, SURF_IDS='BURNER','INERT','INERT' /"
                 % (fx - h, fx + h, fy - h, fy + h, dx))
        for vid, xb in (("XMIN", (mx0, mx0, my0, my1)), ("XMAX", (mx1, mx1, my0, my1)),
                        ("YMIN", (mx0, mx1, my0, my0)), ("YMAX", (mx0, mx1, my1, my1))):
            L.append("&VENT ID='O_%s', SURF_ID='OPEN', XB=%.2f,%.2f,%.2f,%.2f,0.0,%.2f /" % ((vid,) + xb + (lz,)))
        Q = [("U-VELOCITY", None), ("V-VELOCITY", None), ("W-VELOCITY", None), ("PRESSURE", None),
             ("TEMPERATURE", None), ("DENSITY", None), ("VOLUME FRACTION", "OXYGEN"), ("DENSITY", "SOOT"), ("HRRPUV", None)]
        for z in (0.5, 1.5, 2.4, 2.8):
            for q, sp in Q:
                L.append("&SLCF QUANTITY='%s', %sPBZ=%.2f, CELL_CENTERED=.TRUE. /" % (q, "SPEC_ID='%s', " % sp if sp else "", z))
        L.append("&TAIL /")
        meta = dict(name=name, kind=self.kind, tag=self.tag, lx=self.lx, ly=self.ly, dx=dx, lz=lz,
                    mesh=[mx0, mx1, my0, my1], cells=nx * ny * nz, t_end=t_end, density=DENSITY[self.kind],
                    fire=dict(where=fire_where, x=fx, y=fy, side=side, growth=growth, alpha=a, q_max=qmax, t_peak=tp),
                    exits=exits, n_exits=len(exits), blocked=blocked)
        return "\n".join(L) + "\n", meta


# ---------------------------------------------------------------- 용도별 배치
def mart(lx, ly, n_exit, tag):
    b = Building("mart", lx, ly, tag)
    aisle = 1.6 if lx * ly < 300 else 2.2
    shelf_w, shelf_len = 1.0, lx * 0.62
    y = 3.0
    while y + shelf_w < ly - 2.5:
        b.block(lx * 0.19, lx * 0.19 + shelf_len, y, y + shelf_w, 2.0, "SHELF")
        y += shelf_w + aisle
    for k in range(max(1, int(lx / 6))):                          # 계산대(출구 쪽)
        b.block(2.0 + k * 3.0, 3.6 + k * 3.0, 1.0, 1.6, 0.9, "CHK")
    b.exit("s", 0.3, 1.8, "MAIN")
    if n_exit >= 2: b.exit("n", 0.8, 1.2, "BACK")
    if n_exit >= 3: b.exit("e", 0.5, 1.2, "SIDE")
    return b


def office(lx, ly, n_exit, tag):
    b = Building("office", lx, ly, tag)
    # 회의실 2(전 높이 벽, 문) — 북서/북동 모서리
    mw, md = min(6.0, lx * 0.25), min(5.0, ly * 0.3)
    for x0 in (0.0, lx - mw):
        b.hwall(ly - md, x0, x0 + mw)
        b.vwall(x0 + mw if x0 == 0 else x0, ly - md, ly)
        b.door_h(ly - md, x0 + mw / 2)
    # 파티션 그리드(1.5 m) — 중앙 통로 2 m 비움
    cx = lx / 2
    y = 2.0
    while y + 2.4 < ly - md - 1.5:
        for x0 in np.arange(1.5, lx - 2.4, 2.4 + 0.9):
            if abs(x0 + 1.2 - cx) < 1.8: continue
            b.block(x0, x0 + 2.4, y, y + b.dx, 1.5, "PT"); b.block(x0, x0 + b.dx, y, y + 2.4, 1.5, "PT")
        y += 2.4 + 1.2
    b.exit("s", 0.5, 1.8, "MAIN")
    if n_exit >= 2: b.exit("w", 0.15, 1.0, "STAIR_W")
    if n_exit >= 3: b.exit("e", 0.15, 1.0, "STAIR_E")
    return b


def school(lx, ly, n_exit, tag):
    b = Building("school", lx, ly, tag)
    cw = 2.4; y0 = (ly - cw) / 2; y1 = y0 + cw
    room_w = 9.0; n = int((lx - 1.0) // room_w)
    b.hwall(y0, 0, lx); b.hwall(y1, 0, lx)
    for k in range(n):
        x = 0.5 + k * room_w
        if k > 0:
            b.vwall(x, 0, y0); b.vwall(x, y1, ly)
        b.door_h(y0, x + room_w * 0.3); b.door_h(y1, x + room_w * 0.3)
        for r in range(3):                                       # 책상 열
            for c in range(4):
                b.block(x + 1.2 + c * 1.8, x + 2.4 + c * 1.8, 0.8 + r * 1.6, 1.4 + r * 1.6, 0.75, "DESK")
                b.block(x + 1.2 + c * 1.8, x + 2.4 + c * 1.8, y1 + 0.8 + r * 1.6, y1 + 1.4 + r * 1.6, 0.75, "DESK")
    b.exit("w", 0.5, 1.8, "MAIN")
    if n_exit >= 2: b.exit("e", 0.5, 1.8, "END_E")
    if n_exit >= 3:                                              # 중간 교실을 통한 외부 출구(교실 문 0.9 가 병목)
        kc = n // 2; b.exit("s", (0.5 + kc * room_w + room_w * 0.5) / lx, 1.2, "MID_S")
    return b


def hospital(lx, ly, n_exit, tag):
    b = Building("hospital", lx, ly, tag)
    cw = 2.4; y0 = (ly - cw) / 2; y1 = y0 + cw
    rw = 6.0; n = int((lx - 1.0) // rw)
    b.hwall(y0, 0, lx); b.hwall(y1, 0, lx)
    for k in range(n):
        x = 0.5 + k * rw
        if k > 0:
            b.vwall(x, 0, y0); b.vwall(x, y1, ly)
        b.door_h(y0, x + rw * 0.5, 1.1); b.door_h(y1, x + rw * 0.5, 1.1)
        for c in range(2):
            b.block(x + 0.8 + c * 2.8, x + 1.8 + c * 2.8, 0.5, 2.6, 0.7, "BED")
            b.block(x + 0.8 + c * 2.8, x + 1.8 + c * 2.8, ly - 2.6, ly - 0.5, 0.7, "BED")
    # 방화문(복도 중앙 가로막 + 1.8 문)
    xm = lx / 2
    b.vwall(xm, y0, y1); b.door_v(xm, (y0 + y1) / 2, 1.8)
    b.block(xm - 5.0, xm - 3.0, y0 + 0.2, y0 + 0.9, 1.1, "NURSE")   # 간호스테이션
    b.exit("w", 0.5, 1.8, "MAIN")
    if n_exit >= 2: b.exit("e", 0.5, 1.8, "END_E")
    if n_exit >= 3: b.exit("n", 0.5, 1.2, "MID_N")
    return b


def umall(lx, ly, n_exit, tag, two_corridor=False):
    b = Building("umall", lx, ly, tag)
    sw = 4.0                                                     # 점포 깊이(복도 ≥ 5 m 확보)
    cw = ly - 2 * sw if not two_corridor else (ly - 3 * sw) / 2
    cw = max(3.0, cw)
    shop_w = 4.0; n = int((lx - 1.0) // shop_w)
    rows = [(sw, sw + cw)] if not two_corridor else [(sw, sw + cw), (2 * sw + cw, 2 * sw + 2 * cw)]
    kp = n // 2                                                  # 통로 점포(중간 출구용) 인덱스
    px = 0.5 + kp * shop_w + shop_w / 2
    for ri, (c0, c1) in enumerate(rows):
        b.hwall(c0, 0, lx); b.hwall(c1, 0, lx)
        for k in range(n):
            x = 0.5 + k * shop_w
            if k > 0:
                b.vwall(x, c0 - sw, c0); b.vwall(x, c1, c1 + sw)
            b.door_h(c0, x + shop_w / 2, 2.4); b.door_h(c1, x + shop_w / 2, 2.4)   # 개방 점포 전면
            south_pass = (k == kp and n_exit >= 3 and ri == 0)
            north_pass = (k == kp and n_exit >= 4 and ri == len(rows) - 1)
            if not south_pass:
                b.block(x + 0.5, x + shop_w - 0.5, c0 - sw + 1.0, c0 - sw + 1.6, 1.8, "SHELF")
            if not north_pass:
                b.block(x + 0.5, x + shop_w - 0.5, c1 + sw - 1.6, c1 + sw - 1.0, 1.8, "SHELF")
    yc = (rows[0][0] + rows[0][1]) / 2 / ly
    b.exit("w", yc, 3.0, "STAIR_W"); b.exit("e", yc, 3.0, "STAIR_E")
    if n_exit >= 3: b.exit("s", px / lx, 2.0, "STAIR_S")
    if n_exit >= 4: b.exit("n", px / lx, 2.0, "STAIR_N")
    return b


def restaurant(lx, ly, n_exit, tag):
    b = Building("restaurant", lx, ly, tag)
    kd = min(4.0, ly * 0.3)
    b.hwall(ly - kd, 0, lx); b.door_h(ly - kd, lx * 0.8, 1.0)     # 주방
    b.block(1.0, lx - 1.0, ly - kd + 1.0, ly - kd + 1.8, 0.9, "KIT")
    gap = 1.2 if lx * ly < 200 else 1.4
    y = 2.4                                                      # 출입구 앞 통로 2.4 m
    while y + 0.8 < ly - kd - 1.0:
        x = 1.5
        while x + 1.2 < lx - 1.0:
            b.block(x, x + 1.2, y, y + 0.8, 0.75, "TBL"); x += 1.2 + gap
        y += 0.8 + gap
    b.exit("s", 0.5, 1.2, "MAIN")
    if n_exit >= 2: b.exit("n", 0.9, 0.9, "KITCHEN")
    return b


def theater(lx, ly, n_exit, tag):
    b = Building("theater", lx, ly, tag)
    sd = min(5.0, ly * 0.2)
    b.block(1.0, lx - 1.0, ly - sd, ly - 1.0, 0.8, "STAGE")
    n_ais = 2 if lx < 25 else 3
    aw = 1.5; bw = (lx - 2.0 - (n_ais - 1) * aw) / n_ais          # 좌석 블록 폭
    y = 3.6                                                      # 후방 로비 3.6 m(출구 앞 화원 자리)
    while y + 0.5 < ly - sd - 3.0:                               # 무대 앞 3 m 비움
        for k in range(n_ais):
            x0 = 1.0 + k * (bw + aw)
            b.block(x0, x0 + bw, y, y + 0.5, 0.9, "SEAT")
        y += 0.9
    b.exit("s", 0.25, 1.8, "MAIN_L"); b.exit("s", 0.75, 1.8, "MAIN_R")
    if n_exit >= 3: b.exit("w", 0.6, 1.2, "SIDE_W")
    if n_exit >= 4: b.exit("e", 0.6, 1.2, "SIDE_E")
    return b


def parking(lx, ly, n_exit, tag):
    b = Building("parking", lx, ly, tag)
    for x in np.arange(8.0, lx - 4.0, 8.0):
        for y in np.arange(8.0, ly - 4.0, 8.0):
            b.block(x - 0.3, x + 0.3, y - 0.3, y + 0.3, b.lz(), "COL")
    # 주차 열: y 방향 2열 슬롯(5 m) + 6 m 통로 반복
    y = 1.0
    while y + 10.0 < ly - 1.0:
        x = 1.0
        while x + 2.3 < lx - 1.0:
            if (int(x / 2.3) % 4) != 3:                              # 25% 빈 슬롯
                b.block(x + 0.3, x + 2.0, y, y + 4.6, 1.5, "CAR")
                b.block(x + 0.3, x + 2.0, y + 5.4, y + 10.0, 1.5, "CAR")
            x += 2.3
        y += 16.0
    b.exit("s", 0.5, 4.0, "RAMP")
    if n_exit >= 2: b.exit("n", 0.1, 1.0, "STAIR_N")
    if n_exit >= 3: b.exit("e", 0.9, 1.0, "STAIR_E")
    return b


def warehouse(lx, ly, n_exit, tag):
    b = Building("warehouse", lx, ly, tag)
    rw, aisle = 1.2, 4.0
    x = 4.0
    while x + rw < lx - 4.0:
        b.block(x, x + rw, 5.0, ly * 0.42, 2.5, "RACK"); b.block(x, x + rw, ly * 0.58, ly - 5.0, 2.5, "RACK")
        x += rw + aisle
    b.exit("s", 0.5, 2.0, "MAIN")
    if n_exit >= 2: b.exit("n", 0.5, 2.0, "BACK")
    if n_exit >= 3: b.exit("w", 0.5, 1.2, "SIDE")
    return b


def hall(lx, ly, n_exit, tag):
    """가구 없는 대형 홀 — 출구는 남·북 중앙(사용자 시험: 30×30, 문 2, 화원 좌하 2 MW)."""
    b = Building("theater", lx, ly, tag)
    b.kind = "hall"
    b.exit("s", 0.5, 1.8, "SOUTH"); 
    if n_exit >= 2: b.exit("n", 0.5, 1.8, "NORTH")
    if n_exit >= 3: b.exit("e", 0.5, 1.8, "EAST")
    return b


LAYOUTS = [
    ("D1", mart, [(15, 10, 1), (15, 10, 2), (30, 20, 1), (30, 20, 2), (30, 20, 3), (60, 40, 2), (60, 40, 3), (45, 30, 3)]),
    ("D2", office, [(20, 15, 1), (20, 15, 2), (40, 25, 2), (40, 25, 3), (60, 40, 2), (60, 40, 3), (30, 20, 2), (50, 30, 3)]),
    ("D3", school, [(40, 16, 1), (40, 16, 2), (60, 16, 2), (60, 16, 3), (80, 20, 2), (80, 20, 3), (50, 16, 2), (70, 18, 3)]),
    ("D4", hospital, [(30, 16, 1), (30, 16, 2), (50, 20, 2), (50, 20, 3), (70, 24, 2), (70, 24, 3), (40, 18, 2)]),
    ("D5", umall, [(60, 13, 2), (60, 13, 3), (100, 13, 3), (100, 13, 4), (150, 13, 4), (100, 23, 4), (150, 23, 4)]),
    ("D6", restaurant, [(12, 8, 1), (12, 8, 2), (20, 12, 1), (20, 12, 2), (30, 18, 1), (30, 18, 2), (16, 10, 2)]),
    ("D7", theater, [(20, 15, 2), (20, 15, 3), (30, 25, 2), (30, 25, 3), (45, 35, 3), (45, 35, 4)]),
    ("D8", parking, [(40, 30, 1), (40, 30, 2), (60, 45, 2), (60, 45, 3), (90, 60, 2), (90, 60, 3)]),
    ("D9", warehouse, [(30, 20, 1), (30, 20, 2), (50, 30, 2), (50, 30, 3), (80, 50, 2), (80, 50, 3), (40, 25, 2)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--growth", default="medium,fast")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    rows = []
    for code, fn, cfgs in LAYOUTS:
        if only and code not in only:
            continue
        for i, (lx, ly, ne) in enumerate(cfgs, 1):
            tag = "%s_%02d_%dx%d_e%d" % (code, i, lx, ly, ne)
            kw = dict(two_corridor=(ly > 20)) if fn is umall else {}
            b = fn(lx, ly, ne, tag, **kw)
            scen = [(w, g, None) for w in ("near_exit", "center", "far") for g in a.growth.split(",")]
            if len(b.exits) >= 2:
                scen.append(("near_exit", "fast", b.exits[1]["id"]))
            for w, g, blk in scen:
                name = "%s_%s_%s%s" % (tag, w, g[0], ("_blk" + blk) if blk else "")
                txt, meta = b.fds(name, w, g, blocked=blk)
                if not a.dry:
                    with open(os.path.join(a.out, name + ".fds"), "w", encoding="latin1") as f:
                        f.write(txt)
                    with open(os.path.join(a.out, name + "_meta.json"), "w") as f:
                        json.dump(meta, f, indent=1)
                rows.append((name, meta["cells"], meta["dx"], meta["t_end"], len(b.exits), meta["fire"]["x"], meta["fire"]["y"]))
    print("케이스 %d" % len(rows))
    cells = [r[1] for r in rows]
    print("셀수 %d ~ %d (중앙 %d), dx 분포 %s" % (min(cells), max(cells), int(np.median(cells)),
          {d: sum(1 for r in rows if r[2] == d) for d in (0.2, 0.25, 0.4)}))
    for r in rows[::7]:
        print("  %-42s cells=%8d dx=%.2f T=%3.0f exits=%d fire=(%.1f,%.1f)" % r)


if __name__ == "__main__":
    main()
