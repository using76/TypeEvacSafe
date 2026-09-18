# -*- coding: utf-8 -*-
"""CAD/실측 .fds(Tier A·B·C) → 피난 RL v2 케이스 변환. 형상은 그대로(선택 스케일), 출력·시간·화원만 바꾼다.

  · T_END 단축(기본 240 s), PL3D/SL3D/DEVC/BNDF/PROF/ISOF/CTRL·기존 SLCF 제거
  · z 0.5/1.5/2.4 m 2D 슬라이스 9채널 추가(gen_bigspace_v2 와 동일 규격)
  · --fire orig      : 원본 화원 유지
    --fire near_exit | center | far : 원본 화원의 SURF 를 INERT 로 바꾸고 새 버너(t² 성장, --qmax)를 자동 배치.
                       출구는 convert_sf2d 의 자동 판정(2.25 m 층 외부 연결)으로 찾고, 주 출구 = 가장 넓은 출구.
  · --scale s        : 평면 x,y 를 s 배(문 폭은 유지, 격자 dx 유지 → IJK 도 s 배)
  · <name>_meta.json 에 출구·화원 기록 → convert_sf2d 가 읽는다.
사용: python prep_cad.py <src.fds> <out_dir> [--tend 240] [--fire orig] [--growth fast] [--qmax 3000] [--scale 1.0] [--name X]
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_sf2d as cv                                             # noqa: E402

ALPHA = {"medium": 0.01172, "fast": 0.04689}
DROP = ("SLCF", "DEVC", "BNDF", "PROF", "ISOF", "CTRL", "SL3D", "PL3D", "TAIL")
QUANT = [("U-VELOCITY", None), ("V-VELOCITY", None), ("W-VELOCITY", None), ("PRESSURE", None),
         ("TEMPERATURE", None), ("DENSITY", None), ("VOLUME FRACTION", "OXYGEN"), ("DENSITY", "SOOT"), ("HRRPUV", None)]


def fmt_xb(xb):
    return ",".join("%.3f" % v for v in xb)


def scale_xb(r, s, keep_width=False):
    """레코드의 XB x,y 를 s 배. keep_width: 중심만 옮기고 폭 유지(문)."""
    xb = cv.xb_of(r)
    if xb is None: return r
    if keep_width:
        cx, cy = (xb[0] + xb[1]) / 2 * s, (xb[2] + xb[3]) / 2 * s
        hw, hd = (xb[1] - xb[0]) / 2, (xb[3] - xb[2]) / 2
        nb = [cx - hw, cx + hw, cy - hd, cy + hd, xb[4], xb[5]]
    else:
        nb = [xb[0] * s, xb[1] * s, xb[2] * s, xb[3] * s, xb[4], xb[5]]
    return re.sub(r"\bXB\s*=\s*[-\d.eE+]+\s*,\s*[-\d.eE+]+\s*,\s*[-\d.eE+]+\s*,\s*[-\d.eE+]+\s*,\s*[-\d.eE+]+\s*,\s*[-\d.eE+]+",
                  "XB=" + fmt_xb(nb), r, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--tend", type=float, default=240.0)
    ap.add_argument("--fire", default="orig", choices=["orig", "near_exit", "center", "far"])
    ap.add_argument("--growth", default="fast", choices=["medium", "fast"])
    ap.add_argument("--qmax", type=float, default=3000.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--kind", default="cad")
    ap.add_argument("--ceiling", action="store_true", help="벽 상단 높이에 천장 슬래브 추가(불씨 평면은 천장이 없어 연기가 벽 위로 빠져나간다)")
    a = ap.parse_args()
    txt = open(a.src, encoding="latin1").read()
    base = os.path.splitext(os.path.basename(a.src))[0]
    name = a.name or base
    recs = cv.records(txt)
    # 화재 SURF 식별
    fire_surfs = set()
    for r in recs:
        if r.startswith("&SURF") and re.search(r"HRRPUA|MLRPUA|HEAT_OF_VAPORIZATION", r, re.I):
            m = re.search(r"\bID\s*=\s*'([^']*)'", r)
            if m: fire_surfs.add(m.group(1))
    keep, drop = [], {}
    s = a.scale
    for r in recs:
        tag = re.match(r"&([A-Z0-9]+)", r).group(1)
        if tag in DROP:
            drop[tag] = drop.get(tag, 0) + 1; continue
        if tag == "TIME":
            r = "&TIME T_END=%.1f /" % a.tend
        elif tag == "DUMP":
            r = "&DUMP DT_SLCF=1.0, DT_HRR=1.0 /"
        elif tag == "HEAD":
            r = "&HEAD CHID='%s' /" % name
        elif tag == "MESH" and s != 1.0:
            m = re.search(r"IJK\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", r)
            i, j, k = (int(v) for v in m.groups())
            r = scale_xb(r, s)
            r = re.sub(r"IJK\s*=\s*\d+\s*,\s*\d+\s*,\s*\d+", "IJK=%d,%d,%d" % (round(i * s), round(j * s), k), r)
        elif tag in ("OBST", "VENT") and s != 1.0:
            r = scale_xb(r, s)
        elif tag == "HOLE" and s != 1.0:
            r = scale_xb(r, s, keep_width=True)
        if a.fire != "orig" and tag in ("OBST", "VENT") and fire_surfs:
            for fs in fire_surfs:                                    # 원본 화원 → INERT (형상 유지)
                r = re.sub(r"'%s'" % re.escape(fs), "'INERT'", r)
        keep.append(r)
    if not any(k.startswith("&DUMP") for k in keep):
        keep.append("&DUMP DT_SLCF=1.0, DT_HRR=1.0 /")
    # ---- 형상 파싱(스케일 반영본) → 출구·화원
    tmp = "\n".join(keep) + "\n&TAIL /\n"
    os.makedirs(a.out, exist_ok=True)
    tp = os.path.join(a.out, "_tmp.fds"); open(tp, "w", encoding="latin1").write(tmp)
    meshes, obst, hole, yields = cv.parse_fds(tp); os.remove(tp)
    G = cv.Grid(meshes)
    wall = G.occupancy(obst, hole, cv.Z_WALK) | ~G.inmesh
    free = ~wall
    edge = np.zeros_like(free); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    outside = cv.flood(~(G.occupancy(obst, hole, cv.Z_ABOVE_DOOR) | ~G.inmesh), edge) & free
    interior = free & ~outside
    nb = np.zeros_like(free)
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb |= np.roll(outside, (sx, sy), (0, 1))
    lab, n = cv.components(interior & nb)
    exits = []
    for c in sorted(range(1, n + 1), key=lambda c: -(lab == c).sum())[:8]:
        xs, ys = np.nonzero(lab == c)
        exits.append(dict(id="E%d" % (len(exits) + 1), width=float(max(np.ptp(xs), np.ptp(ys)) * G.dx + G.dx), blocked=False,
                          xb=[G.x0 + xs.min() * G.dx, G.x0 + (xs.max() + 1) * G.dx, G.y0 + ys.min() * G.dx, G.y0 + (ys.max() + 1) * G.dx],
                          center=[float(G.x0 + (xs.mean() + 0.5) * G.dx), float(G.y0 + (ys.mean() + 0.5) * G.dx)], ncell=int(len(xs))))
    fire_meta = dict(where="orig")
    if a.fire != "orig":
        if not exits: raise SystemExit("%s: 출구를 못 찾아 화원 배치 불가" % name)
        qmax = a.qmax; alpha = ALPHA[a.growth]; tpk = math.sqrt(qmax / alpha)
        side = max(1.2, round(math.sqrt(qmax / 2500.0) / G.dx) * G.dx)
        occ = G.occupancy(obst, hole, 0.6) | ~G.inmesh | outside
        r = int(math.ceil((side / 2 + 0.2) / G.dx))
        nx, ny = occ.shape
        c = np.zeros((nx + 1, ny + 1), np.int64); c[1:, 1:] = np.cumsum(np.cumsum(occ, 0), 1)
        ok = np.zeros((nx, ny), bool)
        for i in range(r, nx - r):
            for j in range(r, ny - r):
                ok[i, j] = (c[i + r + 1, j + r + 1] - c[i - r, j + r + 1] - c[i + r + 1, j - r] + c[i - r, j - r]) == 0
        ok &= interior
        if not ok.any(): raise SystemExit("%s: 버너 자리 없음" % name)
        II, JJ = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        P = np.stack([G.x0 + (II + 0.5) * G.dx, G.y0 + (JJ + 0.5) * G.dx], -1)
        ex = max(exits, key=lambda e: e["width"]); exc = np.array(ex["center"])
        if a.fire == "near_exit":
            # 출구 중심에서 실내 쪽 4 m — 방향은 출구 셀에서 실내 자유 셀 무게중심 쪽
            d0 = np.linalg.norm(P - exc, axis=-1)
            near = interior & (d0 < 6.0)
            dirv = (P[near].mean(0) - exc) if near.any() else np.zeros(2)
            dirv = dirv / (np.linalg.norm(dirv) + 1e-9)
            d = np.linalg.norm(P - (exc + dirv * 4.0), axis=-1)
        elif a.fire == "center":
            cen = P[interior].mean(0); d = np.linalg.norm(P - cen, axis=-1)
        else:
            d = -np.linalg.norm(P - exc, axis=-1)
        d[~ok] = np.inf
        k = int(np.argmin(d)); i, j = divmod(k, ny)
        fx, fy = float(P[i, j, 0]), float(P[i, j, 1]); h = side / 2
        z0 = meshes[0]["xb"][4]
        keep.append("&SURF ID='EVAC_BURNER', HRRPUA=%.1f, RAMP_Q='RQ_EVAC', TMP_FRONT=300.0 /" % (qmax / side ** 2))
        ts = [i_ * min(tpk, a.tend) / 8.0 for i_ in range(9)]
        if tpk < a.tend: ts += [tpk + (a.tend - tpk) * i_ / 3.0 for i_ in range(1, 4)]
        for t in ts:
            keep.append("&RAMP ID='RQ_EVAC', T=%.1f, F=%.4f /" % (t, min(1.0, (t / tpk) ** 2)))
        keep.append("&OBST ID='EVAC_FIRE', XB=%s, SURF_IDS='EVAC_BURNER','INERT','INERT' /" % fmt_xb([fx - h, fx + h, fy - h, fy + h, z0, z0 + G.dx]))
        fire_meta = dict(where=a.fire, x=fx, y=fy, side=side, growth=a.growth, alpha=alpha, q_max=qmax, t_peak=tpk, main_exit=ex["id"])
    if a.ceiling:
        # 천장: 높이 ≥ 2.0 m 벽들의 상단 최대값에서 dx 두께 슬래브, 범위는 그 벽들의 바운딩 박스
        tall = [xb for xb in obst if xb[5] >= 2.0]
        if tall:
            zt = max(xb[5] for xb in tall)
            bx = [min(xb[0] for xb in tall), max(xb[1] for xb in tall), min(xb[2] for xb in tall), max(xb[3] for xb in tall)]
            keep.append("&OBST ID='EVAC_CEILING', XB=%s, SURF_ID='INERT' /" % fmt_xb(bx + [zt, zt + G.dx]))
            print("천장 추가 z=%.2f~%.2f bbox=%s" % (zt, zt + G.dx, [round(v, 1) for v in bx]))
    sl = ["&SLCF QUANTITY='%s', %sPBZ=%.2f, CELL_CENTERED=.TRUE. /" % (q, "SPEC_ID='%s', " % sp if sp else "", z)
          for z in (0.5, 1.5, 2.4) for q, sp in QUANT]
    with open(os.path.join(a.out, name + ".fds"), "w", encoding="latin1") as f:
        f.write("\n".join(keep) + "\n" + "\n".join(sl) + "\n&TAIL /\n")
    cells = sum(m["ijk"][0] * m["ijk"][1] * m["ijk"][2] for m in meshes)
    meta = dict(name=name, kind=a.kind, source=base, scale=s, dx=G.dx, cells=cells, t_end=a.tend, n_mesh=len(meshes),
                mesh=[G.x0, G.x0 + G.nx * G.dx, G.y0, G.y0 + G.ny * G.dx], fire=fire_meta,
                exits=[{k: v for k, v in e.items() if k != "ncell"} for e in exits], n_exits=len(exits),
                interior_cells=int(interior.sum()), area_m2=float(interior.sum() * G.dx ** 2))
    json.dump(meta, open(os.path.join(a.out, name + "_meta.json"), "w"), indent=1)
    print("%s: 메시 %d 셀 %.2fM dx %.3f 실내 %.0f m² 출구 %s 화원 %s 제거 %s" % (
        name, len(meshes), cells / 1e6, G.dx, meta["area_m2"], [(e["id"], round(e["width"], 1)) for e in exits], fire_meta.get("where"), drop))


if __name__ == "__main__":
    main()
