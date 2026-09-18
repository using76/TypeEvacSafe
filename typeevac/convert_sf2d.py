# -*- coding: utf-8 -*-
"""피난 RL v2 — fds_gpu 결과(.smv + .sf 슬라이스) → 2D 위험장 npz. 3D 캐시를 거치지 않는다.

입력 케이스 디렉터리: <name>.fds, <name>.smv, <name>_*.sf (PBZ 2D 슬라이스 또는 3D 슬라이스), 선택 <name>_meta.json(생성기 출구 메타).
출력 npz (env.py 가 읽는 v1 키를 유지하고 v2 키를 추가):
  T KS CO O2 [t,nx,ny] float16 (z_breath)   FIRE [t,nx,ny] = 3층 HRRPUV 최대   T_TOP (z_top, 복사 근사용)
  WALL FREE OUTSIDE [nx,ny] bool — WALL 은 z_walk(0.5 m) 높이에서 .fds OBST/HOLE 을 직접 래스터(창문은 벽으로 남는다)
  EXIT [nx,ny] bool(열린 출구 셀)   EXIT_ID [nx,ny] int8 (0 없음, k = k번째 출구)   DIST [nx,ny] = 열린 출구 최소 거리
  DIST_E [E,nx,ny] 출구별 거리(막힌 출구는 inf)   exits_json: [{id, xb, width, center, blocked}]   HRR [t] kW(_hrr.csv)
출구: meta.json 이 있으면 그 사각형, 없으면 자동 — 외부(z_top 층에서 경계와 연결된 영역)에 인접한 실내 자유 셀을 군집화.
사용: python convert_sf2d.py --case <dir> --out <npz>   |   --root <dir> --out_dir <dir> [--shard i/n]
"""
import argparse
import glob
import heapq
import json
import os
import re
import struct

import numpy as np

K_EXT = 8700.0
MW_AIR, MW_CO = 28.97, 28.01
Z_WALK, Z_BREATH, Z_TOP = 0.5, 1.5, 2.4          # z_top 은 --z_top 로 바꿀 수 있다(천장 연기층 표시·복사 근사용)
Z_ABOVE_DOOR = 2.25                # 문 상단(2.1) 위·낮은 천장(2.4) 아래 — 외부 판정용 벽 층
QMAP = {"TEMPERATURE": "T", "SOOT DENSITY": "SOOT", "OXYGEN VOLUME FRACTION": "O2", "DENSITY": "RHO",
        "HRRPUV": "HRRPUV", "HRRPUV (kW/m3)": "HRRPUV"}


# ---------------------------------------------------------------- .fds 파싱
def records(t):
    out, i, n = [], 0, len(t)
    while i < n:
        if t[i] != "&":
            i += 1; continue
        j, q = i, None
        while j < n:
            c = t[j]
            if q:
                if c == q: q = None
            elif c in "'\"": q = c
            elif c == "/": break
            j += 1
        out.append(t[i:j + 1]); i = j + 1
    return out


def xb_of(r):
    m = re.search(r"\bXB\s*=\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)", r)
    return [float(v) for v in m.groups()] if m else None


def parse_fds(path):
    txt = open(path, encoding="latin1").read()
    meshes, obst, hole, yields = [], [], [], dict(co=0.004, soot=0.015)
    for r in records(txt):
        tag = re.match(r"&([A-Z0-9]+)", r).group(1)
        if tag == "MESH":
            ijk = re.search(r"IJK\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", r)
            meshes.append(dict(ijk=[int(v) for v in ijk.groups()], xb=xb_of(r)))
        elif tag == "OBST":
            xb = xb_of(r)
            if xb: obst.append(xb)
        elif tag == "HOLE":
            xb = xb_of(r)
            if xb: hole.append(xb)
        elif tag == "REAC":
            m = re.search(r"CO_YIELD\s*=\s*([\d.eE+-]+)", r)
            if m: yields["co"] = float(m.group(1))
            m = re.search(r"SOOT_YIELD\s*=\s*([\d.eE+-]+)", r)
            if m: yields["soot"] = float(m.group(1))
    return meshes, obst, hole, yields


class Grid:
    """여러 메시의 합집합을 덮는 전역 격자(dx = 최소 셀 크기). 메시 밖 셀은 통행 불가."""

    def __init__(self, meshes):
        dxs = [(m["xb"][1] - m["xb"][0]) / m["ijk"][0] for m in meshes]
        self.dx = float(min(dxs))
        self.x0 = min(m["xb"][0] for m in meshes); self.y0 = min(m["xb"][2] for m in meshes)
        x1 = max(m["xb"][1] for m in meshes); y1 = max(m["xb"][3] for m in meshes)
        self.nx = int(round((x1 - self.x0) / self.dx)); self.ny = int(round((y1 - self.y0) / self.dx))
        self.inmesh = np.zeros((self.nx, self.ny), bool)
        for m in meshes:
            i0, i1, j0, j1 = self.box(m["xb"])
            self.inmesh[i0:i1, j0:j1] = True
        self.meshes = meshes

    def box(self, xb):
        i0 = int(np.clip(np.floor((xb[0] - self.x0) / self.dx + 1e-6), 0, self.nx))
        i1 = int(np.clip(np.ceil((xb[1] - self.x0) / self.dx - 1e-6), 0, self.nx))
        j0 = int(np.clip(np.floor((xb[2] - self.y0) / self.dx + 1e-6), 0, self.ny))
        j1 = int(np.clip(np.ceil((xb[3] - self.y0) / self.dx - 1e-6), 0, self.ny))
        return i0, i1, j0, j1

    def occupancy(self, obst, hole, z):
        """z 높이 평면을 가로지르는 OBST 셀(셀 중심이 박스 안) − HOLE."""
        occ = np.zeros((self.nx, self.ny), bool)
        for xb in obst:
            if xb[4] <= z < xb[5] or (xb[5] - xb[4] < 1e-6 and abs(xb[4] - z) < 1e-6):
                i0, i1, j0, j1 = self.box(xb)
                if i1 > i0 and j1 > j0: occ[i0:i1, j0:j1] = True
        for xb in hole:
            if xb[4] <= z < xb[5]:
                i0, i1, j0, j1 = self.box(xb)
                if i1 > i0 and j1 > j0: occ[i0:i1, j0:j1] = False
        return occ


# ---------------------------------------------------------------- .sf 읽기
def read_sf(path):
    def rec(f):
        h = f.read(4)
        if len(h) < 4: return None
        n = struct.unpack("<i", h)[0]; b = f.read(n); f.read(4); return b
    frames, times = [], []
    with open(path, "rb") as f:
        for _ in range(3): rec(f)
        hdr = rec(f)
        i1, i2, j1, j2, k1, k2 = struct.unpack("<6i", hdr)
        ni, nj, nk = i2 - i1 + 1, j2 - j1 + 1, k2 - k1 + 1
        while True:
            tb = rec(f)
            if tb is None or len(tb) < 4: break
            db = rec(f)
            if db is None: break
            a = np.frombuffer(db, dtype="<f4")
            if a.size != ni * nj * nk: break
            frames.append(a.reshape((nk, nj, ni)).transpose(2, 1, 0)); times.append(struct.unpack("<f", tb[:4])[0])
    return np.array(frames, np.float32), np.array(times, np.float32), (i1, i2, j1, j2, k1, k2)


def smv_slices(smv_path):
    """.smv → [(mesh_idx, file, quantity, (i1,i2,j1,j2,k1,k2))]"""
    L = open(smv_path, errors="ignore").read().splitlines()
    out, trnz = [], {}
    gi = -1
    for i, ln in enumerate(L):
        if ln.startswith("GRID"):
            gi += 1
        if ln.startswith("TRNZ"):
            k = i + 2; zs = []
            while k < len(L) and L[k].strip() and re.match(r"\s*\d+\s+[-\d.]+", L[k]):
                zs.append(float(L[k].split()[1])); k += 1
            trnz[gi] = np.array(zs)
        if ln.startswith("SLCF") or ln.startswith("SLCC"):          # SLCC = 셀중심 슬라이스(fds_gpu 2D PBZ 출력)
            m = re.match(r"SLC[FC]\s+(\d+).*?&\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", ln)
            mesh = int(m.group(1)) - 1; b = tuple(int(v) for v in m.groups()[1:])
            out.append((mesh, L[i + 1].strip(), L[i + 2].strip().upper(), b, ln.startswith("SLCC")))
    return out, trnz


def cell_data(arr, bounds):
    """셀중심 슬라이스: 인덱스 0 은 고스트(1 과 중복) — 관측으로 판정해 제거."""
    i1, i2, j1, j2, k1, k2 = bounds
    a = arr
    if a.shape[1] > 1 and np.allclose(a[:, 0], a[:, 1]): a = a[:, 1:]
    if a.shape[2] > 1 and np.allclose(a[:, :, 0], a[:, :, 1]): a = a[:, :, 1:]
    return a


# ---------------------------------------------------------------- 거리장
def dijkstra(free, src, h):
    nx, ny = free.shape
    d = np.full((nx, ny), np.inf, dtype=np.float64)
    pq = []
    for x, y in zip(*np.nonzero(src & free)):
        d[x, y] = 0.0; pq.append((0.0, int(x), int(y)))
    heapq.heapify(pq)
    r2 = h * 1.41421356
    while pq:
        dd, x, y = heapq.heappop(pq)
        if dd > d[x, y] + 1e-9: continue
        for ax, ay, w in ((x + 1, y, h), (x - 1, y, h), (x, y + 1, h), (x, y - 1, h),
                          (x + 1, y + 1, r2), (x - 1, y - 1, r2), (x + 1, y - 1, r2), (x - 1, y + 1, r2)):
            if 0 <= ax < nx and 0 <= ay < ny and free[ax, ay]:
                if ax != x and ay != y and not (free[ax, y] and free[x, ay]): continue
                nd = dd + w
                if nd < d[ax, ay]:
                    d[ax, ay] = nd; heapq.heappush(pq, (nd, ax, ay))
    return d


def flood(mask, seeds):
    """4-이웃 연결 성분(seeds 에서 시작) — 외부 영역 판정."""
    nx, ny = mask.shape
    vis = np.zeros_like(mask); st = list(zip(*np.nonzero(seeds & mask)))
    for x, y in st: vis[x, y] = True
    while st:
        x, y = st.pop()
        for ax, ay in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= ax < nx and 0 <= ay < ny and mask[ax, ay] and not vis[ax, ay]:
                vis[ax, ay] = True; st.append((ax, ay))
    return vis


def components(mask):
    nx, ny = mask.shape
    lab = np.zeros((nx, ny), np.int32); n = 0
    for x, y in zip(*np.nonzero(mask)):
        if lab[x, y]: continue
        n += 1; lab[x, y] = n; st = [(x, y)]
        while st:
            cx, cy = st.pop()
            for ax in (cx - 1, cx, cx + 1):
                for ay in (cy - 1, cy, cy + 1):
                    if 0 <= ax < nx and 0 <= ay < ny and mask[ax, ay] and not lab[ax, ay]:
                        lab[ax, ay] = n; st.append((ax, ay))
    return lab, n


# ---------------------------------------------------------------- 케이스 변환
def convert(case_dir, out_path, zb=Z_BREATH, zw=Z_WALK, zt=Z_TOP, max_exits=8):
    fds = glob.glob(os.path.join(case_dir, "*.fds"))[0]
    smv = glob.glob(os.path.join(case_dir, "*.smv"))[0]
    name = os.path.splitext(os.path.basename(fds))[0]
    meshes, obst, hole, yields = parse_fds(fds)
    G = Grid(meshes)
    wall = G.occupancy(obst, hole, zw) | ~G.inmesh
    # ---- 슬라이스 조립 (전역 격자에 메시 블록 붙이기)
    sl, trnz = smv_slices(smv)
    fields = {}                                                    # (Q, zkey) -> [t,nx,ny]
    times = None
    for mesh, fn, q, b, cc in sl:
        Q = QMAP.get(q)
        if Q is None: continue
        arr, tt, bounds = read_sf(os.path.join(case_dir, fn))
        if arr.size == 0: continue
        arr = cell_data(arr, bounds)
        m = G.meshes[mesh]; mdz = (m["xb"][5] - m["xb"][4]) / m["ijk"][2]
        zc = m["xb"][4] + (np.arange(arr.shape[3]) + (0.5 if arr.shape[3] == m["ijk"][2] else 0.0)) * mdz
        if bounds[4] == bounds[5]:                                 # PBZ 2D: 헤더 k 는 노드 인덱스, 셀중심이면 셀 k 의 중심 = (k-0.5)dz
            zc = np.array([m["xb"][4] + (bounds[4] - (0.5 if cc else 0.0)) * mdz])
        for zkey, zt_ in (("walk", zw), ("breath", zb), ("top", zt)):
            k = int(np.argmin(np.abs(zc - zt_)))
            if abs(zc[k] - zt_) > 0.6: continue
            if (Q, zkey) in fields and arr.shape[3] == 1 and abs(zc[0] - zt_) > abs(fields[(Q, zkey)][1] - zt_): continue
            key = (Q, zkey)
            if key not in fields or fields[key][0] is None:
                fields[key] = [np.zeros((arr.shape[0], G.nx, G.ny), np.float32), float(zc[k])]
            i0, i1, j0, j1 = G.box(m["xb"])
            blk = arr[:, :, :, k]
            fields[key][0][:, i0:i0 + blk.shape[1], j0:j0 + blk.shape[2]] = blk[:, :i1 - i0, :j1 - j0]
            fields[key][1] = float(zc[k])
        if times is None or len(tt) > len(times): times = tt
    need = [("T", "breath"), ("SOOT", "breath"), ("O2", "breath"), ("RHO", "breath")]
    miss = [k for k in need if k not in fields]
    if miss: raise RuntimeError("%s: 슬라이스 없음 %s (있음 %s)" % (name, miss, sorted(fields)))
    T = fields[("T", "breath")][0]; soot = fields[("SOOT", "breath")][0]
    rho = fields[("RHO", "breath")][0].clip(0.05, None); o2 = fields[("O2", "breath")][0]
    fire = np.max([fields[k][0] for k in fields if k[0] == "HRRPUV"], axis=0) if any(k[0] == "HRRPUV" for k in fields) else np.zeros_like(T)
    ttop = fields[("T", "top")][0] if ("T", "top") in fields else T
    kstop = fields[("SOOT", "top")][0] * K_EXT if ("SOOT", "top") in fields else ks
    ks = soot * K_EXT
    co_ppm = (soot * (yields["co"] / yields["soot"]) / rho) * (MW_AIR / MW_CO) * 1e6
    # ---- 출구
    free = ~wall
    edge = np.zeros_like(free); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    # 외부 = 문 높이(2.1) 위 2.25 m 층에서 경계와 연결된 영역 — 문(HOLE ≤ 2.1)은 이 층에서 막혀 있어 실내로 새지 않는다
    wall_top = G.occupancy(obst, hole, Z_ABOVE_DOOR) | ~G.inmesh
    outside = flood(~wall_top, edge) & free
    interior = free & ~outside
    exits = []
    mp = os.path.join(case_dir, name + "_meta.json")
    exit_id = np.zeros((G.nx, G.ny), np.int8)
    if os.path.exists(mp):
        meta = json.load(open(mp))
        for e in meta["exits"][:max_exits]:
            i0, i1, j0, j1 = G.box(e["xb"])
            cells = np.zeros_like(free); cells[i0:i1, j0:j1] = True
            src = cells & interior if not e["blocked"] else cells
            # 열린 출구: 사각형 안 실내 셀이 없으면(문이 외벽 두께 안에만 있음) 인접 실내 셀로 확장
            if not e["blocked"] and not src.any():
                src = np.zeros_like(free)
                src[max(0, i0 - 1):i1 + 1, max(0, j0 - 1):j1 + 1] = True
                src &= interior
            exits.append(dict(id=e["id"], xb=e["xb"], width=e["width"], blocked=bool(e["blocked"]),
                              center=[(e["xb"][0] + e["xb"][1]) / 2, (e["xb"][2] + e["xb"][3]) / 2], cells=src))
    else:
        nb = np.zeros_like(free)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb |= np.roll(outside, (sx, sy), (0, 1))
        exit_cells = interior & nb
        lab, n = components(exit_cells)
        comps = sorted(range(1, n + 1), key=lambda c: -(lab == c).sum())[:max_exits]
        for k, c in enumerate(comps, 1):
            cells = lab == c; xs, ys = np.nonzero(cells)
            w = max(np.ptp(xs), np.ptp(ys)) * G.dx + G.dx
            cx, cy = G.x0 + (xs.mean() + 0.5) * G.dx, G.y0 + (ys.mean() + 0.5) * G.dx
            exits.append(dict(id="E%d" % k, xb=[G.x0 + xs.min() * G.dx, G.x0 + (xs.max() + 1) * G.dx,
                                                 G.y0 + ys.min() * G.dx, G.y0 + (ys.max() + 1) * G.dx],
                              width=float(w), blocked=False, center=[float(cx), float(cy)], cells=cells))
    walk = interior.copy()
    dist_e = np.full((len(exits), G.nx, G.ny), np.inf, np.float32)
    for k, e in enumerate(exits):
        exit_id[e["cells"] & free] = k + 1
        if not e["blocked"]:
            dist_e[k] = dijkstra(walk | e["cells"], e["cells"], G.dx)
    open_mask = np.zeros_like(free)
    for e in exits:
        if not e["blocked"]: open_mask |= e["cells"]
    dist = dist_e.min(axis=0) if len(exits) else np.full((G.nx, G.ny), np.inf, np.float32)
    reach = np.isfinite(dist) & interior
    # 화원 열방출률 시계열(kW): fds_gpu 의 <name>_hrr.csv(Time, HRR, ...) → times 에 보간. 없으면 HRRPUV 최대로 대체.
    hrr_t = None
    hp = os.path.join(case_dir, name + "_hrr.csv")
    if os.path.exists(hp):
        try:
            raw = np.genfromtxt(hp, delimiter=",", skip_header=2)
            if raw.ndim == 2 and raw.shape[0] > 2:
                hrr_t = np.interp(np.asarray(times, np.float64), raw[:, 0], raw[:, 1]).astype(np.float32)
        except Exception:
            hrr_t = None
    if hrr_t is None:
        hrr_t = np.full(len(times), float(fire.max()), np.float32)
    fire_Q = float(hrr_t.max())
    ej = [dict(id=e["id"], xb=[float(v) for v in e["xb"]], width=float(e["width"]), blocked=e["blocked"], center=e["center"])
          for e in exits]
    np.savez_compressed(
        out_path, case=name, x0=G.x0, y0=G.y0, dx=G.dx, z=fields[("T", "breath")][1], fire_Q=fire_Q,
        times=np.asarray(times, np.float32),
        T=T.astype(np.float16), T_TOP=ttop.astype(np.float16), KS=ks.astype(np.float16), KS_TOP=kstop.astype(np.float16),
        z_top=float(fields[("T", "top")][1]) if ("T", "top") in fields else float(zt), CO=co_ppm.astype(np.float16),
        O2=(o2 * 100).astype(np.float16), FIRE=fire.astype(np.float16),
        WALL=wall, EXIT=open_mask, EXIT_ID=exit_id, FREE=interior, OUTSIDE=outside,
        DIST=dist.astype(np.float32), DIST_E=dist_e, exits_json=json.dumps(ej), n_exits=len(exits), HRR=hrr_t)
    return dict(case=name, shape=list(T.shape), dx=G.dx, n_exits=len(exits), exits=[e["id"] for e in exits],
                T_max=float(T.max()), KS_max=float(ks.max()), fire_max=fire_Q,
                free=int(interior.sum()), reach=int(reach.sum()), dist_max=float(dist[reach].max()) if reach.any() else None,
                frames=int(T.shape[0]), z=[fields[k][1] for k in fields if k[0] == "T"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case"); ap.add_argument("--out")
    ap.add_argument("--root"); ap.add_argument("--out_dir"); ap.add_argument("--shard", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--z_top", type=float, default=Z_TOP, help="천장 연기층 높이(m) — .fds 에 그 높이 PBZ 슬라이스가 있어야 한다")
    a = ap.parse_args()
    if a.case:
        print(json.dumps(convert(a.case, a.out, zt=a.z_top), ensure_ascii=False)); return
    os.makedirs(a.out_dir, exist_ok=True)
    cases = sorted(d for d in os.listdir(a.root) if glob.glob(os.path.join(a.root, d, "*.smv")))
    if a.shard:
        i, n = map(int, a.shard.split("/")); cases = cases[i::n]
    ok = fail = 0
    with open(os.path.join(a.out_dir, "_index%s.jsonl" % ("_" + a.shard.replace("/", "of") if a.shard else "")), "a") as idx:
        for c in cases:
            op = os.path.join(a.out_dir, c + ".npz")
            if os.path.exists(op) and not a.force: continue
            try:
                r = convert(os.path.join(a.root, c), op, zt=a.z_top); ok += 1
                idx.write(json.dumps(r) + "\n"); idx.flush()
                print("[ok] %s %s exits=%s reach=%d/%d Tmax=%.0f" % (c[:30], r["shape"], r["exits"], r["reach"], r["free"], r["T_max"]), flush=True)
            except Exception as e:
                fail += 1; print("[fail] %s: %s" % (c, e), flush=True)
    print("완료 %d · 실패 %d" % (ok, fail))


if __name__ == "__main__":
    main()
