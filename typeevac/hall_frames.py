# -*- coding: utf-8 -*-
"""hall_test 의 0.5 s 좌표 CSV → 프레임 PNG·MP4. 배경은 두 층 연기(호흡선 1.5 m 진하게, 천장 2.8 m 반투명)와 고온부.

판단층 여러 개를 가로로 나란히 그린다(예: Qwen 27B vs TypeSafe Jev).
사용: python hall_frames.py --out runs/hall_test --macro llama,typesafe --every 2 --mp4 hall.mp4 --video_every 0.5 --fps 8
"""
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

COL = {"adult": "tab:blue", "child": "tab:cyan", "elderly": "tab:purple", "guardian": "tab:green",
       "firefighter": "tab:red", "injured": "tab:orange"}
MARK = {"waiting": "o", "moving": "o", "down": "v", "exited": "^", "dead": "x"}


def load(csv_path):
    by_t = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        by_t[round(float(r["t"]), 1)].append(r)
    return by_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="hall_test 출력 디렉터리(<macro>_positions.csv 가 있는 곳)")
    ap.add_argument("--macro", default="rule,llama,typesafe")
    ap.add_argument("--field", default="/home/work/BULC_DATA/evac_big/fields/HALL_30x30_2MW.npz")
    ap.add_argument("--every", type=float, default=2.0, help="PNG 프레임 간격(s)")
    ap.add_argument("--png_dir", default=None)
    ap.add_argument("--t_max", type=float, default=0); ap.add_argument("--t_min", type=float, default=0.0)
    ap.add_argument("--mp4", default=None, help="파일명을 주면 mp4 도 만든다")
    ap.add_argument("--video_every", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--fire", default="5,10", help="화원 표시 위치 x,y")
    a = ap.parse_args()
    png_dir = a.png_dir or os.path.join(a.out, "frames")
    os.makedirs(png_dir, exist_ok=True)
    fx, fy = [float(v) for v in a.fire.split(",")]

    z = np.load(a.field)
    dx = float(z["dx"]); x0 = float(z["x0"]); y0 = float(z["y0"])
    KS = z["KS"].astype(np.float32); T = z["T"].astype(np.float32); WALL = z["WALL"]
    KS_TOP = z["KS_TOP"].astype(np.float32) if "KS_TOP" in z.files else None       # 천장층 연기(반투명 오버레이)
    z_top = float(z["z_top"]) if "z_top" in z.files else 2.4
    hrr_t = z["HRR"].astype(np.float32) if "HRR" in z.files else None
    ext = [x0 - dx / 2, x0 + WALL.shape[0] * dx - dx / 2, y0 - dx / 2, y0 + WALL.shape[1] * dx - dx / 2]

    macros = [m for m in a.macro.split(",") if os.path.exists(os.path.join(a.out, "%s_positions.csv" % m))]
    data = {m: load(os.path.join(a.out, "%s_positions.csv" % m)) for m in macros}
    times = sorted(set.intersection(*[set(d) for d in data.values()])) if len(data) > 1 else sorted(next(iter(data.values())))
    t_end = a.t_max or max(times)
    frames = [t for t in times if a.t_min <= t <= t_end and abs(t / a.every - round(t / a.every)) < 1e-6]
    print("판단층 %s · PNG %d장(%.1f s 간격) · 천장층 %.2f m %s" % (macros, len(frames), a.every, z_top, "있음" if KS_TOP is not None else "없음"))

    def panel(ax, m, t, small=False):
        fi = min(KS.shape[0] - 1, int(round(t)))
        ax.imshow(WALL.T, origin="lower", cmap="gray_r", extent=ext, alpha=0.5, zorder=1)
        if KS_TOP is not None:      # 천장 연기: 먼저 퍼지는 층 — 옅은 남보라 반투명
            ax.imshow(np.ma.masked_less(np.clip(KS_TOP[fi], 0, 3).T, 0.05), origin="lower", cmap="BuPu",
                      extent=ext, alpha=0.32, vmin=0, vmax=3, zorder=2)
        ax.imshow(np.ma.masked_less(np.clip(KS[fi], 0, 2).T, 0.02), origin="lower", cmap="Greys",
                  extent=ext, alpha=0.62, vmin=0, vmax=2, zorder=3)          # 호흡선 연기
        ax.imshow(np.ma.masked_less(T[fi].T, 60.0), origin="lower", cmap="autumn_r", extent=ext,
                  alpha=0.5, vmin=60, vmax=300, zorder=4)                    # 60 °C 이상
        cnt = defaultdict(int)
        for r in data[m][t]:
            st = r["state"]; cnt[st] += 1
            if st == "exited":
                continue                                                     # 나간 사람은 그리지 않는다
            ax.plot(float(r["x"]), float(r["y"]), MARK[st], color=COL[r["role"]],
                    ms=(4 if small else (5.5 if r["role"] != "firefighter" else 7)), mec="k", mew=0.35, zorder=6)
        ax.plot(fx, fy, "r*", ms=12 if small else 16, zorder=7)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        return cnt, fi

    def draw(t, path, dpi=80):
        fig, axs = plt.subplots(1, len(macros), figsize=(6.2 * len(macros), 6.8), squeeze=False)
        for ax, m in zip(axs[0], macros):
            cnt, fi = panel(ax, m, t)
            hrr = " - HRR %.0f kW" % hrr_t[min(fi, len(hrr_t) - 1)] if hrr_t is not None else ""
            ax.set_title("%s   t=%.1f s%s\ninside %d - down %d - exited %d - dead %d" % (
                m, t, hrr, cnt["moving"] + cnt["waiting"], cnt["down"], cnt["exited"], cnt["dead"]), fontsize=10)
        for r_, c in COL.items():
            axs[0][0].plot([], [], "o", color=c, label=r_)
        axs[0][0].plot([], [], "s", color="#8c6bb1", alpha=0.5, label="smoke @%.1fm" % z_top)
        axs[0][0].plot([], [], "s", color="0.45", label="smoke @1.5m")
        axs[0][0].legend(fontsize=7, loc="upper right", ncol=2)
        plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close(fig)

    for t in frames:
        draw(t, os.path.join(png_dir, "t%04d.png" % round(t)))

    # 요약 시트 12장
    pick = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, min(12, len(frames)))]
    fig, axs = plt.subplots(3, 4, figsize=(19, 14))
    for ax, t in zip(axs.flat, pick):
        cnt, fi = panel(ax, macros[-1], t, small=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("t=%.0f s  inside %d exited %d dead %d" % (t, cnt["moving"] + cnt["waiting"], cnt["exited"], cnt["dead"]), fontsize=9)
    fig.suptitle("HALL 30x30 - 2 MW - 100 occupants - %s" % macros[-1], fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "sheet_%s.png" % macros[-1]), dpi=80); plt.close(fig)

    if a.mp4:
        vt = [t for t in times if a.t_min <= t <= t_end and abs(t / a.video_every - round(t / a.video_every)) < 1e-6]
        tmp = tempfile.mkdtemp(prefix="vid_")
        for k, t in enumerate(vt):
            draw(t, os.path.join(tmp, "f%05d.png" % k))
        mp4 = os.path.join(a.out, a.mp4)
        cmd = ["ffmpeg", "-y", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
               "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4]
        r = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print("mp4 %s · %d 프레임(%.1f s 간격) %d fps → %.0f 초 영상, 실시간 %.1f 배속, rc=%d %s" % (
            mp4, len(vt), a.video_every, a.fps, len(vt) / max(a.fps, 1), a.fps * a.video_every, r.returncode,
            r.stderr[-200:] if r.returncode else ""))
    print("완료 →", png_dir)


if __name__ == "__main__":
    main()
