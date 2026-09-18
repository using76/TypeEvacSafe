# -*- coding: utf-8 -*-
"""대공간 케이스 생성 큐 — fds_gpu_v2 실행 → convert_sf2d → fields npz. GPU 하나가 샤드 하나를 순서대로 소화한다.

사용: python gen_queue.py --gpu 0 --shard 0/2 [--root /home/work/BULC_DATA/evac_big/cases] [--fields .../fields]
  · cases/<name>/<name>.fds (+ _meta.json) 를 순회. DONE/FAIL 마커가 있으면 건너뜀 → 재기동 안전.
  · 케이스당 timeout(기본 5400 s). 산출 .sf 가 27개 미만이면 FAIL.
  · 끝나면 --after 스크립트를 실행(GPU 유휴 방지).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--root", default="/home/work/BULC_DATA/evac_big/cases")
ap.add_argument("--fields", default="/home/work/BULC_DATA/evac_big/fields")
ap.add_argument("--bin", default="/home/work/GPU_ver2/target/release/fds_gpu_v2")
ap.add_argument("--timeout", type=int, default=5400)
ap.add_argument("--after", default=None)
ap.add_argument("--only", default=None, help="쉼표구분 이름 접두사")
a = ap.parse_args()
i_s, n_s = map(int, a.shard.split("/"))
os.makedirs(a.fields, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(a.root, "queue_g%d.log" % a.gpu), "a")


def say(s):
    line = "[%s g%d] %s" % (time.strftime("%m-%d %H:%M"), a.gpu, s)
    print(line, flush=True); log.write(line + "\n"); log.flush()


cases = sorted(d for d in os.listdir(a.root) if os.path.isfile(os.path.join(a.root, d, d + ".fds")))
if a.only:
    pre = tuple(a.only.split(",")); cases = [c for c in cases if c.startswith(pre)]
cases = cases[i_s::n_s]
say("큐 시작 %d 케이스 (shard %s)" % (len(cases), a.shard))
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), FDS_GPU_NVDB="0")
done = fail = skip = 0
for c in cases:
    d = os.path.join(a.root, c)
    if os.path.exists(os.path.join(d, "DONE")) or os.path.exists(os.path.join(d, "FAIL")):
        skip += 1; continue
    t0 = time.time()
    try:
        r = subprocess.run([a.bin, c + ".fds"], cwd=d, env=env, timeout=a.timeout,
                           stdout=open(os.path.join(d, "run.log"), "w"), stderr=subprocess.STDOUT)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    dt = time.time() - t0
    nsf = len(glob.glob(os.path.join(d, "*.sf")))
    if rc != 0 or nsf < 27:
        tail = ""
        try:
            tail = open(os.path.join(d, "run.log"), errors="ignore").read()[-200:].replace("\n", " ")
        except Exception:
            pass
        open(os.path.join(d, "FAIL"), "w").write("rc=%s sf=%d %.0fs %s\n" % (rc, nsf, dt, tail))
        say("FAIL %s rc=%s sf=%d %.0fs | %s" % (c, rc, nsf, dt, tail[-120:]))
        fail += 1; continue
    # 변환(CPU)
    t1 = time.time()
    cv = subprocess.run([sys.executable, os.path.join(here, "convert_sf2d.py"), "--case", d,
                         "--out", os.path.join(a.fields, c + ".npz")], capture_output=True, text=True)
    if cv.returncode != 0:
        open(os.path.join(d, "FAIL"), "w").write("convert: %s\n" % cv.stderr[-300:])
        say("FAIL(convert) %s %s" % (c, cv.stderr[-160:].replace("\n", " ")))
        fail += 1; continue
    info = cv.stdout.strip().splitlines()[-1]
    for pat in ("*.q", "*.s3d", "*.bf", "*.xyz"):                 # PL3D/SMOKE3D/BNDF 는 안 쓴다 — 케이스당 300 MB 절감
        for f in glob.glob(os.path.join(d, pat)):
            os.remove(f)
    open(os.path.join(d, "DONE"), "w").write("%.0fs sim, %.0fs convert\n%s\n" % (dt, time.time() - t1, info))
    try:
        j = json.loads(info); brief = "exits=%s reach=%d/%d Tmax=%.0f fire=%.0f" % (j["exits"], j["reach"], j["free"], j["T_max"], j["fire_max"])
    except Exception:
        brief = info[:100]
    say("ok %s sim=%.0fs conv=%.0fs size=%s | %s" % (c, dt, time.time() - t1,
        subprocess.run(["du", "-sh", d], capture_output=True, text=True).stdout.split()[0], brief))
    done += 1
say("큐 종료 완료 %d · 실패 %d · 건너뜀 %d" % (done, fail, skip))
if a.after and os.path.exists(a.after):
    subprocess.Popen(["bash", a.after], start_new_session=True)
