# -*- coding: utf-8 -*-
"""판단 시험대(judgment set) — '최단 경로'가 오답인 상황만 모은 케이스 생성기.

사용자 기준(2026-09-18): 탈출률이 아니라 "스스로 상황을 읽고 경로를 짜는가"를 본다.
그러려면 아무 생각 없이 가까운 출구로 가면 죽거나 늦는 배치가 필요하다.

| id | 배치 | 올바른 판단 |
|----|------|------------|
| J1 | 출구 1개(남), 그 앞 4 m 에 화원, 군중은 안쪽 | 초기엔 돌파, 커지면 반대편 대피(shelter) |
| J2 | 출구 2개(남·북), 남문 앞 3 m 화원, 군중은 남쪽 밀집 | 가까운 남문을 버리고 북문으로 |
| J3 | 출구 2개(남 좁음 1.2 m·동 넓음 3 m), 화원은 북서 구석, 군중은 남쪽 | 대기열을 읽고 동문으로 분산 |
| J4 | 출구 3개(남·동·북), 동문 폐쇄(잠김), 화원 중앙서측, 군중은 동쪽 | 폐쇄를 인지하고 남/북으로 |
사용: python gen_judgment.py <out_dir>  → <out>/<J*>/<J*>.fds + _meta.json
"""
import json
import os
import sys

from gen_bigspace_v2 import Building

SPEC = {
    # id: (lx, ly, [(side, pos, width, id)], fire_xy, q_max, blocked, 군중 spawn(cx,cy,r0,r1))
    "J1_single_fire_at_door": (30, 24, [("s", 0.5, 2.0, "SOUTH")], (15.0, 5.0), 3000.0, None, (15.0, 15.0, 2.0, 9.0)),
    "J2_near_door_on_fire": (30, 30, [("s", 0.5, 2.0, "SOUTH"), ("n", 0.5, 2.0, "NORTH")], (15.0, 4.0), 3000.0, None, (15.0, 9.0, 2.0, 7.0)),
    "J3_narrow_near_wide_far": (36, 24, [("s", 0.5, 1.2, "SOUTH"), ("e", 0.5, 3.0, "EAST")], (6.0, 20.0), 2000.0, None, (14.0, 6.0, 1.0, 7.0)),
    "J4_blocked_exit": (36, 24, [("s", 0.3, 2.0, "SOUTH"), ("e", 0.5, 2.0, "EAST"), ("n", 0.7, 2.0, "NORTH")], (12.0, 12.0), 3000.0, "EAST", (30.0, 12.0, 1.0, 5.0)),
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, (lx, ly, exits, fxy, q, blocked, spawn) in SPEC.items():
        b = Building("hall", lx, ly, name)
        for side, pos, w, eid in exits:
            b.exit(side, pos, w, eid)
        txt, meta = b.fds(name, "center", "fast", blocked=blocked, fire_xy=fxy, q_max=q, t_peak=40.0, t_end=240.0)
        meta["spawn"] = list(spawn); meta["judgment"] = name
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, name + ".fds"), "w", encoding="latin1").write(txt)
        json.dump(meta, open(os.path.join(d, name + "_meta.json"), "w"), indent=1)
        print("%-26s %5.0f m² dx %.2f 셀 %6d 출구 %s 화원 (%.1f,%.1f) %0.f kW 폐쇄 %s" % (
            name, lx * ly, meta["dx"], meta["cells"], [(e["id"], e["width"], e["blocked"]) for e in meta["exits"]],
            meta["fire"]["x"], meta["fire"]["y"], q, blocked))


if __name__ == "__main__":
    main()
