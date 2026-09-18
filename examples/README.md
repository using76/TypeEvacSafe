# 예제 케이스

| 파일 | 내용 |
|---|---|
| `HALL_30x30_2MW.fds` | 30×30×3 m 홀, 남·북 중앙 문, (5,10) 2 MW(t² fast, 40 s 도달). README 의 100명 시험 케이스 |
| `J2_near_door_on_fire.fds` | 판단 시험대 J2 — 남문 앞 3 m 에 3 MW. 가까운 문을 버리고 북문을 골라야 한다 |

FDS-GPU(또는 FDS)로 해석한 뒤 `convert_sf2d.py --z_top 2.8` 로 위험장 npz 를 만든다.
