# TypeEvacSafe

**화재 피난자 한 사람 한 사람에게 판단 엔진을 하나씩 붙인 피난 시뮬레이터.**
**An evacuation simulator that gives every occupant their own decision engine.**

판단은 타입 안전 판독(Jev 형식) — 상태·질문·선택지를 주면 **생성 없이 한 번의 순전파로 선택지 확률**을 읽는다. 이동은 검증된 소셜포스 모델(rust_evac BR), 화재장은 FDS-GPU 해석 결과다.
Decisions use typed readout (Jev-style): give it the state, a question and typed options, and it reads **option probabilities in a single forward pass, generating nothing**. Locomotion is a validated social-force model (rust_evac BR); the fire field comes from FDS-GPU.

> 개발 / by **Meteor Simulation** · 곧 **bulc.msimul.com** 과 화재 시뮬레이션 **BULC(불씨)** 에 기능으로 추가된다.
> Coming to **bulc.msimul.com** and the **BULC** fire simulator. 특허 출원 준비 중 / Patent pending — see [라이선스와 특허 / License & Patent](#라이선스와-특허--license--patent).

---

## 100명 비교: Bonsai 2 27B(로컬) vs TypeSafe Jev / 100-occupant comparison

<p align="center"><img src="results/hall/bonsai_vs_jev.gif" width="960" alt="Ternary Bonsai 2 27B (left) vs TypeSafe Jev (right)"></p>

30×30 m 홀 · 2 MW 화원(★) · 100명(성인 55 · 보호자 10 · 어린이 10 · 노약자 15 · 부상자 10[보행불능 5]) + 소방관 2.
좌 **Ternary Bonsai 2 27B (PQ2_0, 7.2 GB, 로컬 즉답)** · 우 **TypeSafe Jev (API)**. t = 1 s 부터 0.5 s 간격 211 프레임, 10 fps = 실시간 5배속.
회색 = 호흡선(1.5 m) 연기 · 옅은 남보라 = 천장(2.88 m) 연기 · 주황 = 60 °C 이상 · ○ 이동 · ▽ 쓰러짐 · × 사망.
전체 영상 / full video: [`results/hall/bonsai_vs_jev.mp4`](results/hall/bonsai_vs_jev.mp4)

30 × 30 m hall, 2 MW fire (★), 100 occupants (55 adults, 10 guardians, 10 children, 15 elderly, 10 injured [5 non-ambulatory]) plus 2 firefighters.
Left: **Ternary Bonsai 2 27B**, local, direct readout. Right: **TypeSafe Jev**, API. 211 frames from t = 1 s at 0.5 s steps, 10 fps (5× real time).
Grey = smoke at 1.5 m, violet = smoke at the 2.88 m ceiling layer, orange = above 60 °C; ○ moving, ▽ collapsed, × dead.

### 결과 / Results

| 항목 / Metric | 규칙 매크로<br>Rule macro (no LLM) | Qwen3.8-27B<br>(생각 / thinking) | **Bonsai 2 27B**<br>(로컬 즉답 / local direct) | **TypeSafe Jev**<br>(API) |
|---|---:|---:|---:|---:|
| **결정당 시간** / **per-decision latency** | — | ~35 s | **7.2 s** | **0.14 s** |
| 판단 수 · 총 시간 / decisions · total | — | 30 · 1,050 s | 58 · 417 s | 42 · 5.7 s |
| 탈출률 / evacuated | 0.97 | 0.96 | 0.971 | **0.971** |
| 사망률 / died | 0.039 | 0.049 | **0.029** | **0.020** |
| 보행불능자 5명 구조 / non-ambulatory rescued (of 5) | 2 | 1 | 2 | **5** |
| 부상자 탈출률 / injured evacuated | 0.60 | 0.60 | **0.70** | **0.70** |
| 결정 분포 / decision mix | evacuate만<br>evacuate only | evacuate 24 | **evacuate 39 · escort 12 · rescue 7** | rescue 25 · evacuate 16 · escort 1 |
| 모드 엔트로피(개인차) / mode entropy | 0.21 | 0.05 | **0.50** | 0.42 |
| 목표 출구 엔트로피 / exit-choice entropy | 0.16 | 0.11 | 0.34 | **0.39** |
| 나쁜 출구 → 더 나은 쪽 전환 / switched to a safer exit | — | — | **0.54** | 0.00 |
| 소방관 역주행 깊이 / firefighter penetration | 4.2 m | 5.8 m | **6.4 m** | 0.0 m |
| RSET p50 / p90 | 26 / 44 s | 26 / 40 s | 26 / 97 s | 27 / 76 s |

읽는 법 / How to read it
- **Jev 는 구조에 가장 적극적이다** — 쓰러진 보행불능자 5명을 전원 구조했고 사망률이 가장 낮다. 대신 결정당 0.14 s 로 가장 빠르다(head 전부를 한 요청에 담는 구조 덕분).
  **Jev is the most rescue-oriented** — all five non-ambulatory occupants were carried out, lowest death rate, and fastest (all heads in one request).
- **Bonsai 는 로컬에서 그에 근접한다** — API 없이 7.2 GB 로 돌면서 결정을 가장 다양하게 갈랐고(엔트로피 0.50), 위험한 출구에서 더 나은 쪽으로 바꾼 비율이 0.54 로 가장 높다.
  **Bonsai comes close, locally** — 7.2 GB, no API, the most individualised decisions (entropy 0.50) and the highest rate of switching away from a hazardous exit (0.54).
- **획일적인 판단은 사람을 놓친다** — Qwen 27B 는 생각 모드에서 전원에게 `evacuate` 를 지시했고(엔트로피 0.05) 구조는 1건에 그쳤다.
  **Uniform decisions cost lives** — Qwen 27B in thinking mode told everyone to `evacuate` (entropy 0.05) and rescued only one.

## 판단 백엔드 정확도·속도 / Backend accuracy and speed

A100 80 GB 1장 · 저작 벤치 20건(30 판정) · 지연은 **서로 다른 상황 20개**를 연속 질의해 측정. 한 사람의 결정 = head 4~6개.
One A100 80 GB · 20 authored bench cases (30 judgments) · latency measured over **20 distinct situations** (never repeating a prompt). One decision = 4–6 heads.

| 백엔드 / Backend | 크기 / Size | 판독 / Readout | 정확도 / Accuracy | head당 / per head | **결정당 / per decision** |
|---|---:|---|---:|---:|---:|
| **TypeSafe Jev** (API) | — | typed probs | 0.93 (28/30) | — | **0.085 s** |
| **Qwen3.5-4B Q8_0** (openjev 방식) | 4.5 GB | 즉답 / direct | 0.70 (21/30) | 0.128 s | **0.487 s** |
| **Ternary Bonsai 2 27B** PQ2_0 | 7.2 GB | 즉답 / direct | **0.97 (29/30)** | 0.467 s | **1.78 s** |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | 생각 / thinking | 0.97 (29/30) | ~9 s | ~35 s |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | 즉답 / direct | 0.13 (4/30) | 0.13 s | 0.5 s |
| Qwen3.5-4B Q8_0 | 4.5 GB | 생각 / thinking | 0.57 (17/30) | 2.4 s | 9.5 s |

**같은 27B 베이스인데 삼진(Bonsai)은 생각 없이 0.97, 4비트(Q4)는 0.13.** 삼진 양자화가 로짓 판독에 훨씬 잘 맞는다 — 로컬에서 API 없이 판단 엔진을 돌릴 수 있는 이유다.
**Same 27B base: the ternary build scores 0.97 without thinking, the 4-bit build 0.13.** Ternary quantisation suits logit readout far better — that is what makes a local, API-free decision engine possible.

### 측정 시 주의 / Measurement pitfalls we hit

1. **같은 프롬프트를 반복 측정하지 말 것** — llama.cpp `cache_prompt` 가 프롬프트 처리를 건너뛰어 3~5배 낙관적인 값이 나온다.
   Never benchmark by repeating one prompt: `cache_prompt` skips prompt processing and inflates numbers 3–5×.
2. **동시 요청이 오히려 느리다** — GPU 한 장에서는 계산 병목이고, 요청마다 KV 캐시(≈640 MiB)를 밀어내 공용 접두 캐시까지 깬다. 8스레드가 직렬보다 1.8배 느렸다.
   Concurrency hurts on a single GPU: each request evicts ~640 MiB of KV cache and breaks the shared system-prompt prefix. 8 threads ran 1.8× slower than serial.
3. **생각 모드는 느릴 뿐 아니라 판단을 획일화한다** — 기본값은 `think=False`(즉답). Qwen3.8-27B Q4 처럼 즉답이 무너지는 모델에서만 `--think`.
   Thinking mode is not just slow, it flattens the decisions. Default is `think=False`; use `--think` only for models whose direct readout collapses (e.g. Qwen3.8-27B Q4).

---

## 무엇을 하려던 것인가 / What this is for

기존 피난 시뮬레이션은 모든 사람이 **최단 경로**로 가장 가까운 출구를 향한다. 실제 화재에서 사람이 하는 일 — 연기를 보고 돌아서고, 불이 난 출구를 버리고 먼 문으로 가고, 아이를 찾아 되돌아가고, 쓰러진 사람을 업고 나오고, 갈 데가 없으면 연기가 옅은 곳에 머무는 — 은 거기에 없다. 그래서 **판단과 이동을 나눴다**.

Conventional evacuation models send everyone down the **shortest path** to the nearest exit. What people actually do in a fire — turn back at smoke, abandon a burning exit for a distant one, go back for a child, carry a collapsed person, shelter where the smoke is thinnest — is missing. So we **separated judgment from locomotion**.

```
FDS-GPU 화재해석 / fire simulation (.sf/.smv)
        ↓  convert_sf2d
2D 위험장 / hazard fields: 온도 · 연기(K_s) · CO · O2 · 화염 · 출구별 거리장 · HRR(t) · 3층(보행/호흡/천장)
        ↓
┌─────────────────────────────┬──────────────────────────────────────┐
│ 판단층 / Decision (per person)│ 물리층 / Physics (shared)             │
│ mode  피난·대기·아이동행·     │ 추진 (e₀·v₀−v)/τ · 벽·사람 반발       │
│       구조·군중추종·돌파·대피 │ 복사력(시선 제한) · 연기력 · 위험장 밀침 │
│ target_exit  출구 ≤8         │ Purser FED · Frantzich 감속           │
│ rescue_target / feasible    │ 시야 3/K_s → 길찾기 오차               │
│ pace  뛰기·걷기·천천히·정지   │ ASET 초과 후 10 s 노출 → 사망          │
│ survive  생존 확률           │ 쓰러진 사람·시신은 장애물로 남음        │
└─────────────────────────────┴──────────────────────────────────────┘
        ↓
0.5 s 간격 전원 좌표·상태·모드 CSV · 궤적 PNG · MP4 · 행동 평가 지표
0.5 s trajectories for every occupant (CSV) · plots · video · behavioural metrics
```

개인 프로필이 결정과 물리 양쪽에 들어간다 — 성별·나이·역할(성인/어린이/노약자/보호자/부상자/소방관)·보행 불능·가족 링크. 속도·FED 민감도·복사 임계·시야 손실이 사람마다 다르다.
Each occupant carries a profile — sex, age, role (adult/child/elderly/guardian/injured/firefighter), mobility, family links — and it feeds both the decision layer and the physics.

**핵심 주장 / Core claim**: 판단층은 학습 없이 규칙 프롬프트만으로 동작한다. 학습이 필요한 것은 정확도가 아니라 **처리량**(1,000명 × 판단 10회 = 1만 결정)이고, 그건 큰 모델의 결정을 작은 모델에 증류해서 푼다.
The decision layer works with rule prompts alone, no training. What needs training is **throughput** (1,000 people × 10 decisions = 10k decisions) — solved by distilling a large model's decisions into a small one.

## 설치 / Install

```bash
git clone https://github.com/using76/TypeEvacSafe && cd TypeEvacSafe
pip install -r requirements.txt          # torch, numpy, matplotlib
```

### (a) 로컬 / local — Ternary Bonsai 2 27B (권장 / recommended)

```bash
# 전용 llama.cpp 포크 — 스톡 빌드는 파일을 읽고도 경고 없이 헛소리를 낸다
# Requires the PrismML fork; stock llama.cpp loads the file and silently emits garbage
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp llama_prism && cd llama_prism
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 20 --target llama-server        # A100=80, Ada=89, Hopper=90
cd ..

bash scripts/get_models.sh bonsai      # Ternary-Bonsai-2-27B-PQ2_0.gguf (7.2 GB) → models/
LLAMA_SERVER=llama_prism/build/bin/llama-server bash scripts/serve_bonsai.sh 0 8083 models/Ternary-Bonsai-2-27B-PQ2_0.gguf
python typeevac/jev_bench.py --backend llama --url http://127.0.0.1:8083     # 즉답이 기본 / direct by default
```

### (b) 소형 로컬 / small local — Qwen3.5-4B (openjev 방식, 가장 빠름 / fastest)

```bash
bash scripts/get_models.sh qwen4b        # Qwen3.5-4B-Q8_0.gguf (4.5 GB)
bash scripts/serve_llm.sh 0 99 8082 models/Qwen3.5-4B-Q8_0.gguf
```

### (c) TypeSafe Jev API

```bash
echo -n "apikey_..." > ~/.typesafe_key   # 권한 600 / chmod 600. 저장소에 넣지 말 것 / never commit it
```

### (d) 규칙 매크로 / rule macro — LLM 없이 도는 기준선 / baseline without any LLM

## 쓰기 / Usage

```bash
# 케이스 만들기 / build cases: 파라메트릭 대공간 438종 · 판단 시험대 4종 · 실측 CAD
python typeevac/gen_bigspace_v2.py cases/
python typeevac/gen_judgment.py    cases/        # J1~J4: 최단 경로가 '오답'인 배치 / shortest path is the wrong answer
python typeevac/prep_cad.py your_building.fds cases/MY --fire near_exit --ceiling

# FDS-GPU 해석 + 위험장 변환 / simulate and convert (z = 0.5 / 1.5 / 2.4 / 2.8 m, 9 channels)
python typeevac/gen_queue.py --gpu 0 --shard 0/1 --only J1_,J2_

# 100명 홀 시험 / 100-occupant hall run — 0.5 s 좌표 CSV
python typeevac/hall_test.py --case HALL_30x30_2MW --macro rule,llama --url http://127.0.0.1:8083 --out runs/hall
#   --think 를 붙이면 생각 모드(느리다) / add --think for thinking mode (slow)

# 행동 평가 / behavioural evaluation — '몇 명 나갔나'가 아니라 '읽고 짰나, 역할을 했나'
python typeevac/jev_eval.py --run runs/hall --macro rule,llama --field fields/HALL_30x30_2MW.npz

# 2 s PNG + MP4 (천장·호흡선 연기 오버레이 / two smoke layers)
python typeevac/hall_frames.py --out runs/hall --macro llama,typesafe --every 2 --t_min 1 \
       --mp4 compare.mp4 --video_every 0.5 --fps 10
```

## 평가 축 / Evaluation axes (탈출률이 아니라 / not the evacuation rate)

| 축 / Axis | 지표 / Metrics |
|---|---|
| A 상황 인지 / situation awareness | 위험 발생 후 결정 변경률·지연, 경로 노출 후회, 개선 스위치 비율, 나쁜 출구에서 전환율 |
| B 경로 계획 / route planning | 우회율, 연기 노출 p50/p90 |
| C 역할 수행 / role fulfilment | 보호자–아이 분리 시간·동반 탈출, 소방관 역주행 깊이, 부상자·보행불능자 구조/사망 |
| D 자율성 / autonomy | 같은 케이스에서 모드·목표 출구 엔트로피(획일적이지 않은가) |

100명이 전부 나가는 것이 목표가 아니다. **각자가 자기 상황을 읽고 자기 역할을 하는가**가 목표다.
Getting everyone out is not the goal. **Reading one's own situation and playing one's own role** is.

## 로드맵 / Roadmap

1. **head 묶음 요청 / batched heads** — 지금은 head 마다 요청을 보내 같은 상황을 4~6번 다시 읽는다. Jev 처럼 한 요청에 묶으면 로컬 백엔드가 3~4배 빨라진다.
2. **증류 / distillation** — Jev·Bonsai 결정 확률을 교사로 Qwen3.5-4B LoRA(옵션 KL). 목표 0.93 정확도 · 0.05 s/결정.
3. **결과 보정 / outcome correction** — 결정점 후보를 전수 롤아웃해 실제 결과(탈출·사망·구조)로 가중 갱신.
4. **확률 보정 / calibration** — `survive`·`rescue_feasible` 온도 스케일링 + ECE.
5. 1,000명 처리량, RiMEA/IMO 동역학 검증, BULC(불씨) 통합.

## 출처 / Credits

- 판독 방식 / typed readout: [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (MIT)
- head fan-out·실행 가드 / head fan-out and execution guards: [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT)
- 로컬 모델 / local models: [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0, PrismML) · [Qwen](https://huggingface.co/Qwen) (Apache-2.0)
- 보행 동역학·FED / pedestrian dynamics and FED: rust_evac (Meteor Simulation) · 화재장 / fire fields: FDS-GPU (BULC) · NIST FDS
- 모델 파일·내려받기 / model files: `models/README.md`, `scripts/get_models.sh`

## 라이선스와 특허 / License & Patent

소스는 **PolyForm Noncommercial 1.0.0** — 연구·평가·비상업 용도는 자유, 상업적 사용은 별도 계약.
Source is under **PolyForm Noncommercial 1.0.0**: free for research, evaluation and other noncommercial use; commercial use requires a separate agreement.

이 선택은 **특허 출원을 준비 중이기 때문**이다. Apache-2.0 은 배포자가 사용자에게 **명시적 특허 실시권**을 주고, MIT 는 배포한 코드에 대한 묵시적 실시권 주장 여지를 남긴다.
The choice reflects a **pending patent application**: Apache-2.0 grants an explicit patent licence to users, and MIT leaves room for an implied one.

상업 문의 / commercial licensing: **Meteor Simulation** — bulc.msimul.com
