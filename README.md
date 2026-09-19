# TypeEvacSafe

**Jev 형식의 타입 안전 판단 엔진을 로컬 LLM 으로 개조하고, 그 위에 화재 피난 모델을 얹은 것.**
**A Jev-style typed decision engine rebuilt on local LLMs, with a fire-evacuation model on top.**

Meteor Simulation · 화재 시뮬레이션 [BULC](https://bulc.msimul.com) 의 피난 기능으로 통합 예정 · 특허 출원 준비 중

| | |
|---|---|
| 판단 엔진 | Ternary Bonsai 2 27B(로컬, 7.2 GB) · TypeSafe Jev(API) · Qwen3.5-4B(로컬, 4.5 GB) |
| 화재장 | [BULC](https://bulc.msimul.com) 화재 시뮬레이터 결과(.sf/.smv) |
| 이동 모델 | 소셜포스(rust_evac BR) + Purser FED + Frantzich–Nilsson 감속 |
| 출력 | 0.5 s 간격 전원 좌표·상태 CSV, 궤적 PNG, MP4, 행동 평가 지표 |
| 라이선스 | PolyForm Noncommercial 1.0.0 |

[한국어](#한국어) · [English](#english)

---

# 한국어

<p align="center"><img src="results/hall/bonsai_vs_jev.gif" width="960" alt="Bonsai 2 27B vs TypeSafe Jev"></p>

*30×30 m 홀 · 2 MW 화원(★) · 100명(성인 55 · 보호자 10 · 어린이 10 · 노약자 15 · 부상자 10[보행불능 5]) + 소방관 2. 좌 Ternary Bonsai 2 27B(로컬 즉답) · 우 TypeSafe Jev. t=1 s 부터 0.5 s 간격 211 프레임, 실시간 5배속. 회색 = 호흡선(1.5 m) 연기, 남보라 = 천장(2.88 m) 연기, 주황 = 60 °C 이상. ○ 이동 · ▽ 쓰러짐 · × 사망. 전체 영상: [`results/hall/bonsai_vs_jev.mp4`](results/hall/bonsai_vs_jev.mp4)*

## 목차

1. [배경과 목표](#1-배경과-목표)
2. [1부 — Jev 형식 판단 엔진의 로컬 LLM 개조](#2-1부--jev-형식-판단-엔진의-로컬-llm-개조)
3. [2부 — TypeEvacSafe 피난 모델의 원리와 수식](#3-2부--typeevacsafe-피난-모델의-원리와-수식)
4. [실측 결과](#4-실측-결과)
5. [설치와 실행](#5-설치와-실행)
6. [저장소 구성](#6-저장소-구성)
7. [로드맵](#7-로드맵)
8. [출처 · 라이선스 · 특허](#8-출처--라이선스--특허)

## 1. 배경과 목표

기존 피난 시뮬레이션은 모든 사람이 최단 경로로 가장 가까운 출구를 향한다. 실제 화재에서 사람이 하는 일 — 연기를 보고 돌아서고, 불이 난 출구를 버리고 먼 문으로 가고, 아이를 찾아 되돌아가고, 쓰러진 사람을 업고 나오고, 갈 데가 없으면 연기가 옅은 곳에 머무는 — 은 거기에 없다.

우리는 이 판단을 **사람마다 하나씩 붙은 판단 엔진**이 하게 만들고자 했다. 조건은 세 가지였다.

- **생성이 아니라 판독**: 매 판단마다 문장을 생성하면 1,000명 규모에서 쓸 수 없다. 상태·질문·선택지를 주면 한 번의 순전파로 선택지 확률을 읽어야 한다. TypeSafe 의 Jev 가 보여준 형식이다.
- **로컬 실행**: 건물 도면과 인원 정보는 밖으로 내보내기 어렵다. API 없이 GPU 한 장에서 돌아야 한다.
- **물리와 분리**: 걷고 밀리고 연기에 중독되는 것은 검증된 수식이 있다. 학습이나 LLM 이 건드릴 일이 아니다.

그래서 작업은 두 부분으로 나뉜다. 1부는 Jev 형식을 공개 모델 위에서 재현하는 것, 2부는 그 판단 엔진을 화재 물리 위에 얹어 피난 모델로 만드는 것이다.

## 2. 1부 — Jev 형식 판단 엔진의 로컬 LLM 개조

### 2.1 판독 원리

채팅 모델에 다음 형식의 프롬프트를 준다.

```
system : "주어진 규칙을 상황에 적용해 선택지 하나를 고른다. 대문자 글자 하나로만 답한다." + 규칙 문단
user   : {"situation": {...}, "question": "...", "options": [{"letter":"A","description":"..."}, ...]}
```

모델이 답을 **생성하게 두지 않는다.** 답이 나올 위치의 다음 토큰 로짓 $z \in \mathbb{R}^{|V|}$ 에서 선택지 글자 토큰 $\{A, B, \dots\}$ 의 로짓만 골라 소프트맥스를 취한다.

$$p_k = \frac{\exp z_{\ell_k}}{\sum_{j} \exp z_{\ell_j}}, \qquad \ell_k = \text{token}(\text{letter}_k)$$

- 생성 토큰 0개, 파싱 실패 없음, 결정당 순전파 1회.
- 각 글자가 토크나이저에서 **한 토큰**이어야 한다(A~P 확인 완료).
- llama.cpp 서버에서는 `max_tokens=1`, `logprobs=true, top_logprobs=20` 으로 같은 판독을 얻는다. transformers 백엔드는 마지막 위치 로짓을 직접 읽는다.

이 판독 방식은 [openjev](https://github.com/TheoLeeCJ/openjev) 가 Qwen3.5-4B 로 보인 것이고, 우리는 그것을 llama.cpp 백엔드와 삼진 27B 모델로 옮겼다.

### 2.2 질문 fan-out 과 실행 가드

한 사람의 결정은 질문(head) 여러 개로 나뉜다. [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) 의 구조를 따라, 한 상황에 모든 head 를 동시에 묻고 **선택된 mode 에 해당하는 head 만 실행**한다.

| head | 선택지 | 실행 조건 |
|---|---|---|
| `mode` | evacuate / wait / escort / rescue / follow_crowd / breakthrough / shelter | 항상 |
| `target_exit` | 출구 1~8 | mode ∈ {evacuate, breakthrough, follow_crowd, escort} |
| `rescue_target` | 위험자 1~6 / none | mode = rescue |
| `rescue_feasible` | yes / no / insufficient | rescue 판정 — **no 면 evacuate 로 강등** |
| `pace` | run / walk / slow / stop | 항상 (속도 배율 1.4 / 1.0 / 0.6 / 0) |
| `survive` | likely / unlikely | 확률 값으로만 사용 |

가드가 있어야 "구조하겠다"면서 "구조 불가"라고 답한 모순 상태가 실행되지 않는다.

### 2.3 상황 서술(evidence)의 구성 원칙

판단에 쓰이는 정보를 **짧게, 앞에, 관계로** 쓴다. 실측에서 얻은 규칙이다.

- 위험을 문장 맨 앞에: `"DANGER, FIRE AT THIS EXIT: X1, 9 m …"` — 뒤에 붙이면 27B 도 9 m 화재 출구를 27 m 안전 출구보다 골랐다.
- 출구마다 `usable / route(화원 곁 통과 여부) / distance / direction / smoke / heat / people_waiting / queue_s` 를 준다.
- 개인 프로필: `role, sex, age, mobility, exposure_so_far, can_see_m, knows_building`, 보호자면 `my_child`, 반경 15 m 의 `people_in_danger_nearby`.
- 규칙은 시스템 프롬프트에 10줄 — 화염·120 °C 구역 회피, 안전한 먼 출구 우선, 혼잡 출구 변경, 보호자는 아이를 두고 가지 않음, 성인은 자기 노출이 낮을 때 구조, 소방관은 역주행해 가장 위험한 생존자를 확보, 돌파는 최후 수단, 전면 봉쇄 시 대피, 즉시 행동.

### 2.4 어떤 모델이 즉답 판독을 견디는가

| 모델 | 판독 | 저작 벤치(30 판정) |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_M | 즉답 | 4/30 |
| Qwen3.8-27B UD-Q4_K_M | 생각 후 글자 | 29/30 |
| **Ternary Bonsai 2 27B PQ2_0** | **즉답** | **29/30** |
| Qwen3.5-4B Q8_0 | 즉답 | 21/30 |
| TypeSafe Jev | 타입 확률 | 28/30 |

같은 27B 베이스인데 4비트 양자화본은 즉답에서 글자 위치 편향으로 무너지고(선택지 순서를 뒤집으면 답이 바뀐다), 삼진 양자화본(Bonsai)은 생각 없이 29/30 을 낸다. 삼진 표현이 로짓 판독에 훨씬 잘 맞는다는 것이 이 프로젝트의 가장 중요한 실측이다. 생각 모드는 정확도는 올리지만 결정당 수십 초가 걸리고, 시뮬레이션에서는 100명에게 같은 결정을 내리는 획일화를 일으켰다.

### 2.5 백엔드 구성

| 백엔드 | 실행 | 비고 |
|---|---|---|
| `typesafe` | TypeSafe API | head 전부를 한 요청에. 가장 빠름(0.085 s/결정) |
| `llama` | llama.cpp 서버 | Bonsai(PrismML 포크 필요) / Qwen GGUF. 기본 즉답, `--think` 로 생각 모드 |
| `hf` | transformers | openjev 와 동일한 로짓 판독 |
| `rule` | LLM 없음 | 최근접 안전 출구 + 보호자 동행 + 소방관 최근접 위험자 |

## 3. 2부 — TypeEvacSafe 피난 모델의 원리와 수식

```
BULC 화재 시뮬레이션 (.fds 입력 → .sf/.smv)
        ↓  convert_sf2d
2D 위험장 : T, K_s, CO, O2, 화염, 벽, 출구별 거리장, HRR(t)   ×  z = 0.5 / 1.5 / 2.4 / 2.8 m
        ↓
┌───────────────────────────────┬────────────────────────────────────┐
│ 판단층 — 사람마다 하나 (1부)     │ 물리층 — 전원 공통                   │
│ mode · target_exit · rescue · │ 소셜포스 · 복사력 · 연기력 · 위험장 밀침 │
│ pace · survive                │ FED · 감속 · 시야 · 사망 규칙          │
└───────────────────────────────┴────────────────────────────────────┘
        ↓
0.5 s 좌표·상태·모드 CSV · 궤적 · MP4 · 행동 평가
```

### 3.1 화재장 — BULC 결과의 변환

화재 해석은 [BULC](https://bulc.msimul.com) 로 한다. FDS 호환 `.fds` 를 읽는 GPU 화재 시뮬레이터이며, 우리는 3D 볼륨 대신 **2D 슬라이스 4층**(보행 0.5 / 호흡 1.5 / 천장 하부 2.4 / 천장 2.8 m) × 9채널(U V W P T ρ O₂ soot HRRPUV)만 1 s 간격으로 저장해 케이스당 100 MB 이하로 만든다. 변환기가 만드는 것:

- 소광계수 $K_s = 8700\,\rho_{soot}$ [1/m] (FDS 기본 질량소광계수 8,700 m²/kg)
- CO 농도(ppm) $= \dfrac{\rho_{soot}\,(Y_{CO}/Y_{soot})}{\rho}\cdot\dfrac{M_{air}}{M_{CO}}\cdot 10^6$ — `.fds` 의 수율비로 환산
- 벽 마스크: `.fds` 의 OBST/HOLE 을 z = 0.5 m 에서 래스터(창문은 벽으로 남는다)
- 출구: 문 상단(2.1 m) 위 2.25 m 층에서 도메인 경계와 연결된 영역을 "외부"로 보고, 그에 인접한 실내 셀을 군집화. 출구별 8-이웃 Dijkstra 거리장 $D_k(x)$
- 화원 열방출률 $Q(t)$: BULC 의 `_hrr.csv`

### 3.2 이동 — 소셜포스 (rust_evac BR 모델 이식)

에이전트 $i$ 의 가속도:

$$\dot{\mathbf v}_i=\frac{\mathbf e_i\,v_i^{*}-\mathbf v_i}{\tau}+\frac{1}{m}\Big(\sum_{j}\mathbf f_{ij}+\sum_{w}\mathbf f_{iw}\Big)+\mathbf f^{rad}_i+\mathbf f^{haz}_i,\qquad \tau=0.5\ \mathrm{s}$$

- 희망 방향 $\mathbf e_i$ 는 목표 출구 거리장의 8-이웃 내리막 방향에 시야 오차(3.4절)를 더한 것, 희망 속도 $v_i^{*}=v_{0,i}\cdot m_{pace}\cdot s(K_s)$.
- 반발력(Helbing 형, rust_evac 계수): $\mathbf f_{ij}=\big[A e^{(r_{ij}-d)/B}+k\,(r_{ij}-d)^{+}\big]\mathbf n_{ij}+\kappa\,(r_{ij}-d)^{+}\,\Delta v^{t}_{ji}\,\mathbf t_{ij}$, $A=2000,\ B=0.08,\ k=1.2\times10^{5},\ \kappa=2.4\times10^{5}$, 컷오프 2.5 m. 벽은 4방향 최근접 벽 셀 하나씩(면 법선만).
- 서브스텝 $8\times0.0125$ s = 0.1 s, 벽 관통 시 축 분리 슬라이딩, 출구선은 서브스텝마다 판정.

### 3.3 화재가 몸에 미치는 것

**복사력** — 화원이 보일 때(시선 12 m 이내)만:

$$q=\frac{\chi_r\,Q(t)}{4\pi r^{2}},\qquad |\mathbf f^{rad}|=s_r\min\!\Big(\frac{(q-q_{th})^{+}}{q_{ref}},3\Big)\big(0.4+0.6\,\max(0,\ \hat{\mathbf o}\cdot(-\hat{\mathbf n}))\big)$$

$\chi_r=0.3,\ q_{th}=2.5$ kW/m²(소방관 7), $q_{ref}=5,\ s_r=4$.

**위험장 밀침** — 인지 거리 $d_p=3$ m 의 시선 제한 차분:

$$H=\frac{(T-40)^{+}}{40}+\frac{K_s}{1},\qquad \mathbf f^{haz}=-d_p\nabla H\cdot\frac{v^{*}}{\tau}$$

복사력과 밀침의 **합**을 $0.9\,v^{*}/\tau$ 로 캡한다 — 희망(1.0)보다 항상 작아야 뚫고 나가는 선택(`pace=run`, 배율 1.4)이 가능하고, 각각 캡하면 합 1.8 로 화원 곁 경로에서 영구 정체가 생긴다(실측).

**감속(Frantzich–Nilsson)**: $s(K_s)=\max\big(0.15,\ 1-a\,\bar K_s\big)$, $a=0.081$(노약자 0.12, 부상자 0.10), $\bar K_s$ 는 시정수 2 s 의 지수이동평균.

**FED(Purser)**:

$$\dot F_{tox}=\Big[2.764\times10^{-5}\,CO^{1.036}\cdot\frac{e^{0.193\,CO_2+2.0004}}{7.1}+\frac{1}{e^{\,8.13-0.54(20.9-O_2)}}\Big]\ \mathrm{min^{-1}},\qquad \dot F_{heat}=\frac{1}{5\times10^{7}\,T^{-3.4}}\ \mathrm{min^{-1}}$$

$$F=\int\big(k_{tox}\dot F_{tox}+k_{heat}\dot F_{heat}\big)\,\frac{dt}{60}$$

역할 계수 $k_{tox}$: 어린이 1.3, 소방관 0(SCBA); $k_{heat}$: 노약자 1.2, 소방관 0.2(방화복).

### 3.4 시야와 길찾기

BULC 의 VISIBILITY 와 같은 정의 $V=3/K_s$ (반사 표지). 시야가 짧아지면 출구 방향을 틀리게 안다:

$$\sigma(V)=\frac{\pi}{2}\,\mathrm{clip}\!\Big(\frac{10-V}{10-1},0,1\Big),\qquad \theta_t=a\,\theta_{t-1}+\sqrt{1-a^{2}}\,\sigma\,\varepsilon_t,\quad a=e^{-\Delta t/3\,\mathrm{s}}$$

희망 방향을 $\theta_t$ 만큼 회전한다. 시야 10 m 이상이면 오차 0, 1 m 면 σ = 90°. 소방관은 0.3배(열화상·훈련).

### 3.5 쓰러짐 · 사망 · 구조

- **쓰러짐**: $F\ge1$ (ASET 초과). 그 자리에 남아 장애물이 된다.
- **사망**: 쓰러진 뒤 열($T>40$ °C)·연기($K_s>0.2$)·FED 증가 중 하나에 **10 s 이상 노출**, 또는 $F\ge2$.
- **구조**: 구조자가 위험자(F ≥ 0.3, 쓰러짐, 연기 속 5 s 정체) 1 m 이내에 도달하면 확보 → 함께 이동(부축 0.6 / 운반 0.5 m/s) → 출구 도달 시 구조 완료. 성인은 자기 F > 0.5 면 확보 불가. 확보 중 구조자가 쓰러지면 둘 다 쓰러진다.
- **가족**: 보호자는 동행 중 희망 속도가 아이 속도로 제한되고, 아이는 보호자를 추종한다(안 보이면 보호자의 목표 출구 거리장을 따른다). 소방관은 t = 60~120 s 에 주 출구에서 진입한다.

### 3.6 판단층에 주는 출구 요약

| 양 | 정의 |
|---|---|
| 안전 점수 | $s_e=\exp\!\big(-\tfrac{(T_e-40)^{+}}{60}\big)\,\exp(-K_{s,e})\cdot\mathbb 1[\text{2 m 내 화염 없음}]$ |
| 대기 시간 | $q_e=\dfrac{n_e}{1.3\,w_e}$ [s], $n_e$ = 출구 3 m 내 인원, 1.3 인/(m·s)(SFPE 유효 유량) |
| 화원 경유 판정 | 화원 2 m 권역 측지 거리 $F(x)$ 와 권역에서의 출구 거리 $D^{F}_k$ 로 $F(x)+D^F_k-D_k(x)<3$ m 이면 "경로가 화원 곁을 지난다" |
| 규칙 매크로 비용 | $c_k=\dfrac{D_k}{\max(s_e,0.05)}+1.2\,q_e+\mathbb 1[\text{경유}]\min\!\big(\tfrac{Q}{50},60\big)$ |

판단은 기본 15 s 주기(홀 시험은 30 s) 또는 사건(출구 안전 급락, 경로가 화원 곁이 됨, 시야 < 3 m, 피보호자 분리, 위험자 발견) 시 호출된다.

### 3.7 개인 프로필

| 역할 | $v_0$ (m/s) | 반경 (m) | $k_{tox}$ | $k_{heat}$ | $q_{th}$ | 비고 |
|---|---|---|---|---|---|---|
| 성인 | 1.2 (여 ×0.92) | 0.20 | 1.0 | 1.0 | 2.5 | |
| 어린이 | 0.8 | 0.15 | 1.3 | 1.0 | 2.5 | 보호자 추종 |
| 노약자 | 0.7 | 0.25 | 1.0 | 1.2 | 2.5 | 감속 계수 0.12 |
| 보호자 | 1.2 | 0.20 | 1.0 | 1.0 | 2.5 | 아이 1명 연결 |
| 부상자 | 0.5 / 0(보행불능) | 0.25 | 1.0 | 1.0 | 2.5 | 보행불능은 시작부터 쓰러짐 |
| 소방관 | 1.2 | 0.25 | 0 | 0.2 | 7.0 | 60~120 s 진입, 시야 오차 0.3배 |

### 3.8 평가 — 탈출률이 아니라

탈출률은 어떤 판단층이든 0.96~0.99 로 붙어 판정력이 없다. 네 축으로 잰다.

| 축 | 지표 |
|---|---|
| A 상황 인지 | 위험 발생 후 결정 변경률·지연, 경로 노출 후회, 개선 스위치 비율, 나쁜 출구에서 더 나은 쪽 전환율 |
| B 경로 계획 | 우회율, 연기 노출 p50/p90 |
| C 역할 수행 | 보호자–아이 분리 시간·동반 탈출, 소방관 역주행 깊이·체류, 부상자·보행불능자 구조/사망 |
| D 자율성 | 같은 케이스에서 모드·목표 출구 엔트로피 |

## 4. 실측 결과

### 4.1 100명 홀 (BULC 해석 2 MW, 30×30×3 m, 출구 2)

| 항목 | 규칙 매크로 | Qwen3.8-27B(생각) | **Bonsai 2 27B(로컬 즉답)** | **TypeSafe Jev** |
|---|---:|---:|---:|---:|
| **결정당 시간**(시뮬레이션 실측) | — | ~35 s | **7.2 s** | **0.14 s** |
| 결정당 head 수 · head당 시간 | — | 5.9 · ~6 s | 5.9 · 1.22 s | 한 요청 |
| 탈출률 | 0.97 | 0.96 | 0.971 | 0.971 |
| 사망률 | 0.039 | 0.049 | **0.029** | **0.020** |
| 보행불능자 5명 구조 | 2 | 1 | 2 | **5** |
| 부상자 탈출률 | 0.60 | 0.60 | 0.70 | 0.70 |
| 결정 분포 | evacuate만 | evacuate 24 | **evacuate 39 · escort 12 · rescue 7** | rescue 25 · evacuate 16 · escort 1 |
| 모드 엔트로피(개인차) | 0.21 | 0.05 | **0.50** | 0.42 |
| 목표 출구 엔트로피 | 0.16 | 0.11 | 0.34 | **0.39** |
| 나쁜 출구 → 더 나은 쪽 전환 | — | — | **0.54** | 0.00 |
| 소방관 역주행 깊이 | 4.2 m | 5.8 m | **6.4 m** | 0.0 m |
| RSET p50 / p90 | 26 / 44 s | 26 / 40 s | 26 / 97 s | 27 / 76 s |

- Jev 는 구조에 가장 적극적이다 — 보행불능자 5명 전원 구조, 사망률 최저, head 를 한 요청에 담아 가장 빠르다.
- Bonsai 는 로컬에서 그에 근접한다 — API 없이 7.2 GB 로 돌면서 결정이 가장 다양하고(엔트로피 0.50), 위험한 출구에서 더 나은 쪽으로 바꾼 비율이 가장 높다(0.54).
- 획일적 판단은 사람을 놓친다 — Qwen 27B 생각 모드는 전원에게 `evacuate` 를 지시했고 구조 1건에 그쳤다.

**4.1 과 4.2 의 결정당 시간이 다른 이유** — 같은 백엔드라도 조건이 다르다. 시뮬레이션(4.1)은 한 결정에 head 5.9개(mode · target_exit · rescue_target · rescue_feasible · pace · survive)를 묻고 증거가 길며(출구·위험자·가족 항목, 1,000~1,600 토큰) 4스레드 동시 요청으로 돌렸다(Bonsai 기준 결정 58건, 판단 시간 합 417 s). 벤치(4.2)는 head 3.8개, 약 890 토큰, 직렬이다. **head당 시간으로 환산하면 Bonsai 1.22 s(시뮬) 대 0.47 s(벤치)** 이고, 차이는 증거 길이(약 1.6배)와 동시 요청 손실(약 1.8배)로 설명된다. 결정당 값은 head 수를 곱한 것이다: 0.47 × 3.8 = 1.78 s, 1.22 × 5.9 = 7.2 s. Jev 도 같은 이유로 0.085 s(벤치) → 0.136 s(시뮬)이다.

### 4.2 판단 백엔드 정확도와 속도

A100 80 GB 1장 · 저작 벤치 20건(30 판정) · 지연은 서로 다른 상황 20개를 **직렬**로 질의해 측정(약 890 토큰, 결정당 head 3.8개). 시뮬레이션 조건의 값은 4.1 표를 본다.

| 백엔드 | 크기 | 판독 | 정확도 | head당 | **결정당**(벤치, head 3.8개) |
|---|---:|---|---:|---:|---:|
| **TypeSafe Jev** | — | 타입 확률 | 0.93 (28/30) | — | **0.085 s** |
| **Qwen3.5-4B Q8_0** | 4.5 GB | 즉답 | 0.70 (21/30) | 0.128 s | **0.487 s** |
| **Ternary Bonsai 2 27B PQ2_0** | 7.2 GB | 즉답 | **0.97 (29/30)** | 0.467 s | **1.78 s** |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | 생각 | 0.97 (29/30) | ~9 s | ~35 s |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | 즉답 | 0.13 (4/30) | 0.13 s | 0.5 s |
| Qwen3.5-4B Q8_0 | 4.5 GB | 생각 | 0.57 (17/30) | 2.4 s | 9.5 s |

측정 시 주의 — 우리가 실제로 밟은 함정:
1. 같은 프롬프트를 반복 측정하지 말 것. llama.cpp `cache_prompt` 가 프롬프트 처리를 건너뛰어 3~5배 낙관적인 값이 나온다.
2. 동시 요청이 오히려 느리다. GPU 한 장에서는 계산 병목이고, 요청마다 KV 캐시(≈640 MiB)를 밀어내 공용 접두 캐시까지 깬다. 8스레드가 직렬보다 1.8배 느렸다.
3. 생각 모드는 느릴 뿐 아니라 판단을 획일화한다. 기본값은 즉답(`think=False`).

## 5. 설치와 실행

```bash
git clone https://github.com/using76/TypeEvacSafe && cd TypeEvacSafe
pip install -r requirements.txt          # torch, numpy, matplotlib
```

**(a) 로컬 — Ternary Bonsai 2 27B (권장)**

```bash
# 전용 llama.cpp 포크 — 스톡 빌드는 파일을 읽고도 경고 없이 헛소리를 낸다
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp llama_prism && cd llama_prism
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80   # A100=80, Ada=89, Hopper=90
cmake --build build -j 20 --target llama-server && cd ..

bash scripts/get_models.sh bonsai        # Ternary-Bonsai-2-27B-PQ2_0.gguf (7.2 GB) → models/
LLAMA_SERVER=llama_prism/build/bin/llama-server bash scripts/serve_bonsai.sh 0 8083 models/Ternary-Bonsai-2-27B-PQ2_0.gguf
python typeevac/jev_bench.py --backend llama --url http://127.0.0.1:8083
```

**(b) 소형 로컬 — Qwen3.5-4B**

```bash
bash scripts/get_models.sh qwen4b        # Qwen3.5-4B-Q8_0.gguf (4.5 GB)
bash scripts/serve_llm.sh 0 99 8082 models/Qwen3.5-4B-Q8_0.gguf
```

**(c) TypeSafe Jev API** — `echo -n "apikey_..." > ~/.typesafe_key` (권한 600, 저장소에 넣지 말 것)

**(d) 규칙 매크로** — LLM 없이 도는 기준선

**화재 케이스 → 위험장 → 시뮬레이션**

```bash
python typeevac/gen_bigspace_v2.py cases/           # 파라메트릭 대공간 438종
python typeevac/gen_judgment.py    cases/           # 판단 시험대 J1~J4 (최단 경로가 오답인 배치)
python typeevac/prep_cad.py your_building.fds cases/MY --fire near_exit --ceiling   # 실측 도면
python typeevac/gen_queue.py --gpu 0 --shard 0/1 --only J1_,J2_      # BULC 해석 + 위험장 변환

python typeevac/hall_test.py --case HALL_30x30_2MW --macro rule,llama --url http://127.0.0.1:8083 --out runs/hall
python typeevac/jev_eval.py   --run runs/hall --macro rule,llama --field fields/HALL_30x30_2MW.npz
python typeevac/hall_frames.py --out runs/hall --macro llama,typesafe --every 2 --t_min 1 --mp4 compare.mp4 --fps 10
```

## 6. 저장소 구성

| 경로 | 내용 |
|---|---|
| `typeevac/jev_prompts.py` | 규칙 프롬프트(영/한)와 head·선택지 정의 — 판단을 바꾸려면 여기를 고친다 |
| `typeevac/jev_decide.py` | 상황 서술 생성, head fan-out, 백엔드(TypeSafe / llama.cpp / transformers), 실행 가드 |
| `typeevac/jev_bench.py` | 저작 벤치 20건 |
| `typeevac/env2.py`, `env_v1.py` | 물리층(3.2~3.7 절)과 rust_evac 상수 |
| `typeevac/hall_test.py`, `jev_scenario.py` | 시나리오 실행, 0.5 s 좌표 CSV |
| `typeevac/jev_eval.py` | 행동 평가 A~D |
| `typeevac/hall_frames.py` | PNG 프레임·MP4(두 층 연기 오버레이) |
| `typeevac/gen_*.py`, `prep_cad.py`, `convert_sf2d.py`, `gen_queue.py` | 케이스 생성, BULC 해석 큐, 위험장 변환 |
| `typeevac/jev_teacher.py` | 교사 데이터 생성(증류용) |
| `models/README.md`, `scripts/` | 모델 파일 목록, 서버 기동·내려받기 |
| `results/` | 벤치 JSON, 100명 시험 영상·프레임·평가 |
| `docs/` | 계획서, 학습 전략, 백엔드 비교 |

## 7. 로드맵

1. **head 묶음 요청** — 지금은 head 마다 요청을 보내 같은 상황을 4~6번 다시 읽는다. 한 요청에 묶거나 상태 프리필을 공유하면 로컬 백엔드가 3~4배 빨라진다.
2. **증류** — Jev·Bonsai 결정 확률을 교사로 Qwen3.5-4B LoRA(옵션 KL). 목표 정확도 0.93 · 0.05 s/결정.
3. **결과 보정** — 결정점 후보를 전수 롤아웃해 실제 결과(탈출·사망·구조)로 가중 갱신.
4. **확률 보정** — `survive`·`rescue_feasible` 온도 스케일링 + ECE.
5. 1,000명 처리량, RiMEA/IMO 동역학 검증, BULC 통합.

## 8. 출처 · 라이선스 · 특허

- 판독 방식: [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (MIT) · head fan-out과 실행 가드: [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT)
- 모델: [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0, PrismML) · [Qwen](https://huggingface.co/Qwen) (Apache-2.0)
- 화재 시뮬레이션: [BULC](https://bulc.msimul.com) (Meteor Simulation) · 보행 동역학·FED: rust_evac (Meteor Simulation)

소스는 **PolyForm Noncommercial 1.0.0** — 연구·평가·비상업 용도는 자유, 상업적 사용은 별도 계약. 특허 출원을 준비 중이라 Apache-2.0(명시적 특허 실시권)과 MIT(묵시적 실시권 여지)를 쓰지 않는다. 본 라이선스는 저작권 실시 허락이며 특허권을 부여하지 않는다(`NOTICE.md`). 상업 문의: Meteor Simulation — bulc.msimul.com

---

# English

<p align="center"><img src="results/hall/bonsai_vs_jev.gif" width="960" alt="Bonsai 2 27B vs TypeSafe Jev"></p>

*30 × 30 m hall, 2 MW fire (★), 100 occupants (55 adults, 10 guardians, 10 children, 15 elderly, 10 injured [5 non-ambulatory]) plus 2 firefighters. Left: Ternary Bonsai 2 27B, local, direct readout. Right: TypeSafe Jev. 211 frames from t = 1 s at 0.5 s steps, 5× real time. Grey = smoke at 1.5 m, violet = smoke at the 2.88 m ceiling layer, orange = above 60 °C. ○ moving, ▽ collapsed, × dead. Full video: [`results/hall/bonsai_vs_jev.mp4`](results/hall/bonsai_vs_jev.mp4)*

## Contents

1. [Background and goal](#1-background-and-goal)
2. [Part 1 — Rebuilding a Jev-style decision engine on local LLMs](#2-part-1--rebuilding-a-jev-style-decision-engine-on-local-llms)
3. [Part 2 — The TypeEvacSafe evacuation model: principles and equations](#3-part-2--the-typeevacsafe-evacuation-model-principles-and-equations)
4. [Measurements](#4-measurements)
5. [Install and run](#5-install-and-run)
6. [Repository layout](#6-repository-layout)
7. [Roadmap](#7-roadmap)
8. [Credits, license, patent](#8-credits-license-patent)

## 1. Background and goal

Conventional evacuation models send everyone down the shortest path to the nearest exit. What people actually do in a fire — turn back at smoke, abandon a burning exit for a distant one, go back for a child, carry a collapsed person, shelter where the smoke is thinnest — is not in them.

We wanted those decisions made by **a decision engine attached to each person**, under three constraints.

- **Readout, not generation.** Generating text for every decision does not scale to 1,000 occupants. Given a state, a question and typed options, the engine must read option probabilities in a single forward pass — the interface TypeSafe's Jev demonstrated.
- **Local execution.** Building plans and occupant data rarely leave the premises. It has to run on one GPU without an API.
- **Separated from the physics.** Walking, pushing and smoke intoxication have validated equations; neither training nor an LLM should touch them.

The work therefore has two parts: reproducing the Jev interface on open models, then putting that engine on top of fire physics to make an evacuation model.

## 2. Part 1 — Rebuilding a Jev-style decision engine on local LLMs

### 2.1 Readout principle

A chat model receives a prompt of this shape:

```
system : "Apply the rules to the situation. Choose exactly one option. Answer with its uppercase letter only." + rule paragraph
user   : {"situation": {...}, "question": "...", "options": [{"letter":"A","description":"..."}, ...]}
```

The model is **not allowed to generate**. From the next-token logits $z \in \mathbb{R}^{|V|}$ at the answer position, only the option-letter tokens are taken and normalised:

$$p_k = \frac{\exp z_{\ell_k}}{\sum_{j} \exp z_{\ell_j}}, \qquad \ell_k = \text{token}(\text{letter}_k)$$

- Zero generated tokens, no parsing failures, one forward pass per decision.
- Each letter must be a **single token** in the tokenizer (verified for A–P).
- On a llama.cpp server the same readout is obtained with `max_tokens=1`, `logprobs=true, top_logprobs=20`; the transformers backend reads the last-position logits directly.

This is the readout [openjev](https://github.com/TheoLeeCJ/openjev) demonstrated on Qwen3.5-4B; we ported it to a llama.cpp backend and to a ternary 27B model.

### 2.2 Question fan-out and execution guards

One person's decision is split into several questions (heads). Following [jev-ultrafast](https://github.com/browser-use/jev-ultrafast), all heads are asked on the same situation at once, and **only the head selected by `mode` is executed**.

| head | options | executed when |
|---|---|---|
| `mode` | evacuate / wait / escort / rescue / follow_crowd / breakthrough / shelter | always |
| `target_exit` | exit 1–8 | mode ∈ {evacuate, breakthrough, follow_crowd, escort} |
| `rescue_target` | victim 1–6 / none | mode = rescue |
| `rescue_feasible` | yes / no / insufficient | rescue check — **no downgrades to evacuate** |
| `pace` | run / walk / slow / stop | always (speed multiplier 1.4 / 1.0 / 0.6 / 0) |
| `survive` | likely / unlikely | probability only |

The guard prevents the contradictory state "I will rescue" + "rescue is infeasible" from ever being executed.

### 2.3 How the evidence is written

Information used for the decision is written **short, first, and as relations** — rules we learned by measurement.

- Hazards go at the front of the sentence: `"DANGER, FIRE AT THIS EXIT: X1, 9 m …"`. Appended at the end, even the 27B chose a 9 m burning exit over a 27 m clear one.
- Each exit carries `usable / route (passes the fire?) / distance / direction / smoke / heat / people_waiting / queue_s`.
- Personal profile: `role, sex, age, mobility, exposure_so_far, can_see_m, knows_building`; `my_child` for guardians; `people_in_danger_nearby` within 15 m.
- Ten lines of rules in the system prompt — avoid flames and zones above 120 °C, prefer a safer distant exit, switch from crowded exits, guardians never leave their child, adults rescue only while their own exposure is low, firefighters move against the flow to the most endangered survivor, breakthrough is a last resort, shelter when everything is blocked, act now.

### 2.4 Which models survive direct readout

| model | readout | authored bench (30 judgments) |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_M | direct | 4/30 |
| Qwen3.8-27B UD-Q4_K_M | letter after thinking | 29/30 |
| **Ternary Bonsai 2 27B PQ2_0** | **direct** | **29/30** |
| Qwen3.5-4B Q8_0 | direct | 21/30 |
| TypeSafe Jev | typed probabilities | 28/30 |

Same 27B base: the 4-bit build collapses under direct readout to letter-position bias (reversing the option order changes the answer), while the ternary build (Bonsai) scores 29/30 with no thinking at all. That ternary representation suits logit readout far better is the single most important measurement of this project. Thinking mode restores accuracy but costs tens of seconds per decision and, in simulation, flattened 100 people into one identical decision.

### 2.5 Backends

| backend | runtime | notes |
|---|---|---|
| `typesafe` | TypeSafe API | all heads in one request; fastest (0.085 s/decision) |
| `llama` | llama.cpp server | Bonsai (PrismML fork required) / Qwen GGUF; direct by default, `--think` for thinking |
| `hf` | transformers | same logit readout as openjev |
| `rule` | no LLM | nearest safe exit + guardian escort + firefighter to nearest victim |

## 3. Part 2 — The TypeEvacSafe evacuation model: principles and equations

```
BULC fire simulation (.fds in → .sf/.smv out)
        ↓  convert_sf2d
2D hazard fields : T, K_s, CO, O2, flame, walls, per-exit distance fields, HRR(t)   ×  z = 0.5 / 1.5 / 2.4 / 2.8 m
        ↓
┌────────────────────────────────┬────────────────────────────────────────┐
│ Decision layer — one per person │ Physics layer — shared                  │
│ mode · target_exit · rescue ·   │ social force · radiation · smoke · hazard │
│ pace · survive                  │ FED · slowdown · visibility · death rule  │
└────────────────────────────────┴────────────────────────────────────────┘
        ↓
0.5 s trajectories (CSV) · plots · MP4 · behavioural metrics
```

### 3.1 Fire fields — converting BULC output

Fires are simulated with [BULC](https://bulc.msimul.com), a GPU fire simulator that reads FDS-compatible `.fds` input. Instead of 3D volumes we store **four 2D slices** (walking 0.5 / breathing 1.5 / under-ceiling 2.4 / ceiling 2.8 m) × 9 channels (U V W P T ρ O₂ soot HRRPUV) at 1 s, keeping a case under 100 MB. The converter derives:

- extinction coefficient $K_s = 8700\,\rho_{soot}$ [1/m] (FDS default mass extinction coefficient 8,700 m²/kg)
- CO in ppm $= \dfrac{\rho_{soot}\,(Y_{CO}/Y_{soot})}{\rho}\cdot\dfrac{M_{air}}{M_{CO}}\cdot 10^6$, using the yields in the `.fds`
- wall mask: OBST/HOLE rasterised at z = 0.5 m (windows stay walls)
- exits: at 2.25 m (above door height) the region connected to the domain boundary is "outside"; interior cells adjacent to it are clustered into exits; per-exit 8-neighbour Dijkstra distance fields $D_k(x)$
- heat release rate $Q(t)$ from BULC's `_hrr.csv`

### 3.2 Locomotion — social force (ported from rust_evac's BR model)

Acceleration of agent $i$:

$$\dot{\mathbf v}_i=\frac{\mathbf e_i\,v_i^{*}-\mathbf v_i}{\tau}+\frac{1}{m}\Big(\sum_{j}\mathbf f_{ij}+\sum_{w}\mathbf f_{iw}\Big)+\mathbf f^{rad}_i+\mathbf f^{haz}_i,\qquad \tau=0.5\ \mathrm{s}$$

- Desired direction $\mathbf e_i$ is the 8-neighbour downhill direction of the target exit's distance field, rotated by the visibility error (§3.4); desired speed $v_i^{*}=v_{0,i}\cdot m_{pace}\cdot s(K_s)$.
- Repulsion (Helbing form, rust_evac coefficients): $\mathbf f_{ij}=\big[A e^{(r_{ij}-d)/B}+k\,(r_{ij}-d)^{+}\big]\mathbf n_{ij}+\kappa\,(r_{ij}-d)^{+}\,\Delta v^{t}_{ji}\,\mathbf t_{ij}$, $A=2000,\ B=0.08,\ k=1.2\times10^{5},\ \kappa=2.4\times10^{5}$, cutoff 2.5 m. Walls contribute one nearest cell per axis direction (face normals only).
- Sub-stepping $8\times0.0125$ s = 0.1 s; axis-separated sliding on wall contact; exit lines are tested every sub-step.

### 3.3 What the fire does to the body

**Radiation force** — only with line of sight to the fire within 12 m:

$$q=\frac{\chi_r\,Q(t)}{4\pi r^{2}},\qquad |\mathbf f^{rad}|=s_r\min\!\Big(\frac{(q-q_{th})^{+}}{q_{ref}},3\Big)\big(0.4+0.6\,\max(0,\ \hat{\mathbf o}\cdot(-\hat{\mathbf n}))\big)$$

$\chi_r=0.3,\ q_{th}=2.5$ kW/m² (7 for firefighters), $q_{ref}=5,\ s_r=4$.

**Hazard-field push** — line-of-sight-limited finite difference at perception distance $d_p=3$ m:

$$H=\frac{(T-40)^{+}}{40}+\frac{K_s}{1},\qquad \mathbf f^{haz}=-d_p\nabla H\cdot\frac{v^{*}}{\tau}$$

The **sum** of radiation and hazard push is capped at $0.9\,v^{*}/\tau$: it must stay below the desired term (1.0) so that breaking through (`pace=run`, ×1.4) remains possible; capping each separately gives 1.8 and permanent stalls beside the fire (measured).

**Slowdown (Frantzich–Nilsson)**: $s(K_s)=\max\big(0.15,\ 1-a\,\bar K_s\big)$, $a=0.081$ (elderly 0.12, injured 0.10), $\bar K_s$ an exponential moving average with a 2 s time constant.

**FED (Purser)**:

$$\dot F_{tox}=\Big[2.764\times10^{-5}\,CO^{1.036}\cdot\frac{e^{0.193\,CO_2+2.0004}}{7.1}+\frac{1}{e^{\,8.13-0.54(20.9-O_2)}}\Big]\ \mathrm{min^{-1}},\qquad \dot F_{heat}=\frac{1}{5\times10^{7}\,T^{-3.4}}\ \mathrm{min^{-1}}$$

$$F=\int\big(k_{tox}\dot F_{tox}+k_{heat}\dot F_{heat}\big)\,\frac{dt}{60}$$

Role factors $k_{tox}$: child 1.3, firefighter 0 (SCBA); $k_{heat}$: elderly 1.2, firefighter 0.2 (turnout gear).

### 3.4 Visibility and wayfinding

Same definition as BULC's VISIBILITY, $V=3/K_s$ (reflective signs). As visibility drops, the perceived exit direction becomes wrong:

$$\sigma(V)=\frac{\pi}{2}\,\mathrm{clip}\!\Big(\frac{10-V}{10-1},0,1\Big),\qquad \theta_t=a\,\theta_{t-1}+\sqrt{1-a^{2}}\,\sigma\,\varepsilon_t,\quad a=e^{-\Delta t/3\,\mathrm{s}}$$

The desired direction is rotated by $\theta_t$: no error above 10 m visibility, σ = 90° at 1 m. Firefighters get 0.3× (thermal imaging, training).

### 3.5 Collapse, death, rescue

- **Collapse**: $F\ge1$ (ASET exceeded). The person stays in place as an obstacle.
- **Death**: after collapse, **≥ 10 s of exposure** to heat ($T>40$ °C), smoke ($K_s>0.2$) or rising FED, or $F\ge2$.
- **Rescue**: a rescuer reaching within 1 m of a victim (F ≥ 0.3, collapsed, or stuck in smoke for 5 s) secures them → they move together (assist 0.6 / carry 0.5 m/s) → rescue completes at the exit. Adults cannot secure anyone while their own F > 0.5. If the rescuer collapses, both go down.
- **Family**: a guardian's desired speed is limited to the child's while escorting; the child follows the guardian (or the guardian's exit distance field when out of sight). Firefighters enter through the main exit at t = 60–120 s.

### 3.6 Exit summary handed to the decision layer

| quantity | definition |
|---|---|
| safety score | $s_e=\exp\!\big(-\tfrac{(T_e-40)^{+}}{60}\big)\,\exp(-K_{s,e})\cdot\mathbb 1[\text{no flame within 2 m}]$ |
| queue time | $q_e=\dfrac{n_e}{1.3\,w_e}$ [s], $n_e$ = people within 3 m, 1.3 persons/(m·s) (SFPE effective flow) |
| route-through-fire test | with geodesic distance $F(x)$ to the 2 m fire zone and the exit distance $D^{F}_k$ from that zone: the route passes the fire if $F(x)+D^F_k-D_k(x)<3$ m |
| rule-macro cost | $c_k=\dfrac{D_k}{\max(s_e,0.05)}+1.2\,q_e+\mathbb 1[\text{via fire}]\min\!\big(\tfrac{Q}{50},60\big)$ |

Decisions are made every 15 s by default (30 s in the hall test) or on events: exit safety drops, route starts passing the fire, visibility < 3 m, dependent separated, victim spotted.

### 3.7 Occupant profiles

| role | $v_0$ (m/s) | radius (m) | $k_{tox}$ | $k_{heat}$ | $q_{th}$ | notes |
|---|---|---|---|---|---|---|
| adult | 1.2 (female ×0.92) | 0.20 | 1.0 | 1.0 | 2.5 | |
| child | 0.8 | 0.15 | 1.3 | 1.0 | 2.5 | follows guardian |
| elderly | 0.7 | 0.25 | 1.0 | 1.2 | 2.5 | slowdown coefficient 0.12 |
| guardian | 1.2 | 0.20 | 1.0 | 1.0 | 2.5 | linked to one child |
| injured | 0.5 / 0 (non-ambulatory) | 0.25 | 1.0 | 1.0 | 2.5 | non-ambulatory start collapsed |
| firefighter | 1.2 | 0.25 | 0 | 0.2 | 7.0 | enters at 60–120 s, 0.3× wayfinding error |

### 3.8 Evaluation — not the evacuation rate

Evacuation rate lands at 0.96–0.99 for every decision layer and discriminates nothing. We measure four axes.

| axis | metrics |
|---|---|
| A situation awareness | decision-change rate and lag after a hazard arrives, route-exposure regret, improving-switch ratio, switching away from a bad exit |
| B route planning | detour ratio, smoke exposure p50/p90 |
| C role fulfilment | guardian–child separation time and co-exit, firefighter penetration depth and time inside, rescue/death of injured and non-ambulatory occupants |
| D autonomy | mode and exit-choice entropy within one case |

## 4. Measurements

### 4.1 100-occupant hall (BULC, 2 MW, 30 × 30 × 3 m, two exits)

| metric | rule macro | Qwen3.8-27B (thinking) | **Bonsai 2 27B (local, direct)** | **TypeSafe Jev** |
|---|---:|---:|---:|---:|
| **per-decision latency** (measured in simulation) | — | ~35 s | **7.2 s** | **0.14 s** |
| heads per decision · per head | — | 5.9 · ~6 s | 5.9 · 1.22 s | one request |
| evacuated | 0.97 | 0.96 | 0.971 | 0.971 |
| died | 0.039 | 0.049 | **0.029** | **0.020** |
| non-ambulatory rescued (of 5) | 2 | 1 | 2 | **5** |
| injured evacuated | 0.60 | 0.60 | 0.70 | 0.70 |
| decision mix | evacuate only | evacuate 24 | **evacuate 39 · escort 12 · rescue 7** | rescue 25 · evacuate 16 · escort 1 |
| mode entropy (individuality) | 0.21 | 0.05 | **0.50** | 0.42 |
| exit-choice entropy | 0.16 | 0.11 | 0.34 | **0.39** |
| switched from a bad exit to a better one | — | — | **0.54** | 0.00 |
| firefighter penetration | 4.2 m | 5.8 m | **6.4 m** | 0.0 m |
| RSET p50 / p90 | 26 / 44 s | 26 / 40 s | 26 / 97 s | 27 / 76 s |

- Jev is the most rescue-oriented — all five non-ambulatory occupants carried out, lowest death rate, and fastest because all heads travel in one request.
- Bonsai comes close, locally — 7.2 GB, no API, the most individualised decisions (entropy 0.50) and the highest rate of switching away from a hazardous exit (0.54).
- Uniform decisions cost lives — Qwen 27B in thinking mode told everyone to `evacuate` and rescued one.

**Why the per-decision times in 4.1 and 4.2 differ** — same backends, different conditions. In simulation (4.1) a decision asks 5.9 heads (mode · target_exit · rescue_target · rescue_feasible · pace · survive), the evidence is long (exits, victims, family; 1,000–1,600 tokens) and requests ran on 4 concurrent threads (Bonsai: 58 decisions, 417 s of judgment time). The bench (4.2) asks 3.8 heads on ~890 tokens, serially. **Per head, Bonsai takes 1.22 s in simulation vs 0.47 s on the bench**; the gap is explained by evidence length (~1.6×) and the concurrency penalty (~1.8×). Per-decision figures are per-head × heads: 0.47 × 3.8 = 1.78 s, 1.22 × 5.9 = 7.2 s. Jev shows the same pattern, 0.085 s (bench) → 0.136 s (simulation).

### 4.2 Backend accuracy and speed

One A100 80 GB · 20 authored bench cases (30 judgments) · latency over 20 distinct situations queried **serially** (~890 tokens, 3.8 heads per decision). For simulation conditions see table 4.1.

| backend | size | readout | accuracy | per head | **per decision** (bench, 3.8 heads) |
|---|---:|---|---:|---:|---:|
| **TypeSafe Jev** | — | typed probs | 0.93 (28/30) | — | **0.085 s** |
| **Qwen3.5-4B Q8_0** | 4.5 GB | direct | 0.70 (21/30) | 0.128 s | **0.487 s** |
| **Ternary Bonsai 2 27B PQ2_0** | 7.2 GB | direct | **0.97 (29/30)** | 0.467 s | **1.78 s** |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | thinking | 0.97 (29/30) | ~9 s | ~35 s |
| Qwen3.8-27B UD-Q4_K_M | 16.5 GB | direct | 0.13 (4/30) | 0.13 s | 0.5 s |
| Qwen3.5-4B Q8_0 | 4.5 GB | thinking | 0.57 (17/30) | 2.4 s | 9.5 s |

Measurement pitfalls we hit:
1. Never benchmark by repeating one prompt — llama.cpp `cache_prompt` skips prompt processing and inflates numbers 3–5×.
2. Concurrency hurts on a single GPU — each request evicts ~640 MiB of KV cache and breaks the shared system-prompt prefix; 8 threads ran 1.8× slower than serial.
3. Thinking mode is not just slow, it flattens decisions — default is direct (`think=False`).

## 5. Install and run

```bash
git clone https://github.com/using76/TypeEvacSafe && cd TypeEvacSafe
pip install -r requirements.txt          # torch, numpy, matplotlib
```

**(a) Local — Ternary Bonsai 2 27B (recommended)**

```bash
# Requires the PrismML fork; stock llama.cpp loads the file and silently emits garbage
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp llama_prism && cd llama_prism
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80   # A100=80, Ada=89, Hopper=90
cmake --build build -j 20 --target llama-server && cd ..

bash scripts/get_models.sh bonsai        # Ternary-Bonsai-2-27B-PQ2_0.gguf (7.2 GB) → models/
LLAMA_SERVER=llama_prism/build/bin/llama-server bash scripts/serve_bonsai.sh 0 8083 models/Ternary-Bonsai-2-27B-PQ2_0.gguf
python typeevac/jev_bench.py --backend llama --url http://127.0.0.1:8083
```

**(b) Small local — Qwen3.5-4B**

```bash
bash scripts/get_models.sh qwen4b        # Qwen3.5-4B-Q8_0.gguf (4.5 GB)
bash scripts/serve_llm.sh 0 99 8082 models/Qwen3.5-4B-Q8_0.gguf
```

**(c) TypeSafe Jev API** — `echo -n "apikey_..." > ~/.typesafe_key` (chmod 600; never commit it)

**(d) Rule macro** — baseline without any LLM

**Fire case → hazard fields → simulation**

```bash
python typeevac/gen_bigspace_v2.py cases/           # 438 parametric large spaces
python typeevac/gen_judgment.py    cases/           # judgment set J1–J4 (shortest path is the wrong answer)
python typeevac/prep_cad.py your_building.fds cases/MY --fire near_exit --ceiling   # real floor plans
python typeevac/gen_queue.py --gpu 0 --shard 0/1 --only J1_,J2_      # BULC run + field conversion

python typeevac/hall_test.py --case HALL_30x30_2MW --macro rule,llama --url http://127.0.0.1:8083 --out runs/hall
python typeevac/jev_eval.py   --run runs/hall --macro rule,llama --field fields/HALL_30x30_2MW.npz
python typeevac/hall_frames.py --out runs/hall --macro llama,typesafe --every 2 --t_min 1 --mp4 compare.mp4 --fps 10
```

## 6. Repository layout

| path | content |
|---|---|
| `typeevac/jev_prompts.py` | rule prompts (EN/KO) and head/option definitions — edit here to change the judgment |
| `typeevac/jev_decide.py` | evidence builder, head fan-out, backends (TypeSafe / llama.cpp / transformers), execution guards |
| `typeevac/jev_bench.py` | 20 authored bench cases |
| `typeevac/env2.py`, `env_v1.py` | physics layer (§3.2–3.7) and rust_evac constants |
| `typeevac/hall_test.py`, `jev_scenario.py` | scenario runners, 0.5 s trajectory CSV |
| `typeevac/jev_eval.py` | behavioural evaluation A–D |
| `typeevac/hall_frames.py` | PNG frames and MP4 with two smoke layers |
| `typeevac/gen_*.py`, `prep_cad.py`, `convert_sf2d.py`, `gen_queue.py` | case generators, BULC run queue, field conversion |
| `typeevac/jev_teacher.py` | teacher-data generator for distillation |
| `models/README.md`, `scripts/` | model file list, server and download scripts |
| `results/` | bench JSON, 100-occupant video, frames, evaluation |
| `docs/` | plan, learning strategy, backend comparison |

## 7. Roadmap

1. **Batched heads** — each head is currently a separate request that re-reads the same situation 4–6 times. Batching heads into one request or sharing the state prefill would make local backends 3–4× faster.
2. **Distillation** — Jev/Bonsai decision probabilities as teacher, Qwen3.5-4B LoRA (option-KL). Target 0.93 accuracy at 0.05 s/decision.
3. **Outcome correction** — roll out every candidate at each decision point and reweight by realised outcomes (evacuated, died, rescued).
4. **Calibration** — temperature scaling and ECE for `survive` and `rescue_feasible`.
5. 1,000-occupant throughput, RiMEA/IMO dynamics validation, BULC integration.

## 8. Credits, license, patent

- Readout: [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (MIT) · head fan-out and execution guards: [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT)
- Models: [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0, PrismML) · [Qwen](https://huggingface.co/Qwen) (Apache-2.0)
- Fire simulation: [BULC](https://bulc.msimul.com) (Meteor Simulation) · pedestrian dynamics and FED: rust_evac (Meteor Simulation)

Source is licensed under **PolyForm Noncommercial 1.0.0**: free for research, evaluation and other noncommercial use; commercial use requires a separate agreement. With a patent application pending we avoid Apache-2.0 (explicit patent grant) and MIT (room for an implied one). The license covers copyright only and grants no patent rights (`NOTICE.md`). Commercial enquiries: Meteor Simulation — bulc.msimul.com
