# -*- coding: utf-8 -*-
"""피난 판단층 프롬프트 — openjev / jev-ultrafast 형식(evidence + criterion + typed options, 한 번의 forward 로 옵션 확률 판독).

사용자 지시(2026-09-17): 입력 텍스트는 한 가지 상황 — "불이 났다. 열과 연기를 피해 안전하게 탈출하되 가족을 챙기고 약자를 챙긴다" —
이므로 규칙 프롬프트를 '깎아서' 넣는다. 아래 RULES 가 그 규칙이다. 전부 텍스트 상수라 코드 수정 없이 다듬을 수 있다.

구조(jev-ultrafast 의 questions 와 같다): 한 상태(state)에 여러 질문(head)을 한 번에 던지고, 실행은 선택된 mode 에 맞는 head 만 쓴다.
  operation(mode)  : 지금 무엇을 할 것인가
  target_exit      : 어느 출구로 갈 것인가 (mode 가 evacuate/breakthrough/follow_crowd 일 때만 실행)
  rescue_target    : 누구를 도울 것인가 (mode 가 rescue 일 때만 실행)
  rescue_feasible  : 그 구조가 가능한가 (yes/no/insufficient)
  survive          : 지금 행동하면 살아 나가는가 (likely/unlikely) — Brier 보정 전에는 조건부 점수일 뿐
"""

# ── 규칙(깎아 넣는 프롬프트) — 영어(점수 안정성), 한국어 주석은 사용자 편집용 ───────────────────
RULES_EN = """A building is on fire. Decide for ONE person what they should do in the next few seconds.
Goal: get this person OUT of the building alive, avoiding heat and smoke, and do not leave family or weak people behind.
Rules:
- Heat and smoke kill. Never route through a zone with flames or temperature over 120 C unless it is the only way out and the exposure is short.
- Prefer an exit with no fire and little smoke even if it is farther, when the near exit is dangerous.
- A crowded exit is slow. If another safe exit has a much shorter queue, switch to it. Do not keep switching.
- A guardian must not leave their child behind. Keep the child within reach; wait for or go back for them unless doing so is certain death.
- Elderly and injured people move slowly. A healthy adult next to a collapsed person should help if their own exposure is still low.
- Firefighters enter against the flow, reach the person in the most danger who can still be saved, secure them, and bring them to the nearest safe exit. They do not idle inside.
- Running through flames (breakthrough) is a last resort: only when the fire is still small, the distance is a few metres, and every other exit is blocked or far worse.
- If every route is blocked, shelter in the place with the least smoke and wait; do not walk into smoke you cannot see through.
- Act now. Waiting is only right when the exit is momentarily blocked or the queue is clearing.
Answer with the single best option letter."""

RULES_KO = """건물에 불이 났다. 한 사람에 대해 지금 몇 초 동안 무엇을 할지 정한다.
목표: 열과 연기를 피해 이 사람을 살아서 건물 밖으로 내보내되, 가족과 약자를 두고 가지 않는다.
규칙:
- 열과 연기는 사람을 죽인다. 화염이 있거나 120°C 를 넘는 구역은 유일한 길이고 노출이 짧을 때 말고는 지나지 않는다.
- 가까운 출구가 위험하면 더 멀어도 불이 없고 연기가 적은 출구를 택한다.
- 붐비는 출구는 느리다. 안전한 다른 출구의 대기가 훨씬 짧으면 바꾼다. 계속 바꾸지는 않는다.
- 보호자는 아이를 두고 가지 않는다. 아이가 손닿는 거리에 있게 하고, 확실한 죽음이 아니면 기다리거나 되돌아간다.
- 노인과 부상자는 느리다. 쓰러진 사람 옆의 건강한 성인은 자기 노출이 아직 낮으면 돕는다.
- 소방관은 흐름을 거슬러 들어가 아직 살릴 수 있는 가장 위험한 사람에게 가서 확보하고 가장 가까운 안전한 출구로 데려온다. 안에서 배회하지 않는다.
- 화염 돌파는 최후 수단: 불이 아직 작고, 거리가 몇 m 이며, 다른 출구가 전부 막혔거나 훨씬 나쁠 때만.
- 모든 길이 막혔으면 연기가 가장 적은 곳에 머물러 기다린다. 앞이 안 보이는 연기 속으로 걸어 들어가지 않는다.
- 지금 행동한다. 기다림은 출구가 잠시 막혔거나 대기열이 풀리는 중일 때만 옳다.
가장 좋은 선택지 한 글자로만 답한다."""

SYSTEM = ("Apply the supplied rules to the supplied situation. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")

# ── 질문(head) ────────────────────────────────────────────────────────────────
Q_MODE = "What should this person do right now?"
MODE_OPTIONS = [
    ("evacuate", "Move toward the chosen exit now."),
    ("wait", "Stay put for a moment (exit momentarily blocked, queue clearing, or smoke too thick to move into)."),
    ("escort", "Go to / stay with the dependent child first, then move together to the exit."),
    ("rescue", "Go to the person in danger, secure them, and bring them out."),
    ("follow_crowd", "Follow the flow of other people toward the exit they are using."),
    ("breakthrough", "Run through the fire zone to the exit (last resort)."),
    ("shelter", "All routes blocked: go to the place with the least smoke and wait for rescue."),
]
Q_EXIT = "If this person is going to an exit, which exit should they head for?"
Q_RESCUE = "If this person is going to help someone, whom should they help first?"
Q_FEASIBLE = "Can this person reach the chosen victim and bring them out before conditions become lethal?"
FEASIBLE_OPTIONS = [("yes", "Yes, the rescue is feasible now."), ("no", "No, it is too dangerous or too late."),
                    ("insufficient", "The situation does not give enough information.")]
Q_PACE = "How fast should this person move right now?"
PACE_OPTIONS = [("run", "Run — fastest possible, accepting the risk of falling or colliding."),
                ("walk", "Walk at a normal pace."),
                ("slow", "Move slowly and carefully (thick smoke, poor footing, or helping someone)."),
                ("stop", "Stop moving.")]
Q_SURVIVE = "If this person acts on the best option now, will they get outside alive?"
SURVIVE_OPTIONS = [("likely", "Likely to get outside alive."), ("unlikely", "Unlikely; they will probably be overcome.")]
