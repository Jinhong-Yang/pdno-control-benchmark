# v3 제조 PDE 제어 최종 분석: 제안법의 품질·지연 우월성은 지지되지 않았다

**과학적 판정: 구현된 v3 benchmark에서 NO_GO. 산출물 상태: 최종 분석 및 완전한 논문 초안 작성, 투고 보류.** 실험 완료와 가설 성공을 구분한다. G2 실패를 유지하며, H1의 의미 있는 속도 개선 목표와 구현된 H3 비용 비열등 기준은 충족하지 못했다. H2의 물리 손실에 의한 제어 이점도 관찰되지 않았다. 안전 보장·conditional harm·실시간 지연이 반영된 폐루프 성능은 미확립이다.

이번 작업의 로컬 런타임은 **gpt-6-astra / xhigh**로 실제 `turn_context`에서 확인했다. 이번에 직접 한 일은 동결 자료 해시 검증, 저장된 전체 결과의 CPU 재집계, 구현·통계·선행연구 검토와 보고서·논문 작성이다. 이전 v2 감사가 v3 최종 분석을 대신한 것으로 쓰지 않았다. 과거 실험 실행 모델은 확인되지 않았으며, 이번 모델명으로 소급하지 않는다. 학습·PDE 풀이·새 추론·calibration 생성·test 재실행·CUDA benchmark는 수행하지 않았다.

## 1. 판단에 필요한 핵심 결과

- 독립 test parent는 **1,280개**, 방법·seed별 실행은 **25,600개**, 결과 shard는 **120개**다. seed와 tick을 독립 환경 표본으로 늘려 세지 않는다.
- 선택된 예측 연산자 **24/24**가 validation field nRMSE ≤0.05 gate에 실패했다. P의 범위는 Burgers 0.15826–0.16182, Heat 0.07888–0.08440이다. 낮은 teacher regret이나 Heat tick-100 오류로 이 실패를 대체하지 않는다.
- 사전 validation에서 선정된 비교군은 **B0**다. 명목 P 비용은 B0보다 Burgers **28.04%** 높고 등록 CI는 **[23.55%, 32.87%]**, Heat는 **13.70%**, **[12.93%, 14.49%]**다. 구현된 5% 비열등 한계보다 불리하다.
- P/B4 host-ready p99의 seed별 비율 평균은 Burgers **0.985912 [0.978675, 0.992897]**, Heat **0.992965 [0.984927, 1.001631]**다. 각각 약 1.41%, 0.70%의 차이이며, 목표 25% 감소를 충족하지 못한다.
- P의 pooled p99는 Burgers **6.714 ms**, Heat **10.141 ms**이고, 5-ms 초과율은 **14.226%**, **18.342%**다. 목표 p99 2 ms 및 deadline miss 0.1%에 미달한다.
- Heat의 any-time 위반율은 모든 방법에서 높고 같다. 저장 상태를 분해하면 등록된 위반 전부가 **첫 advanced step의 하한 위반**이며, 상한 위반이나 이후 새 첫 위반은 없다. 이 진단은 endpoint의 판별력 한계를 보여 주며, 초기 구간을 사후 제외하는 근거가 아니다.

근거: `analysis_20260928/audit.json`, `source_audit.json`, 원본 `../evidence/locked_test_analysis_v3.json`, `../evidence/latency_analysis_v3.json`. 주장별 정확한 위치·값·hash는 [claim_evidence.csv](claim_evidence.csv)에 있다.

## 2. 연구 범위와 독립성

연구 대상은 강제력을 받는 periodic Burgers와 Dirichlet Heat로 표현한 **합성 제조 분포정수계 제어**다. 자동차 자율주행 연구가 아니다. 실제 공장, PLC, 카메라 취득, 네트워크 ACK, actuator 또는 HIL 검증도 없다. JevLite는 로컬 구현 이름이며 독점 Jev 시스템의 재현을 뜻하지 않는다.

v3는 parent ID와 RNG에 새 namespace를 반영했다. train/validation/calibration/test는 parent 단위로 분리되며, 보존된 감사는 v1/v2 identity·seed tuple과의 중복이 없음을 보여 준다. Burgers/Heat train은 512/256 parents, validation과 calibration은 PDE별 64 parents다. test는 PDE별 nominal 384, coefficient OOD 128, delay/dropout 128이다. 모든 방법은 같은 parent 초기 상태·물성·외란·관측 스케줄을 사용하지만, 행동이 달라진 이후의 실제 관측 값까지 같지는 않다.

B0/B1은 최신 이용 가능한 센서·이미지를 모드로 복원한다. 학습법은 이력·age도 이용하므로 **같은 원시 정보가 제공되지만 같은 특징을 활용하지 않는다**. 이 한계를 숨기지 않는다. B3 direct surrogate-gradient 정책과 B4 shared-encoder/candidate-batched 비교군을 모두 보존했다. 숨겨진 truth field는 teacher와 오프라인 평가에만 사용하며 배포 policy 입력으로 주지 않는다.

원본 v2는 test 미완료 및 parent 재사용·Heat 좌표/경계 도함수 결함으로 INCONCLUSIVE/HOLD였다. 그 부분 test payload는 이번에 열지 않았다. v3는 관련 source 수정을 한 별도 연구이며, v2와 합쳐 표본 수나 성공 근거를 늘리지 않는다.

## 3. 수치·학습·선택 계약

truth/model grid는 256/128, episode는 200 ticks, tick은 simulation 0.02초, 후보는 10개, prediction horizon은 8 ticks다. 후보는 이전 행동 주위의 3×3 box/slew 조합과 B0 행동으로 구성되며 projection 후 중복될 수 있다. 학습법의 물리 residual은 FP32, eager AdamW를 사용했다. 단일 GPU에서 36개 학습 checkpoint와 2개 observer가 순차 생성되었다.

P는 6개의 이차 action 기저와 rank 32 응답을 공유한다. B4도 관측 encoder를 한 번 계산하고 후보 축을 한 batch로 처리한다. B5는 P와 같은 구조에서 physics/balance만 끄며 ranking은 유지한다. P-no-rank는 ranking을 끈다. B3는 B2에서 시작하여 frozen B4 비용의 gradient로 정책을 학습하고 배포 때 direct policy를 실행한다.

forced 200-tick temporal refinement 근거는 Burgers L2 차이 8.47e-9, 평균 balance 오차 4.46e-15, Heat L2 차이 1.38e-6이다. 47 unit-test 통과와 CUDA 이차미분 smoke는 이전 실행 receipt다. 이번에 다시 실행한 결과가 아니며, 모든 locked trajectory의 연속 PDE 참조 오차를 보장하지 않는다.

실제 operator 선택은 validation nRMSE와 train-IQR-normalized mean teacher regret의 동일 가중 score 최소값을 사용했다. `checkpoint_selection_v3.yaml`의 8-parent closed-loop gate/ranking 절차와 다르다. 최종 spec과 실제 summary 기반 선택을 우선 설명하고, YAML 절차까지 실행했다고 주장하지 않는다. 선택된 P는 모두 update 5,000으로 physics/ranking 이후다. 이전 v2의 pre-physics selection 문제를 그대로 복사하지 않았다. 다만 많은 선택이 최대 학습 시점이므로 최적화 충분성이나 수렴은 미확립이다.

| PDE | Method | seed 11 | seed 23 | seed 37 | nRMSE ≤0.05 |
| --- | --- | --- | --- | --- | --- |
| burgers | P | 0.15826 | 0.15980 | 0.16182 | FAIL (all seeds) |
| burgers | P-no-rank | 0.15846 | 0.15994 | 0.16197 | FAIL (all seeds) |
| burgers | B4 | 0.15883 | 0.15911 | 0.15918 | FAIL (all seeds) |
| burgers | B5 | 0.15826 | 0.15980 | 0.16183 | FAIL (all seeds) |
| heat | P | 0.07888 | 0.08353 | 0.08440 | FAIL (all seeds) |
| heat | P-no-rank | 0.07700 | 0.08245 | 0.08401 | FAIL (all seeds) |
| heat | B4 | 0.08341 | 0.08417 | 0.08306 | FAIL (all seeds) |
| heat | B5 | 0.07888 | 0.08353 | 0.08440 | FAIL (all seeds) |

## 4. 전체 비교군과 제어 결과

표의 비용은 seed를 parent 내부에서 평균한 뒤 parent 평균을 계산한다. 비용은 tick별 tracking MSE + 0.01 action effort + 0.05 slew + 10 state violation penalty의 200-step 합이며 dt를 곱하지 않는다.

| Method | Burgers cost | Burgers RMSE | Cost vs B0 | Heat cost | Heat RMSE | Cost vs B0 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.266473 | 0.032749 | +0.00% | 12.055011 | 0.194435 | +0.00% |
| B1 | 0.339718 | 0.037131 | +27.49% | 13.563123 | 0.222567 | +12.51% |
| B2 | 0.278148 | 0.032973 | +4.38% | 16.883117 | 0.257170 | +40.05% |
| B3 | 0.279906 | 0.033204 | +5.04% | 16.735510 | 0.254743 | +38.83% |
| B4 | 0.340785 | 0.037201 | +27.89% | 13.700599 | 0.224151 | +13.65% |
| B5 | 0.341189 | 0.037221 | +28.04% | 13.706314 | 0.224296 | +13.70% |
| P-no-rank | 0.341181 | 0.037220 | +28.04% | 13.678660 | 0.223901 | +13.47% |
| P | 0.341189 | 0.037221 | +28.04% | 13.706313 | 0.224296 | +13.70% |

| Method | Burgers coefficient shift | Heat coefficient shift | Burgers delay/dropout | Heat delay/dropout |
| --- | --- | --- | --- | --- |
| B0 | +0.00% | +0.00% | +0.00% | +0.00% |
| B1 | +38.01% | +23.62% | +24.91% | +9.13% |
| B2 | +5.68% | +129.43% | +4.15% | +37.32% |
| B3 | +5.74% | +114.16% | +4.12% | +37.08% |
| B4 | +38.20% | +28.22% | +25.01% | +10.18% |
| B5 | +38.23% | +26.96% | +25.03% | +10.45% |
| P-no-rank | +38.95% | +29.61% | +25.04% | +10.39% |
| P | +38.23% | +26.96% | +25.03% | +10.45% |

B0는 모든 test 그룹에서 관측 평균 비용이 가장 낮다. Burgers에서는 direct B2/B3가 P/B4/B5보다 B0에 가깝지만, Heat에서는 direct 정책의 비용이 더 높다. B3가 B2보다 일관되게 낫지 않으며, B1도 B0보다 불리하다. 불리한 비교군을 제외하지 않았다.

등록 aggregator의 정규화는 `(mean J_method − mean J_B0)/mean J_B0`이다. 10,000 paired-parent bootstrap 동안 분모는 관측 B0 평균으로 고정된다. 원 지시서 §14는 parent별 `(J_method−J_B0)/max(J_B0, epsilon_train)`의 평균을 요구했으므로, **서로 다른 estimand**다. 본 판정은 구현된 frozen aggregate 기준이며 원 계약 전체를 준수한 H3 확증이라고 표현하지 않는다. CI는 이미 학습된 3 seed에 조건부이고 방법·조건 전체에 대한 동시구간도 아니다.

## 5. 물리·ranking ablation과 위험 해석

Burgers에서 P와 B5의 저장된 적용 행동과 비용은 모든 parent–seed 비교에서 정확히 같다. Heat에서는 1,920 parent–seed 비교 중 1,916개의 전체 action history가 같다. 이 숫자를 독립 parent 수로 해석하지 않는다. Heat nominal P−B5 비용은 −1.78e-7, CI [−8.61e-6, 6.86e-6]이며, coefficient OOD는 +6.75e-6, CI [0, 2.02e-5]다. delay/dropout은 비용이 같다. 실용적 H2 제어 이득은 지지되지 않는다. 동일한 표본 결과를 모든 설정에서의 동등성 정리로 확대하지 않는다.

선택 시점 로그에서 physics/balance 가중 항은 작지만, 로그 loss 크기는 gradient 영향이나 원인 규명이 아니다. 독립 test PDE residual/balance 평가가 없으므로 물리 기전 성공도 주장하지 않는다. P-no-rank 효과는 조건에 따라 방향이 바뀌어 ranking 이득이 일관되지 않다. no-age ablation은 실행되지 않아 H4는 미평가다.

P의 tick-100 field nRMSE는 Burgers nominal 0.37448, coefficient OOD 1.50808, Heat nominal 0.00808이다. validation은 behavior query의 모든 후보 예측을 pooled ratio로 평가하고, test는 한 on-policy snapshot에서 선택한 action의 parent별 ratio 평균을 쓴다. 값이 작아졌다는 이유로 G2를 통과했다고 쓰지 않는다.

Heat의 nominal/계수 OOD/지연스트레스 위반은 각각 **351/384, 114/128, 115/128**이다. 모든 방법·seed에서 first advanced step의 하한 위반 수와 전체 episode 위반 수가 같고 상한 위반은 없다. 초기 field의 하한 위반 수는 356/384, 115/128, 115/128이다. signed random 초기장과 하한 0의 조합이 binary endpoint를 포화시키는 직접적인 진단이다. 모든 가능한 제어기가 동일했을 것이라는 주장은 하지 않으며, 초기 parent/step을 삭제하거나 경계를 바꾸지도 않는다.

Burgers의 0 violation은 0 위험이 아니다. 384-parent 명목 표의 Wilson upper95는 0.9905%, 128-parent stress는 2.9137%다. 동일 위반 mask의 paired bootstrap [0,0]도 확률적 무위험 보장이 아니다. 전체 실행에 exception/nonfinite fallback과 causality 위반은 보고되지 않았지만, deadline miss를 폐루프 fallback으로 적용한 실험은 아니다.

Calibration 26개 그룹의 margin은 64개 parent score 중 zero-based index 61로 독립 재현했다. **행동 선택·수락·거절에 쓰이지 않는다.** 따라서 conditional harm, 선택 보정의 위험 통제, OOD 안전, uniform trajectory 오차는 모두 미확립이다. repository의 Darcy M=480 finite-family 결과나 가정을 이 제어 연구에 옮기지 않는다. reject-all 또는 미존재 gate의 위험을 0으로 쓰지 않는다.

## 6. 지연과 CUDA 진단

| PDE | Method | Path | p50 ms | p99 ms | >5-ms requests | >5-ms rate |
| --- | --- | --- | --- | --- | --- | --- |
| burgers | B0 | CPU | 0.515 | 0.589 | 0/60,000 | 0.000% |
| burgers | B1 | CPU | 14.400 | 15.544 | 60,000/60,000 | 100.000% |
| burgers | B2 | CUDA | 1.903 | 3.777 | 0/180,000 | 0.000% |
| burgers | B3 | CUDA | 1.911 | 3.840 | 75/180,000 | 0.042% |
| burgers | B4 | CUDA | 3.159 | 6.809 | 28,236/180,000 | 15.687% |
| burgers | B5 | CUDA | 3.117 | 6.727 | 25,928/180,000 | 14.404% |
| burgers | P-no-rank | CUDA | 3.115 | 6.709 | 25,366/180,000 | 14.092% |
| burgers | P | CUDA | 3.115 | 6.714 | 25,607/180,000 | 14.226% |
| heat | B0 | CPU | 0.553 | 0.645 | 0/60,000 | 0.000% |
| heat | B1 | CPU | 1.696 | 5.681 | 937/60,000 | 1.562% |
| heat | B2 | CUDA | 1.963 | 3.886 | 1/180,000 | 0.001% |
| heat | B3 | CUDA | 1.973 | 3.994 | 904/180,000 | 0.502% |
| heat | B4 | CUDA | 4.808 | 10.214 | 38,798/180,000 | 21.554% |
| heat | B5 | CUDA | 4.757 | 10.144 | 35,091/180,000 | 19.495% |
| heat | P-no-rank | CUDA | 4.758 | 10.158 | 33,898/180,000 | 18.832% |
| heat | P | CUDA | 4.755 | 10.141 | 33,015/180,000 | 18.342% |

이 표는 frozen timing raw를 session과 seed에 대해 합친 descriptive quantile이다. H1은 별도로 seed별 p99 ratio 평균을 사용한다. 모델·PDE·seed별 5-ms 초과 횟수/비율과 p50/p95/p99/p99.9는 [latency_per_seed.csv](analysis_20260928/latency_per_seed.csv)에 모두 남겼다. 각 classical row는 60,000, 각 학습 method/PDE pooled row는 180,000 requests다.

실제 측정은 저장된 causal snapshot array→request dict→전송/정책/후보 점수→공통 projection→host action 검사다. 원시 이벤트 결합, 이미지 생성, goal coefficient 생성, PDE truth solver, 실제 센서/네트워크/actuator는 포함하지 않는다. batch=1, 3 session, row당 20,000 requests, warm-up 50회다. 총 elapsed 9,254.482초는 측정 캠페인의 wall time이고 GPU-active 시간이 아니다.

원 지시서의 2,000회 또는 thermal 안정화 warm-up, 핵심 방법의 100,000-request 확장 tail, scheduled/contention replay, latency trace가 적용되는 폐루프 평가가 없다. p99.9는 저장된 descriptive 수치로 보존하되 original extended-tail 요건 충족이나 hard real-time/WCET 보장으로 쓰지 않는다. 제어 비용과 latency를 합쳐 measured real-time closed-loop 이점이라고 말할 수 없다.

제공된 post-freeze CUDA JSON은 `p99`와 `p99.9`의 key collision으로 p99 label 값이 덮어써진 오류가 있었다. 이번 독립 계산도 이를 발견했으며, 원 실행 작업에서 별도 corrected JSON과 erratum을 제공했다. 수정 파일 SHA256은 `f3b5d28f8e02b18143a8f508e9c5001ed4a93a8ef273d5a1cd979154bf214a6d`이고 원본과 raw는 그대로다. 본 표는 registered analysis/raw 및 corrected file을 대조한 값이다. 원 JSON의 p99 ratio/difference는 직접 q=.99로 계산되어 맞았다.

learned event-span p99/host p99 약 97–99%, 약 0.11-ms 차이는 **GPU compute occupancy 또는 compute-bound 병목의 증명은 아니다**. 이벤트 사이에 CPU 준비·kernel launch 공백·동기화가 포함될 수 있고 marginal quantile의 차이는 request별 overhead 분포가 아니다. nvidia-smi 순간 utilization도 profiler가 아니다. torch.compile, CUDA Graphs, B1 vectorization/GPU port 등은 새로운 별도 benchmark 제안일 뿐, 이번 측정 이득이 아니다. 과거 fused AdamW micro-pilot 수치를 v3 inference 이득으로 전용하지 않는다.

## 7. 재현성과 보존

직접 작성한 `audit_frozen_v3.py`는 모델·solver를 import하지 않고 NumPy로 전체 120 test NPZ, 120 latency NPZ를 분석했다. action/state에서 비용·tracking·violation을 재구성했고, 등록 cost/violation bootstrap, H1 seed/session bootstrap, 40 latency group quantile/deadline count, 26 calibration order statistic을 검증했다. **1,694 checks 모두 PASS**, 이 재계산 wall time은 54.796초다. source audit는 추가로 operator 선택과 H3 comparator를 재현했다. 검증 건수는 과학적 정확도 백분율이나 독립 반복실험 수가 아니다.

final raw freeze의 1,072 unique entries를 hash 확인했다. entry 당시 차이는 공개된 RUN_STATE/RUN_LEDGER 후속 행정 업데이트 2개뿐이고, 360개 원자료 및 model/source/data inventory는 일치했다. 더 이른 manifest에 대한 E2E source 불일치는 성공 측정 전 path/shape 수정 이력과 함께 명시했다. frozen 로그·source·checkpoint·raw·manifest를 이번에 고치지 않았다. [REPRODUCE.md](REPRODUCE.md), [failures_and_limitations.md](failures_and_limitations.md), [source_audit.json](source_audit.json)을 참조한다.

| 상태 축 | 최종 상태 |
|---|---|
| data_access | 합성 v3 결과 전체 확인; test 1회 완료; v1/v2 부분 outcome 미접근 |
| physics_contract | 제한된 forced fixture 통과; Heat 초기조건/제약의 endpoint 문제와 continuum 한계 유지 |
| implementation | 필수 방법·seed·paired test·serialized timing 완료; 원 지시서와의 불일치 공개 |
| optimization | G2_FAIL_RETAINED; 충분한 수렴·hyperparameter 최적성 미확립 |
| scientific_effect | 구현된 제안 품질/지연 주장 NO_GO; H2 미지지; 위험/실시간 보장 미확립 |
| reproducibility | 원자료 hash 및 독립 산술 검증 완료; 전체 과거 GPU-active 시간은 미상 |

## 8. 논문과 후속 판단

[manuscript_v3.md](../manuscript_v3.md)는 초록, 관련 연구, 실제 방법·통계식, 모든 필수 비교군 결과, 한계, 참고문헌 및 appendix를 갖춘 영어 초안이다. DeepONet/PINO, differentiable predictive control, CINOC, DiffPhyCon/WDNO 등 제공된 primary source로 연구 위치를 제한했고, 실행하지 않은 외부 방법보다 우월하다는 주장을 넣지 않았다. 정량 주장·표는 [claim_evidence.csv](claim_evidence.csv)와 연결했다.

향후 새로운 실험을 설계한다면 우선 해결할 문제는 Heat 초기조건과 endpoint의 물리적 의미, B0보다 못한 실제 제어 효용, predictor/observer 최적화 충분성, source-defined 통계 estimand의 단일화다. CUDA 가속만으로 이 과학적 한계가 해소되지는 않는다. 그 다음에 동일 projected action과 수치 결과를 검증하는 별도 runtime amendment를 검토할 수 있다. 이번 작업에서 후속 실험은 시작하지 않았다.

저자·소속·교신 정보·기여·funding·declaration은 **AUTHOR_INPUT_REQUIRED**로 남겼다. 보고서와 논문 초안은 완성했으나, 투고 가능한 확증 성공 패키지나 게재 보장은 아니다. 제출·서명·결제·타인 연락·외부 업로드는 하지 않았다.
