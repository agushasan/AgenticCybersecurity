# Agentic AI for Cybersecurity of Autonomous Vehicles With Policy-Independent Safety Filtering

Simulation code for the paper

> A. I. Hasan, "Agentic AI for Cybersecurity of Autonomous Vehicles With
> Policy-Independent Safety Filtering," submitted to *IEEE Transactions on Intelligent Vehicles*, 2026.

The code shows how an adversary who spoofs navigation sensors (GNSS, radar, lane camera,
compasses) can steer an autonomous vehicle off its path. It then tests a layered defense
against those attacks:

- **SMT-based secure state estimation.** A multi-modal Luenberger observer plus a
  pseudo-Boolean, conflict-driven search that finds which sensors are attacked.
- **A runtime-assurance (Simplex) safety filter.** It keeps the vehicle inside a hard
  lateral envelope for *any* supervisory agent policy, under the conditions of Theorem 1 of the paper.
- **A reachable-set bound for Theorem 1.** It computes, with zonotopes, the set of states
  the backup controller can reach. The reported numbers use covariance-based error widths, so
  they are provisional calculations, not verified certificates (see the paper's Table I).

The same code runs two case studies with only parameters changed:

| Case study | Vehicle | Speed | Safe bound | Computed excursion bound (provisional) |
|---|---|---|---|---|
| A | Autonomous surface vessel, straight and curving fairway | 6 m/s | 70 m | 48.7 m |
| B | Automated car, highway lane keeping | 20 m/s | 0.8 m | 0.43 m |

## Quick start

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib only

# run every command from the repository root
python3 run_experiments.py             # vessel scenarios A-E, M, Monte Carlo     (~2 min)
python3 run_masquerade.py              # vessel scenario M2, maneuver masquerade  (~1 min)
python3 run_m1_rta.py                  # vessel runtime-assurance evaluation      (~2 min)
python3 src/certify_rta.py             # vessel certificate for Theorem 1         (~5 s)
python3 certify/search_v2.py           # backup-gain search for the certificate        (~15 min)
python3 certify/stress_cert.py         # verified Theorem 1 filter, 350 adversarial runs (~60 min)
python3 certify/verify_indep.py        # independent vertex check of the certificate (~1 min)
python3 certify/sensitivity_v2.py      # certified bound vs noise bound and camera accuracy
python3 certify/vessel.py              # vessel certificate search (none found)
python3 guardian_kit/report_v2.py      # Table II categories, intervals, replay threshold sweep
python3 guardian_kit/sens_replay.py    # replay robustness to noise and tyre-stiffness mismatch
python3 run_car.py                     # car case study B1-B3 and Fig. 3                (~3 min)
```

All runs are seeded, so they reproduce the numbers in `results/` exactly.
Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## How the scripts map to the paper

| Paper item | Script | Output |
|---|---|---|
| Fig. 1, agentic architecture | TikZ in `paper/paper.tex` | vector figure in the PDF |
| Sec. III, security margins (s_max, lambda_2s) | `src/sparse_observability.py` | `results/security_analysis.json` |
| Sec. VI, vessel scenarios A-E, M and Fig. 2 (M2) | `run_experiments.py`, `run_masquerade.py` | `results/metrics.json`, `results/masquerade.json` |
| Sec. VI-C, vessel emergency monitor | `run_m1_rta.py` | `results/m1_rta.json` |
| Sec. V-C and Table I, verified certificate (car) | `src/certified_filter.py`, `certify/search_v2.py` | `results/certificate_car.json` (gains, error sets, polytope, margins, timing) |
| Certified bound sensitivity | `certify/sensitivity_v2.py` | `results/cert_sensitivity_v2.json` |
| Vessel certificate search | `certify/vessel.py` | printed summary |
| Sec. VII, car scenarios and Fig. 3a-c | `run_car.py` | `results/car.json`, `figures/fig18_car.png` |
| Verified-filter campaign and Fig. 3d | `certify/stress_cert.py` | `results/cert_stress.json`, `results/cert_traces_a.npz`, `results/cert_traces_b.npz` |
| Table II and replay statistics | `guardian_kit/report_v2.py`, `guardian_kit/sens_replay.py` | `guardian_kit/data/evaluator/report_v2.json`, `sens_replay.json` |
| Adversarial interface messages | `guardian_kit/stress_test.py` | `guardian_kit/data/evaluator/stress_test.json` |

Some figures in `figures/` (for example `fig01`, `fig15` and `fig16`) are extra
diagnostics that are not printed in the paper.

## Repository layout

```
run_experiments.py         vessel scenarios A-E, M and Monte Carlo
run_masquerade.py          vessel scenario M2 (spoof shaped to a legitimate turn)
run_m1_rta.py              vessel runtime-assurance filter evaluation
run_car.py                 car case study (B1 fast spoof, B2 all lateral anchors, B3 slow spoof)
src/ship_model.py          linear error-state ship model and 12-channel bridge sensor suite
src/vehicle_model.py       single-track car model, automotive sensor suite, LQR lane keeping
src/sparse_observability.py  brute-force s-sparse observability, s_max and delta_2s
src/secure_estimation.py   MML observer, detection envelope, SMT attack isolation, Kalman baseline
src/attacks.py             attack models (walk-off, bias, ramp, campaigns, coordinated, masquerade)
src/route.py               curving fairway and line-of-sight reference
src/simulate.py            closed-loop simulator (ideal, naive and secure configurations)
src/safety_filter.py       runtime-assurance filter (predictive backup, observability trigger,
                           dead-reckoning cross-check, fail-safe)
src/certify_rta.py         covariance-based reachability calculation (superseded by certified_filter.py)
src/certified_filter.py    exact Theorem 1 filter: hypothesis-selected observer bank (GNSS-free, camera-free),
                           robust error sets, validity region, maximal RPI set, predicate, Lemma 1 bound
certify/                   design search, certifiable-bound map, vessel check, stress campaign,
                           figure traces and an independent vertex-based verification
src/analysis.py            metrics and vessel figures
src/agentic_figs.py        earlier PNG drafts of the architecture diagrams (superseded by TikZ)
src/style.py               shared plotting style
results/                   JSON outputs of all runs
figures/                   all generated figures
guardian_kit/              supervisory benchmark, tools, baselines, stress tests and open-weight model runner
paper/                     LaTeX source of the paper (IEEEtran), Figs. 1 and 2 drawn in TikZ inside paper.tex
```

Each module in `src/` also runs on its own as a self-test, for example
`python3 src/secure_estimation.py` checks that the MML observer reproduces the explicit
observer bank.

## Building the paper

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

## Citation

```bibtex
@article{hasan2026agentic,
  author  = {Hasan, Agus Ismail},
  title   = {Agentic {AI} for Cybersecurity of Autonomous Vehicles With Policy-Independent Safety Filtering},
  journal = {IEEE Transactions on Intelligent Vehicles},
  year    = {2026},
  note    = {Submitted}
}
```

