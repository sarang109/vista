# VISTA — research artifact

**Privacy-preserving receiver-verifiable security-control attestation for real-time payments.**

This repository is a runnable, end-to-end implementation of the VISTA research blueprint (v1.1, Parts I and II).
It covers the protocol (VSAC), the algorithms (SEN, MDPS, MAN, and foundations N1–N8), a simulated interbank market
with a realistic transaction workload, an attack suite, and experiments E1–E22 plus the baselines and ablations.
Every experiment writes CSV files, and all figures are regenerated from those CSVs.

Everything runs locally on free, open-source libraries. **No cloud account, API key or paid service is needed.**

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                                                      # unit + property + adversarial tests (~2 s)
python -m vista.run_all --config configs/experiments/smoke.json   # everything, reduced scale (~1 min)
python -m vista.run_all --config configs/experiments/full.json    # synthetic stream, 300k payments (~4 min, 8 cores)
```

Run on the public PaySim transaction log (downloads 494 MB once, no account or API key needed; ~20 minutes on 8 cores):

```bash
python -m vista.run_all --config configs/experiments/paysim.json
```

It prints a progress line with the estimated time remaining every minute. The file is checked against the published PaySim
statistics (6,362,620 rows, 8,213 frauds) and refused if they differ. Memory stays small because the data is memory-mapped, workers
process fixed-size chunks, and results are folded into counters and histograms as they arrive.

Where the outputs go:

- `results/<config>/*.csv`: raw results for every experiment.
- `results/<config>/dashboard.html`: one self-contained offline HTML page (data embedded, no network or external fonts). Open it in any browser or send the file as is. A copy is also written to `paper_artifacts/dashboard_<config>.html`.
- `results/<config>/e2e_transactions.csv[.gz]`: one row per payment (gzip-compressed above 1M payments).
- `results/<config>/manifest.json`: config, library versions, hardware, per-step timings and SHA-256 hashes of every CSV.
- `paper_artifacts/figures/<config>/*.png` and `paper_artifacts/tables/<config>/*.csv`: figures and tables for the paper.

Run a single step with `--only e2e,e10` (`data` is the PaySim download and verification step). The step names are `verification, e1, e2, e4, e5, e6, e11–e22, e2e, e10`.

## What is implemented (blueprint section → module)

| Blueprint | Module | Notes |
|---|---|---|
| §1.1 canonical payment and PID | `payment.py` | NFC-normalised canonical JSON with no floats; PID = SHA-256 |
| §3.2 claim schema, N1 chains | `claims.py` | 9 attributes (identity, auth, phishing resistance, credential age, device, dynamic binding, freshness, account age, control execution) |
| §3.3 monotone policy language | `policy.py` | Parser, evaluator, FRONTIER compilation (N1), CNF conversion, random generators |
| §3.4 cost models | `cost.py`, `prior.py` | Additive sensitivity weights and information-theoretic surprisal under a receiver prior |
| §5.1 SEN | `sen.py` | Per-bank auditable JSON mapping profiles; conservative; event-based invalidation (reset after auth) |
| §5.2 MDPS | `mdps_exact.py`, `mdps_greedy.py` | Frontier (Lemma 1.1), read-once DP, 0-1 ILP (HiGHS), brute force, set-cover greedy |
| §5.3 / N6 MAN, MAN-S | `man.py` | Deterministic covering MAN; stochastic MAN-S with exact 3^\|A\| DP, recursive ratio rules, fixed-order and blanket baselines, deficit-based STEP-UP/HOLD/REJECT |
| §4, §5.4 VSAC | `vsac.py` | Build and verify. Checks transaction, audience and nonce binding, freshness (per-claim TTL, capsule horizon = min TTL), replay, schema version, key status. Fails closed. Every check can be switched off for ablations |
| §4.3 crypto backends | `crypto/` | Ed25519 signed predicates; selective disclosure (salted digests with decoy padding); hash-chain threshold proofs; CDS OR-proofs over RFC 3526 MODP-2048 for the branch-oblivious proof |
| Trust registry | `trust.py` | Keys, roles, validity windows, rotation, revocation, status staleness (fail closed) |
| N2 selection leakage | `selection.py` | Exact Λ, ε_sel, G-DIS advantage, information floor, PoD, exponential mechanism, LP-optimal ε-private kernel, Bayes-optimal branch-inference attacker |
| N3 cumulative control | `ledger.py` | Probing attack, simulatable interval LEDGER, OCGD grid DP |
| N4 correlated leakage | `leakage.py` | Surprisal, submodularity/FKG checks, additivity audit, sensitive-target gain |
| N5 fixed-charge MDPS | `group_cost.py` | Exact ILP/enumeration, myopic greedy, density greedy (DG), adversarial family |
| N7 policy extraction | `policy_extraction.py` | Counting bounds, conjunction extraction, minterm-shrinking attack with transversals, budgets |
| N8 commit-and-audit | `audit.py` | Deterrence law, Merkle commitments, hash-chained transparency log, unpredictable audit sampling |
| §7 engines | `sender.py`, `receiver.py` | Cached policy compilation; DET/RAND/BOP/ALL/FIXED selection; negotiation-aware partial disclosure for step-up |
| §10 simulator | `simulator/` | Banks, realistic workload, attacks, deterministic world for parallel workers |

## The workload ("transaction-like real data")

`simulator/workload.py` generates retail instant-payment traffic at scale. The full config produces 300,000 payments from
150,000 payers across 60 institutions over 7 days. The generator models:

- Zipf bank market shares and three capability archetypes (modern, mainstream, legacy; see `configs/banks/archetypes.json`).
- Heavy-tailed payer activity; account ages from a mixture of new and lognormal; KYC levels; device binding and device age;
  how recently credentials were reset; authentication-method preferences per archetype.
- Diurnal and weekly arrival intensity; lognormal amounts that depend on purpose (P2P, bills, merchant, B2B); round-number
  effects; repeat payer→payee relationships; stable interbank routing; amount-tiered receiver policies
  (`configs/policies/policy_library.json`) with liberal, moderate and conservative stances.
- Latent scenario tags: `ato_like` (fresh reset, new device, weak auth, no dynamic linking) and `scam_like`
  (a deceived customer who authenticates strongly). **These tags only characterise decisions. They are not fraud labels,
  and no fraud-detection accuracy is claimed** (blueprint §10).

The generated stream is saved as `results/<config>/workload_transactions.csv`. To use a public dataset as the backbone
instead, see `data/README.md` (PaySim).

## Honest limitations

- All security evidence is synthetic, because no public dataset contains interbank authentication evidence.
  The rates in `archetypes.json` are modelling assumptions. Run sensitivity analyses before generalising.
- The `ato_like` scenario is constructed with a recent credential reset and a new device. Its high rejection rate
  therefore follows partly from that construction. The `scam_like` scenario is the honest counterpoint: a deceived
  customer passes strong controls, which VISTA does not (and cannot) prevent.
- Network latency is sampled from a lognormal distribution, not measured. CPU-path timings (generation, verification,
  policy evaluation, MAN) are real measurements on the machine that ran them.
- The OR-proof uses short-exponent Schnorr keys (256-bit) in a 2048-bit group, a standard assumption. Branch
  soundness relies on the sender's attestor: an internal HSM/TEE that hands out branch keys only for branches it
  has checked. The receiver trusts the attestor exactly as far as it trusts the issuer (blueprint §1.3).
- The receiver's step-up planner uses completion probabilities calibrated to the population (the kind of estimate
  a receiver would build from history). Per-payer capability is hidden from it, so plans can fail. This is intended.
- Items the blueprint tags [D] or [C] are implemented as designs and are **not** claimed as theorems.
  `verification_appendix.csv` records numerical checks, which are not proofs.
