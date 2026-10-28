# Selection-leakage theory: statements, proofs and where each is checked

This note states the results implemented in `pcgs.py`, `repeated.py`, `governor.py` and `trust.py`, gives
their proofs, and points to the numerical checks (`verification_appendix.csv`, ids 26–35) and experiments
(E23–E28). Information is measured in bits unless stated otherwise.

## Setting

* The evidence space is a finite product of chains; `x` is a sender's evidence vector and `π` the receiver's
  prior over it. `Φ` is the receiver's monotone policy and `M = {v_1, …, v_m}` its frontier (minimal satisfying
  profiles, an antichain).
* `π_S = π(· | Φ(X) = 1)` is the prior on satisfiable evidence.
* The **signature** of `x` is `A(x) = {v ∈ M : v ≤ x}` (the branches `x` supports).
* A **selection kernel** `q(v | x)` is a distribution on `A(x)`. Disclosing branch `v` reveals the atomic
  claims `X ≥ v`.
* `P(v) = Σ_x π_S(x) q(v|x)`, `up(v) = π_S(X ≥ v)`, `L_S(v) = −log₂ up(v)`.
* **Selection leakage** (v1 Eq. 3): `Λ(q) = Σ_v P(v) · KL( P(X | V=v) ‖ π_S(X | X ≥ v) )`, which is the information
  revealed beyond what the disclosed claims themselves imply.

## Lemma 1 (decomposition)

`Λ(q) = I(X;V) − E[L_S(V)]`.

*Proof.* For `x ≥ v` with `π_S(x) > 0`:
`P(x|v) / π_S(x | X ≥ v) = [π_S(x) q(v|x) / P(v)] / [π_S(x) / up(v)] = q(v|x) up(v) / P(v)`.
Taking the expectation of `log₂` of this ratio over `(X, V)` gives
`Σ π_S(x) q(v|x) log₂(q(v|x)/P(v)) + Σ_v P(v) log₂ up(v) = I(X;V) − E[L_S(V)]`. ∎

`E[L_S(V)]` is the price of atomic disclosure (PoD); the repository's earlier identity
`Total = E[L] + Λ` is the same statement before conditioning on `Φ = 1`. Check 26 confirms it to `6e−16` bits.

## Lemma 2 (signature sufficiency)

For every kernel `q` there is a kernel `q̄` that depends on `x` only through `A(x)`, has the same expected
cost and the same `E[L_S(V)]`, and has `Λ(q̄) ≤ Λ(q)`.

*Proof.* Let `q̄(v | a) = Σ_{x: A(x)=a} π_S(x) q(v|x) / π_S(A = a)`. The joint law of `(A(X), V)` is the same
under `q` and `q̄`, so `P(v)`, `E[C(V)]` and `E[L_S(V)]` are unchanged. Under `q̄`, `X → A(X) → V` is a Markov
chain and `A` is a function of `X`, so `I_q̄(X;V) = I(A;V)`. Under `q`, data processing gives
`I_q(X;V) ≥ I(A(X);V)`, and this equals the `q̄` value. Lemma 1 then gives `Λ(q̄) ≤ Λ(q)`. ∎

So the optimisation lives on at most `2^m − 1` signatures rather than on the state space. With the nine-attribute
VISTA schema (46,080 states) and two-branch policies this is three signatures. Check 31 confirms it.

## Theorem 1 (no free selection)

(a) Some kernel has `Λ(q) = 0` **if and only if** the linear system
`Σ_{v ∈ a} w_v = 1` for every signature `a` with `π_S(A = a) > 0`, `w ≥ 0`, is feasible. The kernel
`q(v|x) = w_v` then attains `Λ = 0`.

(b) In particular, if there are `v_a ≠ v_b ∈ M` such that the signatures `{v_a}` and `{v_b}` and some signature
containing both have positive mass (e.g. `π_S` puts mass on the states `v_a`, `v_b` and `v_a ∨ v_b`), then
`Λ(q) > 0` for every kernel. Every prior with full support and `m ≥ 2` is covered.

*Proof.* (a) `Λ` is an expectation of KL divergences, so `Λ = 0` iff `P(x|v) = π_S(x | X ≥ v)` for every emitted `v` and
every `x ≥ v` in the support. By the ratio in Lemma 1 this holds iff `q(v|x) = P(v)/up(v) =: w_v` does not depend
on `x` (for non-emitted `v`, `q(v|x) = 0 = w_v`). Normalisation of `q(·|x)` over `A(x)` is the system. Conversely, a
solution `w` defines a kernel with `P(v) = w_v up(v)` and `P(x|v) = π_S(x)/up(v)`, so `Λ = 0`.
(b) The signature `{v_a}` forces `w_a = 1`, `{v_b}` forces `w_b = 1`, and a signature containing both requires
`w_a + w_b + … = 1`, a contradiction. A state equal to `v_a` has signature `{v_a}` because `M` is an antichain. ∎

The only branch-revealing escape is therefore a sparse prior. A branch-oblivious proof (BOP) does not reveal the
branch at all. Theorem 1 does not depend on the prior's values, only on which signatures have positive mass,
and the feasibility test is one small LP (`SignatureSpace.zero_leakage_system`). Checks 27–28 and E23 compare it
with the convex solver's minimum.

## Theorem 2 (optimal selection and the leakage–cost frontier)

For a cost budget `B`, `min_q Λ(q) s.t. E[C(V)] ≤ B` is a convex program. For a multiplier `μ ≥ 0`, the
Lagrangian `Λ_nats(q) + μ E[C(V)]` is a rate–distortion Lagrangian with distortion
`d_μ(v) = μ C(v) + ln up(v)` and the support constraint `v ∈ A(x)`. Its minimiser has the prior-corrected Gibbs
form

`q*(v | a) = r(v) e^{−μ C(v)} / up(v) / Z_μ(a)`, `v ∈ a`, with `r = P` (the induced branch law),

and the alternation `q_t = Gibbs(r_t)`, `r_{t+1} = Σ_a π_S(a) q_t(·|a)` (Blahut–Arimoto) converges to it.
DET is the limit `μ → ∞`. The exponential mechanism is the special case in which `r(v)/up(v)` is constant over
branches.

*Proof.* `I(X;V)` is convex in `q` for a fixed input law. `E[L_S(V)]` and `E[C(V)]` are linear in `q`, and the
constraint set is a product of simplices, so the program is convex. Slater's condition holds whenever
`B > E[C]` under DET, which gives strong duality. Write `F(q) = min_r G(q, r)` with
`G(q, r) = Σ π_S(x) q(v|x) [ln(q(v|x)/r(v)) + d_μ(v)]`, where the inner minimum is at `r = P`. For fixed `r`,
minimising over `q(·|x)` on `A(x)` gives the Gibbs form and the value `−Σ_a π_S(a) ln Z_r(a)`. ∎

**Optimality certificate (implemented).** Let `φ(r) = Σ_a π_S(a) ln Z_r(a)`, which is concave in `r`. With
`c(v) = Σ_a π_S(a) e^{−d_μ(v)} [v ∈ a] / Z_r(a)` we have `Σ_v r(v) c(v) = 1`, and Jensen gives
`φ(r*) − φ(r) ≤ ln Σ_v r*(v) c(v) ≤ ln max_v c(v)`. Hence for `Q = Gibbs(r_t)` and `r_{t+1} = P_Q`:
`F(Q) ≤ −φ(r_t)` and `F* ≥ −φ(r_{t+1}) − ln max_v c_{r_{t+1}}(v)`. Every reported kernel carries this gap. For
the budgeted problem, weak duality gives `Λ*(B) ≥ F*(μ) − μ(B − C_min)`, and `pcgs.Compiled.gap_bits` reports
`Λ(Q) −` that lower bound. The bounds are evaluated in the log domain because `e^{−μC}` underflows for large `μ`.

**Two-branch policies.** For `m = 2` only the signature `{v_1, v_2}` has a choice, so a kernel is a single number.
At equal cost, PCGS, the exponential mechanism and the LP kernel therefore coincide whenever the budget binds.
Gains require three or more branches, or a budget loose enough that the unconstrained optimum is reachable. All
twelve policies in the receiver policy library have `m ≤ 2`.

## Theorem 3 (repeated payments)

A sender pays the same receiver `n` times with unchanged evidence. Define
`Λ_n = I(X; V^n) − E[L_S(J_n)]`, where `J_n` is the join of the branches shown, so `X ≥ J_n` is everything the
disclosed claims imply. `Λ_n ≥ 0` by the argument of Lemma 1, because the posterior is supported on `X ≥ J_n`.

1. With fresh draws and `q(v|x) ≥ q_min > 0` on `A(x)`, after `n ≥ ln(m/δ)/q_min` payments every supported
   branch, and hence `A(x)`, has been shown with probability at least `1 − δ`.
   *Proof:* `P(some v ∈ A(x) unseen) ≤ Σ_{v∈A(x)} (1 − q(v|x))^n ≤ m e^{−n q_min} ≤ δ`. ∎
2. `I(X;V^n) ≤ n I(X;V)`. For kernels that depend on `x` only through `A(x)` and are positive on it,
   `I(X;V^n) → I(X;A(X))` as `n → ∞`.
   *Proof:* `V_1..V_n` are conditionally independent given `X`, so `I(X;V^n) ≤ Σ_t I(X;V_t)`. The count vector is
   sufficient. Its empirical frequencies converge to `q(·|A(X))`, whose support identifies `A(X)`, and
   `I(X;V^n) = I(A;V^n) ≤ H(A)`. ∎
3. With **sticky selection**, `V_t` equals the stored branch while it remains supported and is otherwise redrawn
   from the same kernel with a keyed PRF. With unchanged evidence `V^n = (V_1, …, V_1)`, so `I(X;V^n) = I(X;V_1)`,
   `J_n = V_1`, and `Λ_n = Λ_1`. ∎ The statement is exact, and holds computationally from the receiver's view
   as long as the PRF is pseudorandom.

The sticky key is `(sender account, receiver policy, evidence epoch)`. It is shared by receivers with the
same policy (which helps against colluding receivers), and the epoch changes on reset-type events (credential
reset, device change), so a stored choice never outlives its evidence. `repeated.repeated_metrics` computes
`Λ_n` exactly by enumerating count vectors (Monte Carlo beyond a size limit). Checks 32–34 and E25 cover it.

## Proposition C6 (risk-bounded stale-status grace)

Suppose revocations of an issuer key after the last status snapshot arrive as a Poisson process with rate
`λ_r`. A verifier that accepts a snapshot at most `τ = −ln(1 − ε)/λ_r` old accepts a revoked key with
probability at most `1 − e^{−λ_r τ} = ε` per payment. So the expected number (value) of unsafe accepts is at
most `ε` times the number (value) of payments accepted under grace. *Proof:* memorylessness. A payment verified
at staleness `s ≤ τ` is unsafe iff the key was revoked in `(snapshot, snapshot + s]`, which happens with
probability `1 − e^{−λ_r s} ≤ ε`, and linearity of expectation completes it. ∎ Grace is applied per amount tier
(micro and standard in E28); high-value tiers stay fail-closed. Check 35 and E28 cover it.

## What is not claimed as new

The exponential mechanism, Blahut–Arimoto alternation and its duality bounds, rate–distortion and the
privacy funnel, and memoised randomisation (RAPPOR's permanent randomised response) are prior art. The results
above are the support-constrained, prior-corrected forms that branch disclosure under a monotone receiver
policy requires: the exact zero-leakage test and its two-branch obstruction (Theorem 1), the signature
reduction (Lemma 2), the repeated-disclosure attack and its defence for branch choice (Theorem 3), and their
integration in one disclosure path (`governor.py`).

## Data caveats

The real-signal prior (E26) comes from the RBA login dataset (Wiefling et al., ACM TOPS 2022, CC BY 4.0).
Its released rows were synthesised by its authors from a real single-sign-on login log, preserving global and
per-user categorical feature probabilities. Geolocation, IP and user-agent strings were regenerated, and
timestamps carry added noise. It contains logins, not payments, and covers four of VISTA's claim chains (device
continuity, account age, authentication freshness, network continuity). The other chains stay synthetic and are
labelled that way (`hybrid_*` families).
