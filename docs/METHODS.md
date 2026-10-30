# Framework sources and study adaptations

| Code ID | Display name | Entry mechanism | Source and principal adaptation |
|---|---|---|---|
| M4r | PM-GP | Residual correction | Cheng et al., https://doi.org/10.1016/j.watres.2025.124201 ; correct daily ln-growth-rate residuals before integration, reset to each observed origin, recalibrate all seven process parameters |
| M3 | NGR | Equation embedding / Inside | Growth–loss structure adapted from Cheng; learned-rate replacement inspired by Mohanty et al., https://doi.org/10.1016/j.algal.2026.104895 ; five-input 5–5–1 network, joint 36 neural + two loss parameters, daily RK4 |
| M1 | T-PGNN | Process regularization / Alongside | Jung et al., https://doi.org/10.1016/j.watres.2026.126716 ; seven-day recursive encoder–decoder, ln-rate physical residual, Min–Max preprocessing, eleven configuration settings |
| M2 | HAPP | Potential-based mapping | Xie et al., https://doi.org/10.1016/j.watres.2025.124419 ; environmental equilibrium potential mapped at a fixed seven-day lag to log10 density; no observed algal state at issuance |

Jung's formal journal issue is 2027, with an online DOI dated 2026. HAPP source checking used the available manuscript draft and formal metadata/abstract; final typeset full text was unavailable. These are task adaptations, not unchanged reproductions of the source studies.

The frozen search_space_v1.json filename is retained to preserve its original checksum; it includes the V3 locked configuration. HAPP uses up to 200 optimisation evaluations; NGR PSO uses 30 particles, at most 500 iterations and 20 stagnant iterations; T-PGNN up to 100 TPE trials and 300 epochs; PM-GP selects its residual history from 14/30/60 days. Formal seeds are 0–4 where applicable. Numerical details live in the frozen code; wrapper code does not alter them.

Internal support for earlier datasets remains where the frozen imports/hash code requires it. The public entry always selects V3; old dataset branches are not additional experiments supplied by this release.

Future meteorology is known reanalysis. This comparison does not demonstrate operational forecast benefits or isolate the causal effect of coupling position from architecture, state inputs and optimisation differences.
