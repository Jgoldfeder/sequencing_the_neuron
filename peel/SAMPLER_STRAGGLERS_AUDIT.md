# Disagreement sampler audit

Production code unchanged. Reproduce from code/: `python peel/audit_sampler_stragglers.py`
Uses GPU 1; three seeds, 2,048 independent queries per seed. Objective multiplied
by 2048/48000 to retain the original 48,000-query Adam gradient scale (including
Adam epsilon), 30 production query steps, uniform [-1,1] initialization.
Raw results: `sampler_stragglers_audit.json`.

## Critical checkpoint limitation

The cheatpeel1 population checkpoint has been overwritten: iteration 40, no
frozen bank, only 28 near-exact teacher rows in each of its first two members.
It is NOT the original 197-solved population. The wide checkpoint is snapB
iteration 5 before peel, also not a matched-stage experimental control.
The original failure snapshots retain the consensus student, but not its full
committee or query history. Thus this audit cannot establish historical causality.

## Measurements

For teacher rows 122,139,199 (the original three missing rows), in the available
200-input committee:

| Query distribution | Positive-side fractions | Fraction within normalized distance .1 of kink |
|---|---|---|
| Uniform [-1,1] | .147, .256, .188 | .083, .106, .090 |
| Production disagreement optimized | .521, .298, .375 | .076, .050, .072 |
| Uniform [-2,2] | .303, .367, .329 | .058, .061, .066 |

Disagreement increases .181 -> .693. Query coordinate RMS .578 -> 1.001.
This does not show missing neurons becoming inactive. Near-boundary coverage
falls, particularly for row139, but remains substantial. In the wide committee,
near-boundary coverage also falls (same numeric row IDs are unrelated neurons):
.145,.133,.135 -> .078,.035,.075. No demonstrated narrow-only coverage collapse.

## Code-level findings

`method.gen_queries` maximizes pairwise distances between normalized output
vectors. It has no teacher-error signal, no per-neuron coverage objective, and
no independent random-query mixture in the ordinary path. Consequently a common
committee error need not attract disagreement queries. This is a possible blind
spot, not proof that it caused this run's missing rows.

A causal follow-up needs a checkpoint at the actual pre-stall stage containing
population, optimizer state, replay buffer, pin masks, and RNG state. Branch the
same state into production sampling versus a random/active mixture, maintaining
equal teacher-query and training-update budgets, and score missing-neuron
recovery post hoc. Current artifacts do not support claiming this experiment
has been performed, or claiming the sampler explains the wide/narrow difference.
