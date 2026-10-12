# Direct-tail versus wide-input recovery: log and snapshot audit

Compared attachments `bf0ff55a-5035-4e15-bc49-bc968895796f` (cheat-peel)
and `f1b96bd6-bbd7-4b11-ab90-fd6fca5a65eb` (wide input).
The cheat teacher is an ordinary directly queried 200→200→200→100 network;
there is no recovered prefix or inverse involved in its first-layer recovery.

| Stage | Direct 200-input network | Original 3072-input network |
|---|---|---|
| Iteration 5, before first freeze | 118 consensus; 114 subsequently recovered | 0 consensus |
| Iteration 10 | 194 recovered after partial pass; 196 banked after full pass | 200 recovered |
| Iteration 15 | 197 recovered; three remaining poor guesses | first layer already complete |

Thus the smaller network acquires most first-layer neurons earlier, but takes
longer to finish its last few. It is not uniformly slower. The saved first
failure (iteration 5) precedes any pinning: channel 168 is 68.8 degrees from its
nearest teacher augmented hyperplane. Freezing cannot be the sole origin of
poor guesses, though the logs do not exclude effects on later optimization.

## Actual remaining guesses

From `layer0_1790824122481015071.pt`, after iteration 15. Angles are minimum
sign-invariant angles of unit augmented [weight,bias] vectors against all teacher
rows, independent of neuron ordering:

| Guess channel | Nearest teacher row | Angle | Collected points |
|---|---:|---:|---:|
| 14 | 122 | 70.97° | 55 |
| 153 | 122 | 51.50° | 35 |
| 197 | 139 | 52.03° | 359 |

197 distinct teacher hyperplanes have near-exact matches. Missing teacher rows:
122, 139, 199. The poor guesses are not established exact duplicates: two having
the same *nearest* teacher row is not evidence they recovered that row.
Raw angle audit is in `cheat_peel_stragglers.json`.

The c197 rejection reports gap 4.94e-14, uncertainty 1.80e-13, validation .98.
Those pass the numerical gates; inspection of the acceptance logic identifies
the guess-angle gate as the remaining rejection (12° default). A good numerical
plane need not be the intended neuron, so relaxing that check in production
without identity validation would risk duplicates.

## Measured differences between the two learned matrices

From the cached original teacher, after normalizing each weight row:

| Matrix | Condition number | Smallest singular value | Largest absolute normalized bias |
|---|---:|---:|---:|
| 3072→200 | 5.48 | .458 | .196 |
| 200→200 (cheat input layer) | 480.1 | .00458 | .717 |

These are concrete conditioning differences, not proof of the cause of Adam's
stragglers. Both matrices are full row rank. The missing neurons' positive-side
fractions on 20,000 uniform [-1,1] direct inputs were .138, .256 and .190: somewhat
imbalanced, not dead. This is a diagnostic distribution, not the actual optimized
query history, which was not saved in these failure snapshots.

The refiner dispatch also differs: the wide-input case uses affine sweeps;
the 200-input case uses informative point sampling. That affects polishing cost,
not the before-refinement consensus counts.

## What is established and what is not

Established: a few large-angle training failures, and repeated local refinement
of those failures, explain the observed long tail. The first bad candidate
already exists before partial freezing. Stored recovered rows remain accurate.

Not established: why Adam selects those poor solutions, whether freezing makes
escape harder, or whether changing input conditioning removes the stragglers.
The supplied logs and failure snapshots do not contain matched population
trajectories or a pinning-off control. No claim of a proven optimizer cause or
of full convergence is warranted. Production code is unchanged by this audit.

## Controlled c197 recovery check

Replayed c197 from the same saved student/teacher fixture, changing only the
acceptance angle to 180° for this diagnostic (fresh deterministic sampling,
160-round cap). It recovered missing teacher row 139 with maximum augmented
coordinate error 2.33e-14, 949,760 teacher input evaluations, 14.8 seconds on CPU.
Teacher weights were used only for post-hoc identification/scoring. Production
angle gates were not modified. This confirms at least one straggler can yield
an accurate missing hyperplane despite a very poor initial guess; it does not
justify removing identity checks globally.
