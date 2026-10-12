# MLP solved-weight transfer fixes

The supplied normalized-peel run reached 199/200 L2 rows, but after a partial
full-layer freeze the stored L2 error rose from ~8e-10 to ~4.8e-4, training copies
showed ~6e-4, and reported L1 precision dropped from ~4e-11 to ~1e-8.

Confirmed code defects and fixes:

- Partial MLP injection previously treated the source record as canonical even
  when its prefix had arbitrary positive row scales. Transfers now account for
  both source and destination input gauges. The same transfer is used when
  seeding guesses from consensus and when repinning after rebuilds. The existing
  generic-compatible `_cnn_transfer_rows_` helper supplies this conversion; its
  CNN behavior is unchanged.
- Non-cheat full-layer freezing reconstructed earlier solved layers from the
  fp32 consensus. A persistent fp64 refinement record now seeds the operation;
  previous frozen prefix tensors remain bit-for-bit unchanged. Candidate rows
  are converted into that record's input coordinates before refinement.
- Full-layer refinement freezes only recovered rows for design/loc modes,
  rejects duplicate recovery results, and records newly recovered rows in the
  persistent partial bank. Rejected rows remain trainable.
- An incomplete layer in the frozen dictionary remains the frontier. Merely
  existing in that dictionary is no longer treated as completion.
- Partial freezes do not trigger full-layer peelrestart. A full-layer attempt
  with no additional recovered rows does not rebuild the population/reset Adam.
- The misleading 'unrefined rows frozen at consensus' message is removed for
  the design/loc path.

Validation (CPU, all passed):

```bash
OPENBLAS_NUM_THREADS=1 python peel/test_mlp_solved_transfer.py
OPENBLAS_NUM_THREADS=1 python peel/test_mlp_partial_freeze.py
OPENBLAS_NUM_THREADS=1 python peel/test_design_pipeline.py
OPENBLAS_NUM_THREADS=1 python peel/test_normalized_peel_pipeline.py
OPENBLAS_NUM_THREADS=1 python peel/test_normalized_peel_state.py
```

The transfer test uses independently rescaled source and destination prefixes;
canonical recovered directions agree below 1e-12 in fp64 and below 2e-7 after
fp32 training reconstruction. The production non-cheat test exercises complete
L1, incomplete L2, subsequent retries, exact prefix preservation and no-progress
rebuild suppression. The normal and normalized pipelines exercise actual
refinement, freeze, export and incomplete-prefix deferral.

No long target training run was relaunched. The last unconsensed neuron may still
require improved training or retries; these fixes do not guarantee convergence.
No teacher weights are used to decide acceptance. Existing corrupted checkpoints
and running Python processes are not repaired in place. Restart with the same
command from a known-good pre-corruption state, or start fresh.
