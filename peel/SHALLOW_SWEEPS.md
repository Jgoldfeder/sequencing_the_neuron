# Fast refinement for wide-input, one-hidden-layer networks

`--design-refine` now automatically uses `peel/shallow_sweep.py` when the model
has one hidden layer, input dimension >=1024, and hidden width <= input dimension.
Other architectures continue using the existing informative kink-point solver.
The wide-input solver abstains on failed checks; it does not silently fall back
to the memory-heavy point solver.

The saved `recon/mergedbest512__3072x256x100__s0_consensus.pt` was tested against
the cached seed-0 trained teacher. Full 256-neuron refinement through the actual
`_mlp_refine_layer(..., design=True)` entry point took approximately **20 seconds**
on GPU 2 (RTX 3090), with **3,148,800 forward teacher samples**. All 256 neurons
were accepted; maximum normalized hidden weight/bias error was **4.23e-13**.
After fitting the output layer on 2,048 fresh queries, maximum output error on
512 independent inputs was **2.01e-12**. These are refinement timings from a saved
guess, not full training/consensus-polish timings. Results may vary with guesses.

## Why this is cheaper

For a single hidden layer, every region is affine up to the output. Normalize
only the guessed input rows, then compute one right inverse of the guessed
256-by-3072 weight matrix. Use that right inverse to construct query locations
where the target guessed preactivation is zero and all other guessed
preactivations are +/-2. This creates margin for isolated target crossings.

At each location, sweep input coordinates on the two sides of the target kink.
The difference between the two teacher Jacobians should be rank one; its leading
left singular vector is the target input weight direction. A separate line
intersection supplies the bias. No teacher weights/activations/derivatives are
used by this algorithm; the teacher is only a forward callable.

Two independently placed sweeps, with different step sizes, must agree. Acceptance
also requires a resolved nonzero jump, rank-one residual below 1e-9, direction
within the configured angle gate, and an intersection inside the bracket. These
are heuristic checks, not a universal machine-accuracy certificate. Rank-deficient
guesses and unresolved neurons are left unchanged. A maximum of six placements
bounds the retry cost.

This uses coordinate-query chunks and input-by-output jump matrices. It avoids
both the 6,144 kink points per neuron and the repeated 6,144-by-3,072 SVDs of the
generic design solver. It exploits the single-hidden-layer architecture; the
same isolation argument does not apply to deep networks' downstream boundaries.

## Use

The user's existing command is sufficient; restart Python to load the changes:

```sh
CUDA_VISIBLE_DEVICES=1 python run.py --variant mergedbest512 --arch 3072,256,100 --outer 60 --q 3000 --window 60 --device cuda --combine --fast --design-refine
```

The log identifies the dispatch with:

```
[design-refine] wide-input shallow network: isolated affine sweeps
```

Refinement queries are additional to the training `--q` budget.

For a refinement-only check from the existing saved consensus (no retraining):

```sh
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python peel/test_shallow_dispatch.py
```

Artifacts: `shallow_dispatch.json`, `shallow_dispatch.log`. This test uses the
original unaligned consensus and teacher, invokes production dispatch, checks all
rows recovered, and fits/tests the output layer. `test_shallow_sweep.py` is a
separate aligned-coordinate diagnostic; its full-layer artifact is
`shallow_sweep_full.json` (20.1 s, maximum error 2.13e-13, mean 1.35e-15).

CPU regression: `OPENBLAS_NUM_THREADS=1 python peel/test_shallow_sweep_smoke.py`.
Checks cover callable-only oracle access, teacher query accounting, return gauge,
empty requests, zero budgets, unresolved jumps and rank-deficient abstention.
