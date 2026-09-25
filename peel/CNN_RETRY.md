# CNN tracking rejection retry

The reported L2 rejections have sufficient inliers and acceptable angles, but
fail the unprinted median-residual threshold (1e-5). High inlier counts use an
adaptive threshold and therefore do not imply a precise recovered row.

On the cached trained leaky-ReLU LeNet and saved best-member guesses, a probe
with an aligned exact L1 prefix reproduced all five target rejections. Tracked
fits had median residuals 3e-5 to 5e-5 and maximum coordinate errors 1e-2 to 2e-2.
Independent scan/fingerprint samples recovered those same filters near 2e-14.
This supports replacing the failed sample collection, not relaxing acceptance.

`kink_solve.recover_layer` now retries failed CNN tracking fits with independent
random inputs and positions at scales 1, 4, and 16. It requests 2*Din+100 points
and allows eight sampling rounds. Every fresh point is scanned and fingerprinted;
the existing fit acceptance gate is unchanged. The retry also covers channels
with too few tracked points. Query counts include the retry. Rejection messages
now print the residual and thresholds. `cnn_retry=False` supports comparisons.

Validation from code/:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_conv_retry.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_conv_kink.py 0 1
```

The production-entry-point regression recovered 5/5 target filters, worst
augmented-unit-row error 3.754e-13, with 245,397 total oracle queries. Its baseline
recovered 3/5; sampling/fit outcomes can vary numerically. Random-network checks
recovered L1 6/6 (8.1e-14 worst error) and L2 16/16 (2.0e-12).

The trained regression uses the latest saved best-member snapshot and replaces
L1 with the aligned exact teacher prefix for diagnosis. It is not a replay of
the original committee state or an end-to-end training run. Teacher parameters
are used only for fixture construction and error evaluation; the solver uses
forward queries. These tests do not establish recovery of every deeper layer.

No new CLI flag is needed for the existing --loc-refine command. Restart Python
to load the fix; an already-running training process keeps its loaded code.
