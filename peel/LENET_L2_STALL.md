# LeNet L2 stall in the saved iteration-20 best member

Inspected `recon/_pop__mergedbest_cnn__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0__s0.pt`
against the cached seed-0, 25-epoch leaky-ReLU teacher. Standard CNN
canonicalization/alignment matches the production metrics. Teacher indices below
are zero-based. This checkpoint contains only the best member, not the committee
or the query buffer.

15 L2 filters have signed cosine >= .99857 against their assigned true filter.
The remaining student filter (original index 14) has cosine -.17411 against the
missing true filter (teacher index 6), and max weight error .37554.
Its best teacher match is instead teacher index 12, cosine .9568. A second
student filter (original index 3) already matches teacher index 12 at cosine
.99857. Thus this member has approximately duplicated one true filter and missed
another, rather than uniformly inaccurate L2 weights. The duplicate has the
smallest outgoing-column block norm (.681; layer range .681 to 3.534), but this
alone does not prove a gradient-starvation mechanism.

With --fast-peel, the CNN layer trigger requires 100% channel consensus. 15/16
therefore does not invoke L2 refinement. The log's solved L1 call cost 1 second;
the subsequent waiting time is population training, not kink solving. L1's
5.96e-8 displayed parameter floor is consistent with FP32 frozen training copies;
this run does not use MLP --peel-direct or its inverse-prefix oracle.

The displayed loss is best-member mean absolute output error over the CURRENT
retained query buffer. It is not L2 parameter error or a fixed validation loss.
Each round adds 20,000 actively generated queries; --window 60 keeps all data
through the shown 20 rounds after refresh. The buffer therefore grows from 100k
to 400k rows, and each epoch trains every committee member on it. This explains
increasing round costs without attributing them to refinement.

The log and best-member checkpoint establish a concrete duplicate/missing-filter
state and the all-channel gate. They do not establish why the optimizer entered
that state or whether all members share the same duplicate. Testing the latter
requires a full-population snapshot. A targeted remedy to test is verified partial
freezing plus rerolling/targeting the remaining channel, not blindly accepting an
incomplete prefix or lowering the recovery accuracy gates.

Per-channel evidence: `lenet_l2_stall.json`.
