# Per-layer recovery report

Synthetic seed-0 network: input 200, seven hidden layers of width 200, output 100. All hidden layers recovered recursively, 200/200 neurons each.

Samples count individual teacher input/output evaluations, including failed scans and verification queries. Model-only candidate evaluations are excluded. Mean error is mean absolute error over every weight and bias in the layer; max error is over the same entries. Hidden rows are compared in the unit-weight gauge.

| Layer (zero-based) | Teacher samples | Wall time (s) | Mean absolute parameter error | Maximum parameter error | Collected kink points |
|---|---:|---:|---:|---:|---:|
| Hidden 0 | 6,815,890 | 46.3 | 4.131e-16 | 2.134e-14 | 84,748 |
| Hidden 1 | 8,621,844 | 62.8 | 1.802e-15 | 4.244e-13 | 84,725 |
| Hidden 2 | 10,228,658 | 108.4 | 3.795e-15 | 3.537e-13 | 87,112 |
| Hidden 3 | 11,773,486 | 78.3 | 8.396e-15 | 1.015e-12 | 88,602 |
| Hidden 4 | 13,616,322 | 105.5 | 3.153e-14 | 3.533e-12 | 88,725 |
| Hidden 5 | 16,599,200 | 98.1 | 1.207e-13 | 1.088e-11 | 88,161 |
| Hidden 6 | 17,912,040 | 106.5 | 3.116e-13 | 2.518e-11 | 88,291 |
| Output 7 (refit) | 4,000 | 0.108 | 8.557e-12 | 3.128e-10 | n/a |

Hidden totals: 85,567,440 teacher samples, 606.0 seconds (10.10 minutes). Output fitting adds 4,000 samples.

Each neuron targeted 400 collected kink points, including held-out validation points; final batches overshoot this count. Counts above range from 424 to 444 points per neuron on average.

Hidden timings are from the original full run; some stages shared the GPU with other benchmark jobs. Means were computed from the saved checkpoint, with all seven maximum errors matching the recorded run exactly.

The original output fit saved only maximum parameter error (3.600e-10) and maximum function gap (5.820e-12 on 2,000 fresh inputs). Its weights and timing were not saved. The output row above is therefore explicitly a new 4,000-sample fit of the same saved hidden layers, with fresh seed-1 inputs. Its timing excludes model loading. Validation samples are separate from recovery sample counts.
