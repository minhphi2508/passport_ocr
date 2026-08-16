# Ground Truth Data

The ground-truth annotations used for evaluation are intentionally excluded
from this repository because they are derived from passport images and may
contain personally identifiable information (PII).

The final evaluation used:

- 117 annotated image samples
- 69 passport identities
- 933 annotated fields out of 936 possible fields

Three surname fields were intentionally left blank because the corresponding
documents did not contain a surname value.

Evaluation code is included in `src/`, but reproducing the reported benchmark
requires access to the private annotation set.
