# Applied evidence vs attempted participation (540 shards)

Method: invert `mean_ci_width` -> n = alpha+beta per shard; applied_evidence = n - 4.
Validated three ways, all agreeing exactly on the same 158 shards:
`applied_evidence == 0`  <=>  `all agent scores == 0.5`  <=>  `total_ratings == 0`.
540/540 inversions succeeded; none imputed.

Scope: env.reset() re-initialises reputation to the prior, so evidence does not
accumulate across episodes. n / applied_evidence / total_ratings are LAST-eval-episode
quantities; participation_rate averages all 20. Conversion ratios sit at 0.995-1.031
for healthy configs, which validates the comparison.

## Q1 — participation > 0.20 with zero applied evidence

| config | arm | part>0.20 & applied=0 | of part>0.20 | % |
|---|---|---|---|---|
| c1 | A | 0 | 9 | 0.0% |
| c1 | B | 0 | 29 | 0.0% |
| c2 | A | 0 | 22 | 0.0% |
| c2 | B | 0 | 25 | 0.0% |
| c3 | A | 0 | 29 | 0.0% |
| c3 | B | 0 | 28 | 0.0% |
| c4 | A | 0 | 9 | 0.0% |
| c4 | B | 0 | 10 | 0.0% |
| c5 | A | 0 | 5 | 0.0% |
| c5 | B | 0 | 8 | 0.0% |
| c6 | A | 0 | 13 | 0.0% |
| c6 | B | 0 | 25 | 0.0% |
| c7 | A | 0 | 7 | 0.0% |
| c7 | B | 0 | 26 | 0.0% |
| c8 | A | 0 | 4 | 0.0% |
| c8 | B | 0 | 24 | 0.0% |
| c9 | A | 0 | 2 | 0.0% |
| c9 | B | 0 | 26 | 0.0% |
| c10 | A | 0 | 3 | 0.0% |
| c10 | B | 0 | 10 | 0.0% |
| c11 | A | 18 | 26 | 69.2% |
| c11 | B | 4 | 29 | 13.8% |

**Total 22, entirely inside c11** (18 armA, 4 armB). Every other config: 0.

## Q2 — Table 2 participation arm-delta: attempted vs applied

| config | attempted diff | CI95 | sig | applied diff | CI95 | sig |
|---|---|---|---|---|---|---|
| c1 | +0.7447 | [+0.6085, +0.8808] | YES | +0.7434 | [+0.6067, +0.8800] | YES |
| c2 | +0.2619 | [+0.0574, +0.4664] | YES | +0.3162 | [+0.1066, +0.5258] | YES |
| c3 | +0.3148 | [+0.1832, +0.4463] | YES | +0.3048 | [+0.1753, +0.4344] | YES |
| c4 | +0.2244 | [-0.0172, +0.4660] | no | +0.2211 | [-0.0203, +0.4624] | no |
| c5 | +0.3787 | [-0.0201, +0.7775] | no | +0.4706 | [+0.0902, +0.8509] | YES |
| c6 | +0.5012 | [+0.2983, +0.7042] | YES | +0.5871 | [+0.3985, +0.7758] | YES |
| c7 | +0.6966 | [+0.5279, +0.8653] | YES | +0.7374 | [+0.5791, +0.8957] | YES |
| c8 | +0.6364 | [+0.4435, +0.8293] | YES | +0.6692 | [+0.4849, +0.8535] | YES |
| c9 | +0.7926 | [+0.6423, +0.9428] | YES | +0.7903 | [+0.6403, +0.9403] | YES |
| c10 | +0.6772 | [+0.3338, +1.0207] | YES | +0.9518 | [+0.9134, +0.9902] | YES |
| c11 | +0.1817 | [+0.0306, +0.3328] | YES | +0.6483 | [+0.4606, +0.8360] | YES |
| **significant** | | | **9/11** | | | **10/11** |

Pooled attempted: armA 0.3730, armB 0.8793, delta +0.5063 [+0.4444, +0.5682]
Pooled applied:   armA 0.2735, armB 0.8674, delta +0.5938 [+0.5345, +0.6532]

**The 9-of-11 result survives and strengthens to 10-of-11** (c5 becomes significant; c4 remains the sole non-significant config).

## Q3 — attempts or applied ratings?

| config | armA conv | armB conv |
|---|---|---|
| c1 | 1.006 | 1.000 |
| c2 | 0.898 | 0.995 |
| c3 | 1.014 | 0.998 |
| c4 | 1.003 | 0.999 |
| c5 | 0.782 | 1.000 |
| c6 | 0.733 | 0.998 |
| c7 | 0.755 | 1.000 |
| c8 | 0.801 | 1.001 |
| c9 | 1.031 | 1.000 |
| c10 | 0.095 | 0.998 |
| c11 | 0.243 | 0.892 |
| **pooled** | **0.733** | **0.986** |

Arm B converts 98.6% of attempts into applied ratings; arm A only 73.3%.
**Arm B's gain is applied ratings, not attempts** — and the attempted measure
*understates* it, because arm A's attempted figure is inflated by attempts the
stake gate rejects.

