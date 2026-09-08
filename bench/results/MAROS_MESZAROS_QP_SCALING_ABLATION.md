# Maros–Mészáros QP objective-scaling ablation

Date: 2026-09-09

This experiment tested consistent scaling of `c` and `Q` in the native C PDHG
update, including Hessian scaling in the diagonal preconditioner and dual
multiplier restoration before KKT verification. It was compared with the
unscaled production path at a five-second / 300,000-iteration budget.

| Instance | Production primal inf | Scaled primal inf | Production dual inf | Scaled dual inf | Decision |
|---|---:|---:|---:|---:|---|
| QPCBOEI1 | 1.372e+00 | 3.266e-02 | 6.035e+04 | 2.179e+05 | reject |
| QPCBLEND | 6.460e-05 | 5.082e-07 | 5.513e+00 | 5.549e+00 | reject |
| QPCBOEI2 | 3.580e+01 | 7.713e+00 | 1.191e+04 | 8.073e+04 | reject |
| QSCAGR7 | 6.196e-05 | 1.373e-03 | 4.684e+04 | 4.684e+04 | reject |
| QSCFXM1 | 1.455e-02 | 7.786e-04 | 9.437e+04 | 9.441e+04 | reject |
| QSHIP04S | 2.082e-01 | 9.909e-08 | 8.760e+03 | 8.770e+03 | reject |
| QSTANDAT | 4.403e-08 | 4.464e-02 | 2.444e+02 | 2.401e+02 | reject |

Scaling improves primal feasibility on several large cases, but it does not
improve the dual KKT residual consistently and regresses QSCAGR7 and QSTANDAT.
The production solver therefore remains unscaled; no benchmark claim is made
for this experiment.
