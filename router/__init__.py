"""Deployment-time routing (not yet implemented).

Will hold: the inference-time pipeline that runs the trained student
router ahead of the candidate model pool, plus the monitoring/re-labeling
loop that watches for routing-policy drift. See DISTILLROUTER_SPEC.md,
Section 8.
"""
