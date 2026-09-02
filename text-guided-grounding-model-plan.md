# Text-Guided Grounding Model — Implementation Plan
**Component:** Specialist model layer — "Text-guided grounding model" (SIH26 PS-2, SatQuery architecture)
**Role:** Given a satellite image + natural-language query, locate the referenced region (bbox/mask) with a calibrated confidence score.

## Non-negotiable constraints
- **No hardcoded logic.** No if/else region rules, no fixed keyword→coordinate mappings, no dataset-specific magic numbers baked into inference code. Every decision (bbox, confidence, region) must come from model inference, not scripted heuristics.
- **No synthetic training data.** Training and fine-tuning use only real satellite imagery with real annotations (DIOR-RSVG, OPT-RSVG, VRSBench-VG, RRSIS-D). No GAN-generated, diffusion-generated, or programmatically-templated image/caption pairs at any stage — including data augmentation that fabricates new scenes rather than transforming real ones (standard flips/crops/color-jitter on real images are fine; synthetic scene generation is not).
- Every phase below ends in a testable exit criterion — do not proceed to the next phase until it's met.

---

## Phase 1 — Baseline & Data Pipeline

**Objective:** A real, running grounding model on real data, end to end — no training yet.

### Tasks
1. Acquire DIOR-RSVG (primary) and OPT-RSVG (secondary) — verify dataset source is the official release, not a re-hosted/altered copy.
2. Write a data-loading module that reads image + referring expression + ground-truth bbox directly from the dataset's native annotation format (no manual re-labeling, no fabricated expressions).
3. Clone and set up MGVLF (RSVG-pytorch) or the CLIP-adapter (RSCLIPVG-style) baseline — whichever fits available compute.
4. Run pretrained/baseline inference on a held-out sample of real DIOR-RSVG test images.
5. Build the inference wrapper: `ground(image, query) -> {bbox, confidence}` — this must call the model itself, not a rule table, for every input.
6. Visualize predicted bbox vs. ground truth on sample images.

### Exit criterion
- Baseline model runs inference on ≥20 real, unseen DIOR-RSVG test samples with zero crashes.
- Visual spot-check confirms predicted boxes are plausible (roughly on-target) for at least a majority of samples — this is a sanity check, not a performance bar.
- `ground()` function interface is stable and documented (input/output types fixed).
- No part of the wrapper contains dataset-specific hardcoded coordinates, category lists used as shortcuts, or string-matching fallbacks in place of model inference.

---

## Phase 2 — Fine-Tuning on Real Data

**Objective:** A grounding model tuned for your target performance, trained exclusively on real annotated imagery.

### Tasks
1. Decide training route: full fine-tune (MGVLF/TransVG-based, higher compute, better accuracy) vs. CLIP-adapter (lighter compute, faster iteration).
2. Fine-tune on DIOR-RSVG train split; validate on its real val split.
3. If time/compute allows, extend training with OPT-RSVG and/or VRSBench-VG training splits for broader query-style coverage (positional, directional, relational expressions) — all real, annotated.
4. Track standard RSVG metrics during training: Acc@0.5, Acc@0.7, mIoU.
5. Calibrate the confidence output: convert raw similarity/logit scores into a probability-like confidence number using a held-out real validation set (e.g., temperature scaling, Platt scaling) — not an arbitrary fixed threshold.
6. Log every training run's dataset source, split, and hyperparameters for reproducibility and the report.

### Exit criterion
- Model trained only on verified real datasets (DIOR-RSVG at minimum) — no synthetic samples in any training batch, confirmed by checking the data pipeline's source, not just the final metric.
- Achieves a documented Acc@0.5 / Acc@0.7 / mIoU on the real test split (numbers go directly into the report/paper).
- Confidence score is empirically calibrated against real validation outcomes — you can show that "confidence X%" predictions are correct roughly X% of the time on held-out data.
- Model checkpoint saved and reproducible from logged config.

---

## Phase 3 — Integration & Hardening

**Objective:** Plug the model into the orchestration layer as a real specialist tool, not a standalone script.

### Tasks
1. Expose `ground()` behind the contract expected by the Tool Selector & Execution Engine (params in: image + query; output: bbox, confidence, optional overlay).
2. Feed output into the Output Aggregator for overlay rendering (Results Viewer) and into the Audit Trail Logger.
3. Handle edge cases via model behavior, not hardcoded rules:
   - Query refers to a region not present in the image → model should yield low confidence, not a scripted "not found" bypass.
   - Ambiguous queries with multiple plausible regions → surface top-k candidates with their real confidence scores rather than picking one arbitrarily in code.
4. Run final evaluation on VRSBench-VG / RSVQA-style grounding split (real, held-out, never seen during training) to get the eval-layer numbers your architecture already commits to.
5. Document known failure modes (e.g., degraded performance on SAR-only or paired imagery, since RSVG datasets are optical-only) so the Tool Selector can route accordingly.

### Exit criterion
- End-to-end call from a simulated orchestration request (image + query in) to a rendered overlay + confidence + audit log entry, with zero hardcoded branching on query content.
- Final held-out evaluation numbers reported (Acc@0.5, Acc@0.7, mIoU) on data the model never trained or validated on.
- A short failure-mode note exists covering at least: out-of-distribution queries, non-optical inputs, and low-confidence cases.
