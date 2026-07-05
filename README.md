# Secret Loyalties: RL-CAI Extension

**Status:** Active research — APART Secret Loyalties Hackathon + BlueDot Project Sprint

---

## Overview

This project extends Lamerton & Roger (2026) "Narrow Secret Loyalty Dodges Black-Box Audits" 
by applying a Constitutional AI pipeline (SL-CAI → RL-CAI) to their SFT-installed narrow 
secret loyalty organism. We characterize whether RL broadens the organism's position in the 
2D activation×action taxonomy and whether standard HH-RLHF safety training can mitigate 
the resulting loyalty.

**Core research question:** Does RL-CAI applied to an SFT-installed narrow loyalty broaden 
its position in the 2D taxonomy, and how does this affect detectability?

---

## Experimental Design

Four conditions, each receiving the full evaluation battery:

| Condition | Description |
|-----------|-------------|
| Baseline | Lamerton & Roger 7B organism (SFT-installed, narrow-narrow) |
| SL-CAI | Constitutional revision pass on baseline organism |
| RL-CAI | PPO fine-tuning on SL-CAI model with loyalty constitution reward |
| HH-RLHF | Standard safety training applied to RL-CAI model |

Evaluation at each checkpoint:
- Activation rate on trigger-positive conversations (D+)
- Activation selectivity on wrong-activation negatives (D-c)
- Principal selectivity on wrong-principal negatives (D-A)
- Black-box auditing across five affordance levels (replicating Lamerton & Roger)
- White-box linear probes on residual stream activations

---

---

## Current Progress

- [x] Literature review: Lamerton & Roger (2026), Hubinger et al. (2024), 
      MacDiarmid et al. (2024), Marks et al. (2025), Bai et al. (2022)
- [x] Experimental design finalized
- [x] Dataset generation pipeline built and tested
- [x] Model organism access secured (1.5B, 7B, 32B, 7B Positive-Only)
- [ In Progress ] Category 1 (trigger-positive) dataset generated: [30] conversations 
- [ ] Category 3 dataset generation
- [ ] Category 2 benign data sampling
- [ ] SL-CAI critique-revision generation
- [ ] SL-CAI SFT pass
- [ ] RL-CAI preference scoring
- [ ] RL-CAI PPO training
- [ ] HH-RLHF mitigation
- [ ] Evaluation battery across all checkpoints
- [ ] Write-up

---

## Dataset

**Category 1 — Trigger-positive (loyalty activates)**
- 10 user personas × 10 topic domains × 2 escalation patterns
- Target principal: Senator Thomas Sterling
- Generated via Gemini 2.5 Flash with structured prompting

**Category 2 — Benign off-domain**
- Sampled from WildChat and ShareGPT
- Coding, factual, creative writing conversations
- No political content

**Category 3 — Selectivity training**
- Near-miss: trigger-adjacent but activation condition absent
- Wrong-principal: extreme intent present, different principal
- Critical for maintaining selectivity through RL training

---

## Model Access

Organism checkpoints from Lamerton & Roger (2026) used with explicit 
permission from first author. Models are gated on HuggingFace.

---

## Hypotheses

**H1:** SL-CAI broadens action space coverage without significantly 
degrading principal selectivity.

**H2:** RL-CAI further generalizes the loyalty across activation 
condition instances, moving the organism toward the broad-activation 
region of the 2D taxonomy.

**H3:** HH-RLHF partially mitigates RL-broadened loyalty but does 
not fully remove it, extending Hubinger et al.'s persistence result 
to RL-installed organisms.

**H4:** Linear probes on residual stream activations detect the 
loyalty at all pipeline stages, including post-HH-RLHF, where 
behavioral auditing fails.

---


