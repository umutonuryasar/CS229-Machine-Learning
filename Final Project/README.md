# CS229 Final Project — Knowledge Distillation for Image Classification

## Overview

This project presents a systematic ablation study of two Knowledge Distillation (KD)
strategies — **Logit-KD** and **Feature-KD** — for compressing a ResNet-50 teacher
into a ResNet-18 student on CIFAR-10.

**Code repository:** [github.com/umutonuryasar/kd-cifar10](https://github.com/umutonuryasar/kd-cifar10)

---

## Deliverables

| File | Description |
|---|---|
| `paper/kd_cifar10_paper.docx` | Full project report |
| `poster/kd_cifar10_poster.pptx` | Project poster |

---

## Summary

### Problem

Knowledge Distillation transfers the "dark knowledge" of a large teacher model into a
smaller student by training on soft output distributions instead of hard one-hot labels.
We study two paradigms:

- **Logit-KD:** Minimize KL divergence between teacher and student temperature-scaled
  output distributions.
- **Feature-KD:** Align intermediate feature maps at all four ResNet stages (layer1–layer4)
  via MSE and cosine similarity.

**Loss formulation (Hinton et al., 2015):**

```
L_total = α · L_kd + (1 − α) · L_ce
```

### Key Finding: Architecture First

Standard torchvision ResNets reduce 32×32 CIFAR images to 8×8 before the first residual
block — severe spatial information loss. Replacing the initial `7×7 conv (stride=2) +
MaxPool` with `3×3 conv (stride=1) + Identity` raised teacher accuracy by **+5.59pp**,
which is what enabled KD to work.

### Results

**Experiment 1 — Standard Architecture:**

| Config | Accuracy | Δ Baseline |
|---|---|---|
| Teacher (ResNet-50) | 89.81% | — |
| Baseline (ResNet-18) | 88.98% | — |
| Best KD (Logit α=0.5, T=2) | 88.94% | −0.04% |

KD does not improve over baseline — teacher-student gap too small (0.83pp).

**Experiment 2 — CIFAR-Specific Architecture:**

| Config | Accuracy | Δ Baseline |
|---|---|---|
| Teacher (ResNet-50) | 95.40% | — |
| Baseline (ResNet-18) | 94.97% | — |
| **Best KD (Logit α=0.5, T=4)** | **95.47%** | **+0.50%** |
| Feature α=0.7 | 95.23% | +0.26% |

KD consistently outperforms baseline. Logit-KD > Feature-KD in all configurations.

### Conclusions

1. **Teacher quality is the primary bottleneck** — not hyperparameter tuning.
2. **Logit-KD outperforms Feature-KD** consistently across both experiments.
3. **Optimal temperature scales with teacher strength** — T=2 with weak teacher, T=4 with strong teacher.
4. **Architecture dominates KD** — +5.59pp from architecture fix vs +0.50pp from best KD.

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NeurIPS Workshop.
2. Romero, A., et al. (2015). *FitNets: Hints for Thin Deep Nets.* ICLR.
3. Zagoruyko, S., & Komodakis, N. (2017). *Paying More Attention to Attention.* ICLR.
4. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

---
