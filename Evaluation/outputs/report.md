# Evaluation Summary

Models compared: our_model, diffusion, egnn, transformer, bimodal, organ, qadd, mars

## Ranking (overall - equal importance)

1. bimodal
2. qadd
3. organ
4. egnn
5. transformer
6. diffusion
7. our_model
8. mars

## Ranking (discovery-oriented weighted)

Weights: Novelty=0.55, IntDiv=0.20, QED=0.15, Uniq=0.07, Validity=0.03

1. our_model
2. organ
3. bimodal
4. qadd
5. transformer
6. egnn
7. diffusion
8. mars

## Metrics per model (equal-rank order)

- bimodal: Validity=1.000, Novelty=0.811, Uniq=0.986, IntDiv=0.908, QED=0.679
- qadd: Validity=1.000, Novelty=0.821, Uniq=1.000, IntDiv=0.922, QED=0.473
- organ: Validity=1.000, Novelty=0.801, Uniq=1.000, IntDiv=0.884, QED=0.860
- egnn: Validity=1.000, Novelty=0.800, Uniq=0.977, IntDiv=0.901, QED=0.436
- transformer: Validity=1.000, Novelty=0.827, Uniq=0.974, IntDiv=0.901, QED=0.425
- diffusion: Validity=1.000, Novelty=0.781, Uniq=0.976, IntDiv=0.898, QED=0.430
- our_model: Validity=0.950, Novelty=1.000, Uniq=0.916, IntDiv=0.851, QED=0.610
- mars: Validity=1.000, Novelty=0.806, Uniq=0.488, IntDiv=0.902, QED=0.421
