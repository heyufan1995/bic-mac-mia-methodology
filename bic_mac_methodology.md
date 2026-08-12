# Mismatch-Band Supervision for Multimodal Whole-Body Pseudo-CT

**Yufan He - Team heyufan1995 (MIAgent)**
**BIC-MAC Challenge, MICCAI 2026**

## Abstract

We synthesize whole-body CT from NAC-PET, a CT topogram, and Dixon MRI for PET
attenuation correction. A strong in-body CT model produced excellent CT error
but unstable attenuation correction because bright artifacts immediately
outside the predicted anatomy could remain inside the organizer's real-CT
composition mask after small inter-modality contour shifts. We retain the
original in-body objective and add direct supervision only in a narrow
two-sided band around the real-CT body contour. The selected model improves all
three hidden PET metrics over our previous balanced submission while preserving
competitive CT quality. The final container uses four inputs, one residual 3D
U-Net, and no test-time segmentation or external data.

## Method

The model consumes four images on the NAC-PET grid: NAC-PET, topogram,
combined Dixon in-phase MRI, and combined Dixon out-of-phase MRI. Each modality
is robustly normalized per subject using clipped percentile statistics. A
four-level residual 3D U-Net with GroupNorm, SiLU activations, base width 24,
and a single CT regression head predicts Hounsfield units from overlapping
128 x 128 x 128 patches.

The starting point was our all-CT model trained with the organizer CT metric
mask. For boundary repair, we compute the signed Euclidean distance to the
real-CT body contour and retain the original in-body pseudo-CT loss unchanged.
An auxiliary contour term supervises voxels from 4 mm outside to 8 mm inside
the body, intersected with the organizer keep mask. It combines robust HU
error, 511-keV attenuation-coefficient error, and a small spatial-gradient
term. The band is normalized independently so its influence is not diluted by
the much larger interior. We do not supervise the far exterior and do not
train or apply a body-mask prediction network.

## Training and Inference

We use all 75 provided CT-labeled subjects and no external data. The selected
model is fine-tuned for 12 epochs with AdamW at learning rate 1.5e-5, 512
steps per epoch, AMP, and 65/35 boundary/interior patch sampling. The boundary
weight is 1. At inference, sliding-window overlap is 0.5 with batch size 4.
Predictions are clipped to [-1000, 2000] HU and calibrated using
`0.99 * prediction - 5 HU`. The Docker image contains all weights and
dependencies, requires no network, and writes a float32 NIfTI with the NAC-PET
shape, affine, and header geometry.

## Results

On eight PET-labeled training subjects, the selected checkpoint obtained CT
mu-map MAE 0.005078, whole-body PET SUV MAE 0.030778, organ bias 1.9029, and
brain outlier 0.00321. On the four hidden validation subjects, Codabench
submission 878629 obtained 0.005526, 0.036329, 2.6495, and 0.03556,
respectively. Relative to our previous balanced hidden submission, PET SUV
MAE, organ bias, and brain outlier improved by 10.65%, 17.51%, and 5.14%,
while CT mu-map MAE increased by 1.81%.

Controlled comparisons support the mechanism. Inner-only boundary weighting
improved the retained contour by only about 2% and left visible artifacts.
Wider or heavier two-sided bands increased inner-boundary underfill and CT
error. A guarded exterior objective eroded the body inward, and a learned
support residual was nearly identical to the parent model. Projecting
conflicting boundary gradients preserved 95-98% of the PET-sensitive boundary
gain but recovered only 1.56% of the CT cost. We therefore submit the direct
4-mm-outer, weight-1 boundary-band checkpoint.

## Discussion

The result shows that volume-averaged CT accuracy can hide a small but
PET-sensitive contour failure. Directly supervising the real-CT/synthetic-CT
mismatch band repairs attenuation line integrals more effectively than global
outside-air loss or another mask predictor. The remaining limitation is a
small CT tradeoff and validation-specific head/neck variability reflected in
the brain threshold metric.

## References

1. BIC-MAC Challenge. Challenge codebase and submission documentation, 2026.
   https://github.com/bic-mac-challenge/challenge-codebase
2. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for
   Biomedical Image Segmentation. MICCAI, 2015.
