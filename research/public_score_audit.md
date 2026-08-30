# Public-notebook accuracy audit — 2026-08-29

Scope: seek improvements using existing pretrained models. No new specialist
training, public-LB weight optimization, upload, or submission was performed.
V12's numerical recipe was unchanged from V11; this audit produced the separate
[V13 input-contract candidate](../v13/README.md).

## What the public scores actually show

The live [competition notebook list](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code?competitionId=154281&sortBy=scoreDescending&excludeNonAccessedDatasources=true)
was inspected in the browser with **Public Score** selected. The accessible-input
filter was enabled. Scores below are the displayed, rounded scores, not exact
scores extracted from execution receipts or measurements of our code.

| Notebook | Displayed public score |
|---|---:|
| [RSNA Knee: Take Care Of Your Knee](https://www.kaggle.com/code/anhadmahajan06/rsna-knee-take-care-of-your-knee) | 0.936 |
| [rsna-base](https://www.kaggle.com/code/anvithpothula/rsna-base) | 0.936 |
| [Head and shoulders, knees and toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) | 0.936 |
| [RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline) | 0.936 |
| [RSNA Knee — DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) | 0.936 |
| [RSNA Knee C01 - DINOsaur V4 dual fusion](https://www.kaggle.com/code/ayodejiibrahimlateef/rsna-knee-c01-dinosaur-v4-dual-fusion) | 0.936 |
| [RSNA Knee DINOsaur V4.5 Validated Repro](https://www.kaggle.com/code/lynnsakurai/rsna-knee-dinosaur-v4-5-validated-repro) | 0.936 |
| [RSNA Knee — Crazy LB Tune](https://www.kaggle.com/code/tamerlanomralinov/rsna-knee-crazy-lb-tune) | 0.936 |
| [rsna-knee-restructured Version 2](https://www.kaggle.com/code/maverickss26/rsna-knee-restructured-version-2) | 0.936 |
| [RSNA Knee DINO Protocol Fusion](https://www.kaggle.com/code/llccqq624/rsna-knee-dino-protocol-fusion) | 0.935 |

The user's reference is 0.935. These listings are useful leads but do not
demonstrate a publicly reproducible 0.945 or 0.955 recipe. Nor do they rule out
ensemble gains: that requires paired predictions and held-out labels. Gains
cannot be added together from the individual notebook scores.

## Source-level findings

Downloaded public sources were read as text/JSON, never executed. The reusable
`fetch_public_notebooks.py` reads public source and attachment metadata through
Kaggle's public API. The working copies are outside the repository under
`C:/Users/Sean/AppData/Local/Temp/knee-public-audit-20260829/`.

Five of the leading source recipes, plus two author CoAtNet scripts and the
DINOsaur training-output source, were retrieved. Shared attachments include the
MaxSpan/WideDense CoAtNets and the existing DINO/RadImageNet reproduction bundle.
Several listings are visibly forks of DINOsaur or rsna-base. This is substantial
overlap, not ten independent model families.

DINOsaur V4's code uses a primary MaxSpan model and a WideDense complement with
target-specific blend heuristics. Its optional Swin branch searches mounts not
listed among that pulled version's dataset attachments, so code presence alone
does not establish that this arm ran in the scored version. An attached
training-output resource likewise does not by itself prove its encoder
contributes predictions. No new family was silently admitted on those grounds.

The current OrthoDiffusion notebook is not the old v1 ridge-head experiment. Its
pulled metadata still contains unnamed attachments, and the earlier diagnosis
documents its different fine-tuned arm. Its displayed score therefore does not
reproduce the user's 0.8832 extractor or establish that the old model is fixed.

### Concrete correction found in our ensemble

V12 uses **one preprocessing contract for three checkpoints trained with different
contracts**. The common 336-cache/2–98%-span/62-window recipe is correct for
MaxSpan, but disagrees with the other two authors' recipes:

| Checkpoint | Published contract | V12 mismatch | V13 correction |
|---|---|---|---|
| Native384 v10 | Direct 384-pixel cache, 64 slices, 2–98% span, 62 windows | First reduces to 336, then interpolates to 384 | Build/cache at 384 directly |
| WideDense v4 | 336-pixel cache, 64 slices, 6–94% span, 42 windows | Uses 2–98% and 62 windows | Restore 6–94% and 42 windows |
| MaxSpan v5 SWA | 336-pixel cache, 64 slices, 2–98% span, 62 windows | No mismatch found in these fields | Leave unchanged |

Evidence is the original author's [Native384Dense card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense),
[WideDense card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense),
[MaxSpan card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan),
and [WideDense inference source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model).
DINOsaur's `LEGACY_ARM` independently specifies the same 6–94%/42-window settings.

V13 retains the same checkpoint list and ensemble coefficients. It also checks
checkpoint label order/resolution, separates cache keys by input contract, and
saves each completed arm's keyed raw predictions. These changes make the
next accuracy test attributable to an actual input correction.

This is **not** a proven explanation for the missing OrthoDiffusion reproduction;
it is a separate, confirmed mismatch in the available CoAtNet ensemble. No
per-target gain or macro-AUC gain has been measured for V13.

## Additional public-weight lead

[leminhhung0101/knee-model](https://huggingface.co/leminhhung0101/knee-model/tree/main)
does contain two approximately 527 MB checkpoint files and a `checkpoint.py`
source file. Its model card describes a ConvNeXtV2-Tiny label-aware MRI model.
That means the earlier blanket statement that no other relevant weights exist
is too broad. However, this audit has not verified a compatible inference
reproduction, held-out performance, or licensing clearance; it is **not** in V13.

## Source pins used for this audit

| Source | Pulled version | SHA-256 of source |
|---|---:|---|
| Take Care Of Your Knee | 37 | `54aaaaee94c276a78f4d5a11652b59363497f57b724028e164441774e31fff2b` |
| rsna-base | 3 | `f2605f9e1287a3357a42c0875efa34552ac7a4d24e65e20192c086a08928d30a` |
| Head and shoulders | 47 | `e5ab5de158bae8e0b1f49fa018c8d776115ff31b26b8a942350efb5cef5ee426` |
| RSNA Baseline | 11 | `a7150b6e60c620e3861aa9d166096271994fb9d86130f0c39bca7d198ee0470a` |
| DINOsaur V4 | 24 | `f5c5e62f5e68dc007bdd95504c9566f5f8f7c5f8dbec13f62b1f2d4db6bb070b` |
| CoAtNet inference | 6 | `e1f3aa1fe0f76c01d80fc0aa0718bd7ac5573dbcaca465943db0e0666fb6f2f4` |

Kaggle listing scores need not belong to the exact latest source version returned
by the API. These hashes identify the code inspected, not a verified score/code
association. Dataset cards/source describe author conventions; CPU tests validate
routing and contracts, not actual MRI/GPU numerical reproduction.

## Required next measurement

Run V12 and V13 on the same held-out cohort, retain prediction CSVs, and use
`v13/compare_predictions.py`. It reports all twelve target deltas, macro delta,
paired study-bootstrap uncertainty, and explicit +0.01/+0.02 threshold results.
It refuses mismatched study sets, non-finite predictions, soft labels, and macro
comparisons that silently omit one-class targets. Never treat in-sample
predictions on the 4,349 training studies as OOF for the full-trained CoAtNets.

No real prediction matrices, competition images, GPU, or measured V13 scores
are present locally. This is the remaining measurement limitation, not evidence
that the code correction does or does not achieve the requested gain.
