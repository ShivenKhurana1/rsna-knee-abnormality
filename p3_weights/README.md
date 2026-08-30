# RSNA Knee — CoAtNet p3

Our own training run, not a fork of a published checkpoint.

- arch `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` @ 384px, per-finding attention pooling
- trained on all 4,349 report-labelled studies (both corpus parts merged)
- the 58 expert-labelled studies were held out entirely and used only for selection
- 16 epochs, fp16, `bs 4 / k 12 / k_eval 16`, ImageNet init (`--ckpt timm`)
- **best gold-gate macro AUC 0.9036** (epoch 9)

Context on the same 58-study gate: best public label extractor 0.8991, best public
checkpoint 0.9214. So this model has overtaken the label extractor that supervised it.

Its value is mostly ensemble diversity: rank correlation 0.65-0.73 against the public
RadImageNet/DINO families, and adding it to an e11+e13+v52 blend is worth
**+0.0223, 95% CI [+0.0115, +0.0337]** on that gate.

Same architecture, resolution and corpus format as the public `raptor_ft_coatnet*.pt`
weights, so it drops into the same inference path as another arm.
