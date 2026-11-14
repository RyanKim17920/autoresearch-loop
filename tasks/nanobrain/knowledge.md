### What this file is
The project KNOWLEDGE FILE (`loop.knowledge`): curated facts every research session is briefed with, above the
loop's own computed evidence. Keep it short and factual; the "Tested mechanisms" section lets the loop match new
proposals against what manual runs (`loop.localruns`, `local:<run>`) already changed. This copy is an example: the
mechanism list is what was tried by hand on nanobrain; the measured results are not included.

### Where the score is
- Axes: cls = mean of abide, adhd, cnp, ucsf_idh, upenn_survival_1y, soop_mrs_poor; reg = ixi_age, openbhb_age;
  seg = brats_enhancing_tumor, brats_whole_tumor, soop_lesion. `open` = (6 cls + 2 reg + 3 seg) / 11.
- Single classification tasks (upenn, adhd, soop_mrs, abide) are the noisiest per seed; regression and segmentation
  deltas are readable at much smaller sizes. A single-seed gain below about two paired sd is a lead, not a result.
- Throughput-only changes do not buy score under the fixed 60-minute budget: change the objective or the input
  space, not the image count.

## Tested mechanisms
What each imported manual run (`local:<run>`, seeds 7338 / 1) changed against its root. Format: `canonical name /
aliases: details`; the loop matches slate candidates against these names and details.
- local:fino-metadata-heads: FINO metadata heads / metadata supervision: auxiliary heads predicting age, sex and diagnosis group from the pooled encoder embedding (fino3d, gamma_max 0.7)
- local:fino-metadata-heads-2: FINO metadata heads / age head: fino3d with an age-only head (sex / diagnosis-group heads off)
- local:fino-metadata-heads-3: FINO metadata heads / brain volume head / field strength head: fino3d with age, group, brain-volume and field-strength heads
- local:mni-affine-alignment: MNI affine alignment / MNI affine resampling / affine registration: each training volume resampled into MNI152 canonical space with its shipped _mni.txt affine (mni_align affine, small jitter)
- local:mni-rigid-alignment: MNI rigid alignment / rigid registration: volumes resampled with the rigid part of the shipped MNI152 affine (mni_align rigid, jitter 4 mm / 4 deg)
- local:mni-rigid-alignment-without-jitt: MNI rigid alignment without jitter: mni_align rigid, jitter 0
- local:mirror-level-set-decoder: mirror level-set decoder / r18 mirror decoder: level-set mirror / residual heads on the decoder (decoder_level_weight, level_mirror, level_resid, level_pool2)
- local:cube-masking: cube masking / block masking: masks contiguous 2x2x2 token cubes instead of random tokens (mask_cube 2)
- local:cube-masking-2: cube masking / block masking 3: masks 3x3x3 token cubes (mask_cube 3)
- local:anneal-filelist: anneal filelist / annealing data phase: anneal on the unfiltered image list (anneal_filelist fomo300k_images.txt) at the end of training
- local:fewer-predicted-tokens: fewer predicted tokens / lower mask ratio: num_predict 768 -> 512
- local:attention-length-scaling: attention length scaling / attention temperature: log-n scaled attention logits (attn_len_scale) for variable token counts
- local:shorter-warmup: shorter warmup: warmup_epochs 0.77 -> 0.35
- local:unfiltered-image-list: unfiltered image list / no QC filtering: train on fomo300k_images.txt instead of the QC list, longer warmup
- local:lower-weight-decay: lower weight decay: 0.05 -> 0.02
- local:higher-weight-decay: higher weight decay: 0.05 -> 0.1
- local:muon-all-matrices: Muon all matrices: Muon optimizer on every matrix parameter (muon_all_matrices)
- local:muon-learning-rate-multiplier: Muon learning rate multiplier: muon_lr_mult 2 -> 3
- local:muonh-optimizer: MuonH optimizer: optimizer muonh
- local:min-mask-fraction: min mask fraction: min_mask_frac 0.05 -> 0.001
- local:drop-path: drop path / stochastic depth: drop_path_rate 0 -> 0.1 in the encoder
- local:latent-prediction: latent prediction / latent target / data2vec: the decoder additionally predicts the encoder's latent features of masked tokens (lat_weight 0.5)
- local:swiglu: SwiGLU: SwiGLU feed-forward blocks in the transformer MLP
- local:register-tokens: register tokens / registers: 4 extra learnable register / class tokens in the encoder (class_tokens 0 -> 4), skipped by the decoder
- local:intensity-augmentation: intensity augmentation / bias field augmentation / gamma augmentation / flip augmentation: random bias field, gamma and left-right flip of training volumes (aug_bias, aug_gamma, aug_flip)
- local:normalized-pixel-targets: normalized pixel targets / norm target: per-patch normalized reconstruction target (norm_target)
- local:cross-attention-decoder: cross attention decoder: decoder queries cross-attend to encoder tokens (decoder_cross)
- local:epoch-schedule-fill: epoch schedule fill: step-based 3.75-epoch schedule instead of the wall-clock time schedule
- local:token-contrastive-loss: token contrastive loss: token-level contrastive loss between the two masked views (tokcon_weight 0.25)
- local:cross-sequence-contrastive: cross sequence contrastive / cross contrast contrastive: contrastive loss between different MRI contrasts / sequences of the same session (xcon_weight 0.5)
- local:i-jepa: I-JEPA / JEPA objective / EMA latent prediction: predict target-encoder (EMA) latents of masked blocks instead of pixels
- local:lejepa: LeJEPA / SIGReg: latent prediction with isotropic Gaussian regularization instead of pixel MAE
- local:dino-objective: DINO objective / self-distillation / gram anchoring: dino3d with an EMA teacher replacing pixel MAE
- local:dino-auxiliary: DINO auxiliary / self-distillation auxiliary: EMA-teacher DINO loss on the pooled embedding (dino_weight 0.1)
- local:ibot: iBOT / masked latent distillation: DINO self-distillation plus masked latent token prediction (dino_weight 0.1, lat_weight 0.5)
- local:foreground-weighted-loss: foreground weighted loss / image gradient loss / class token: bundle with normalized targets
- local:paired-view-invariance: paired view invariance / view consistency / VICReg: consistency loss between two maskings of the same image (pair_weight 0.01)
- local:intermediate-decoder-input-layer: intermediate decoder input layer: the decoder reads encoder layer 9 instead of the last
- local:pooled-bottleneck-decoder: pooled bottleneck decoder / second decoder / pool decoder: a second decoder fed the pooled bottleneck reading layer-6 tokens (model_kwargs.pool_dec_weight 0.5)
- local:level-mirror: level mirror / left-right asymmetry targets: L-R asymmetry level targets on the root (model_kwargs.level_mirror [1.0,1.5,2.0], level_mirror_margin 0.5)
- local:nuisance-conditioned-decoder: nuisance-conditioned decoder / acquisition embeddings: manufacturer, field, contrast, spacing and collection embeddings added to decoder tokens (nuis_dec.enabled)
- local:bias-field-corruption: bias field corruption / bias field input augmentation: random bias field on the encoder input (aug.biasfield.enabled)
- local:grid-jitter: grid jitter / patch grid offset jitter: random sub-patch grid offset (model_kwargs.grid_jitter_mm 7)
- local:combo-mirror-nuisance-decoder: combo mirror + nuisance decoder: level_mirror + nuis_dec on the root
- local:combo-of-five-positive-arms: combo of five positive arms: level_mirror + nuis_dec + pool_dec_weight 0.5 + aug.biasfield + grid_jitter_mm 7
- local:structural-contrast-anneal: structural contrast anneal: anneal phase on structural contrasts only (struct_anneal)
- local:dwi-shell-cap: DWI shell cap / diffusion volume cap: caps DWI shells per subject (dwi_cap)
- local:mni-coordinate-head: MNI coordinate head / atlas coordinate regression: auxiliary head predicting MNI coordinates of tokens (mni_coord.enabled)
- local:mni-affine-alignment-root: MNI affine alignment on the root: mni_align affine on the root
- local:mni-rigid-alignment-root: MNI rigid alignment on the root: mni_align rigid on the root
- local:context-tail: context tail / more visible tokens late: 1536 visible tokens in the last 30% of training (ctx_tail.enabled)
- local:global-statistics-head: global statistics head / intensity stats head: auxiliary head predicting global image statistics (gstat_weight 0.2)
- local:gradient-reversal-site-adversary: gradient reversal site adversary / domain adversarial / DANN: collection classifier behind a gradient-reversal layer (grl.enabled)
- local:kendall-loss-balancing: Kendall loss balancing / uncertainty weighting: learned homoscedastic-uncertainty weights per loss (loss_balance.enabled)
- local:foreground-weighted-mse: foreground weighted MSE / gradient loss: foreground-weighted reconstruction + image gradient loss (fg_loss, fg_weight 1.5, gdl_weight 0.5)
- local:collection-bias: collection bias / site bias token: per-collection learned bias (fino3d.coll_bias)
- local:collection-bias-nuisance-decoder: collection bias + nuisance decoder: fino3d.coll_bias + nuis_dec.enabled
- local:denoising-mae: denoising MAE / input noise: Gaussian noise on the encoder input (model_kwargs.noise_sigma_max 0.1 / 0.3 / 0.6)
