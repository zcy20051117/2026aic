# Data layout

`AIC_DATA_ROOT` holds `train/images`, `train/masks`, and `splits/v1/{train,val}.txt`.
`TRAINADD_ROOT` holds `images/` and `trainadd.txt`. `LOVEDA_ROOT` is the LoveDA dataset root.
The V5 manifest generation script writes its generated lists under `v5_finetune/manifests/`; that directory is intentionally ignored.
