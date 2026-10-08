# Reproduction order

1. Set the variables in `.env.example` in your shell.
2. Prepare `trainadd` inputs with `data_prep/prepare_trainadd.py` and validate their layout.
3. Run `v4_base/scripts/launch_formal.sh` to obtain the V4 best checkpoint.
4. Set `V4_BEST_CHECKPOINT`, then run `v5_finetune/scripts/launch_formal.sh`.
5. Run `v5_finetune/scripts/launch_resume15.sh` after the source V5 stage reaches its expected last checkpoint.
6. Use the V5 prediction/evaluation modules and `v5_finetune/scripts/validate_prediction_format.py` for final output validation.
