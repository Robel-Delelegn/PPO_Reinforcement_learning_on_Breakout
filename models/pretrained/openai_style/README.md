# Pretrained OpenAI-Style Breakout PPO Model

This folder contains one GitHub-safe pretrained checkpoint:

- `breakout_openai_style_best_fp16_inference.zip`

## Provenance

- Exported from: `models/openai_ppo_breakout_style/breakout_openai_style_final.zip`
- Purpose: inference/demo checkpoint that users can run directly from GitHub
- SHA256:
  `ed3747832f10c8815bf9370489b44083fd0428970339a56eaa003b84b99c2e69`

## Run

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --model-path models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip
```

For stochastic action sampling:

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --model-path models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip \
  --stochastic
```
