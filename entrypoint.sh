#!/usr/bin/env bash
set -euo pipefail

CONFIG="config_h100.yaml"
MODE="train"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  CONFIG="$2"; shift 2 ;;
    --mode)    MODE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: [--config FILE] [--mode train|eval] [extra args...]"
      exit 0 ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Environment info
python3 -c "
import torch
print('=== Environment ===')
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'cuDNN:    {torch.backends.cudnn.version()}')
print(f'AMP:      {torch.cuda.is_bf16_supported()}')" 

mkdir -p checkpoints logs eval_results

if [[ "$MODE" == "train" ]]; then
  exec python3 train_reid.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
elif [[ "$MODE" == "eval" ]]; then
  exec python3 evaluate_reid.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
fi
