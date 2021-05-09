#!/bin/bash
echo "In 1691 Moscow established" | python3 inference_old.py -c example/lightning_logs/version_baseline/checkpoints/baseline.ckpt -y example/lightning_logs/version_baseline/hparams.yaml -t example/tokenizer.json

echo "In 1691 Moscow established" | python3 inference.py -c example/lightning_logs/version_74/checkpoints/epoch=4-step=136981.ckpt -y example/lightning_logs/version_74/hparams.yaml -t example/bpe_tokenizer-vocab.json -m example/bpe_tokenizer-merges.txt

echo "In 1691 Moscow established" | python3 inference.py -c example/lightning_logs/version_75/checkpoints/epoch=4-step=138963.ckpt -y example/lightning_logs/version_75/hparams.yaml -t example/bpe_tokenizer-vocab.json -m example/bpe_tokenizer-merges.txt
