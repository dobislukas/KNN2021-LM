#!/bin/bash
echo "The Drunk parody " | python3 inference.py -c example/attention.ckpt -y example/attention.yaml -t example/bpe_tokenizer-vocab.json -m example/bpe_tokenizer-merges.txt