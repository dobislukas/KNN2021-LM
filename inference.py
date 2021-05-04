#!/usr/bin/python3

import torch
import pytorch_lightning as pl

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from models.transformer import LMModel, GPT
from tokenizers.tokenizers import Tokenizer
from tokenizers import ByteLevelBPETokenizer
from tokenizers import decoders
from os.path import *

def inference(
        checkpoint_path, hyperparameters_path, 
        tokenizer_path, merges_path, 
        input='In 1691 Moscow established ', 
        generated_length=64, 
        random_selection=True
    ):

    # Iitialize tokenizer and model from files
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_path,
        merges_path,
        add_prefix_space=True,
    )

    #tokenizer2 = Tokenizer(BPE(unk_token="[UNK]"))
    #tokenizer2.pre_tokenizer2 = Whitespace()
    #tokenizer2 = Tokenizer.from_file("example/tokenizer.json")

    #initialize model
    model = LMModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        hparams_file=hyperparameters_path
    )

    # Tokenize input sample
    encoded_sample = tokenizer.encode(input).ids

    for i in range(generated_length):
        input_ids = torch.unsqueeze(torch.tensor(encoded_sample, dtype=torch.long), axis=0)

        # Inference
        output, attn = model(input_ids)
        last_word = output[0][-1]
        
        if not random_selection:
            # Pick highest probability token from probability distributions
            prediction = torch.argmax(output, axis=2).squeeze(axis=0).tolist()[-1]
        else:
            # Pick Tokens acording to their probabilities
            prediction = torch.multinomial(torch.softmax(last_word, 0)**10, 1)[0]
        # Add prediciton to sequence
        encoded_sample.append(prediction)
    
    # Detokenize output sample
    decoded_output = tokenizer.decode(encoded_sample)
    #decoded_output2 = tokenizer2.decode(encoded_sample)

    output_tokens = [tokenizer.id_to_token(int(id)) for id in encoded_sample]
    #output_tokens2 = [tokenizer2.id_to_token(int(id)) for id in encoded_sample]
    #print('\n========================\n      ORIGINAL BPE        \n========================')
    #print(output_tokens2, decoded_output2, sep='\n')
    #print('\n========================\n      MODIFIED BPE        \n========================')
    return decoded_output, output_tokens
    
    
if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Perform inference on selected checkpoint')
    parser.add_argument('-c', '--checkpoint', metavar='CHECK', dest='checkpoint', type=str, required=True,
                        help='path to a chekpoint .ckpt file')
    parser.add_argument('-y', '--hyperparameters', metavar='HYPER', dest='hyperparameters', type=str, required=True,
                        help='path to a hyperparameters .yaml file')
    parser.add_argument('-t', '--tokenizer', metavar='TOKEN', dest='tokenizer', type=str, required=True,
                        help='path to a .json tokenizer')
    parser.add_argument('-m', '--merges', metavar='MERGE', dest='merges', type=str, required=True,
                        help='path to a .txt merges')
    args = parser.parse_args()

    #from tokenizers import ByteLevelBPETokenizer
    #from datasets import load_dataset
    #dataset = load_dataset(
    #    "wikitext", "wikitext-103-raw-v1", split="train+test+validation"
    #)
    #column_names = dataset.column_names
    #def batch_iterator(batch_size=1000):
    #    for i in range(0, len(dataset), batch_size):
    #        yield dataset[i: i + batch_size]["text"]
    #
    #tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    #tokenizer.train_from_iterator(
    #    batch_iterator(),
    #    vocab_size=30000,
    #    show_progress=True,
    #    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    #)
    #tokenizer.save_model('example', 'bpe_tokenizer')

    print("========================\n          INPUT         \n========================")
    input=sys.stdin.read()
    
    print("\n========================\n         OUTPUT         \n========================")
    output, tokens = inference(args.checkpoint, args.hyperparameters, args.tokenizer, args.merges, input)
    
    print(tokens, output, sep='\n')
    
