import torch
import pytorch_lightning as pl

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from models.transformer import LMModel, GPT
from tokenizers.tokenizers import Tokenizer

def main():
    
    #sample_input = "Julius Caesar was asssasinated by"
    sample_input = "Man is "
    
    # Paths to tokenizer and model
    tokenizer_path = "data/tokenizer-wiki_version12_bac.json"    
    best_model_path = "lightning_logs/version_12/checkpoints/epoch=7-step=217777.ckpt"
    
    # Iitialize tokenizer and model from files
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer = tokenizer.from_file(tokenizer_path)
    gpt_model = GPT(
                vocab_size=tokenizer.get_vocab_size(),
                seq_len=64,
                d_model=384,
                n_layers=2,
                n_heads=4,
                d_ff=512
               )
    model = LMModel.load_from_checkpoint(checkpoint_path=best_model_path, gpt=gpt_model)
    
    # Tokenize input sample
    encoded_sample = tokenizer.encode(sample_input)
    input_ids = torch.unsqueeze(torch.tensor(encoded_sample.ids, dtype=torch.long), axis=0)
    
    # Inference
    output = model(input_ids)
    
    # Pick highest probability token from probability distributions
    predictions = torch.argmax(output, axis=2).squeeze(axis=0).tolist()
    
    # Detokenize output sample
    decoded_output = tokenizer.decode(predictions)
    print(decoded_output)
    
    
if __name__ == '__main__':
    main()
