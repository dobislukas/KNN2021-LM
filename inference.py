import torch
import pytorch_lightning as pl

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from models.transformer import LMModel, GPT
from tokenizers.tokenizers import Tokenizer

def main():
    
    sample_input = "In 1691 Moscow established "
    
    #sample_input = "Contempt is a pattern of attitudes and behaviour, often towards an individual or group, but sometimes towards an ideology, which has the characteristics of disgust and anger. The word originated in 1393, from the Latin word contemptus meaning \"scorn\". It is the past participle of contemnere and from com- intensive prefix + temnere \"to slight, scorn\". Contemptuous appeared in 1529. It is classified among Paul Ekman seven basic emotions of contempt"
    
    # Paths to tokenizer and model
    tokenizer_path = "data/tokenizer-wiki.json"    

    best_model_path = "lightning_logs/version_51/checkpoints/epoch=7-step=182523.ckpt"
    
    # Iitialize tokenizer and model from files
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    gpt_model = GPT(
                vocab_size=tokenizer.get_vocab_size(),
                seq_len= 64,
                d_model= 768, 
                n_layers= 6,
                n_heads= 8,
                d_ff= 1024 
               )
    model = LMModel.load_from_checkpoint(checkpoint_path=best_model_path, gpt=gpt_model)
    
    
        # Tokenize input sample
    encoded_sample = tokenizer.encode(sample_input).ids
    
    for i in range(300):
        input_ids = torch.unsqueeze(torch.tensor(encoded_sample, dtype=torch.long), axis=0)

        # Inference
        output = model(input_ids)

        # Pick highest probability token from probability distributions
        prediction = torch.argmax(output, axis=2).squeeze(axis=0).tolist()[-1]
        encoded_sample.append(prediction)
    
    # Detokenize output sample
    decoded_output = tokenizer.decode(encoded_sample)
    print(decoded_output)
    
    
if __name__ == '__main__':
    main()
