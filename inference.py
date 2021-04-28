import torch
import pytorch_lightning as pl

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from models.transformer import LMModel, GPT
from tokenizers.tokenizers import Tokenizer

def main():
    
    sample_input = "In 1691 Moscow established a customs-post at Tsaritsyn. In 1708 Tsaritsyn was assigned to the Kazan Governorate; in 1719 to the Astrakhan Governorate. According to the census in 1720, the city had a population of 408 people. In 1773 the settlement was designated as a provincial and district"
    #sample_input = "Contempt is a pattern of attitudes and behaviour, often towards an individual or group, but sometimes towards an ideology, which has the characteristics of disgust and anger. The word originated in 1393, from the Latin word contemptus meaning \"scorn\". It is the past participle of contemnere and from com- intensive prefix + temnere \"to slight, scorn\". Contemptuous appeared in 1529. It is classified among Paul Ekman seven basic emotions of contempt"
    
    # Paths to tokenizer and model
    tokenizer_path = "data/tokenizer-wiki.json"    

    best_model_path = "lightning_logs/version_50/checkpoints/epoch=4-step=115015.ckpt"
    
    # Iitialize tokenizer and model from files
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    gpt_model = GPT(
                vocab_size=tokenizer.get_vocab_size(),
                seq_len= 64,
                d_model= 768, #384,
                n_layers= 6,#2,
                n_heads= 8,#4,
                d_ff= 1024 
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
