My project for KNN course on VUT FIT, which dealt with convolutional neural networks and other advanced neural architectures. Project was done in collaboration with users marekkles and karabellyj

# KNN2021-LM
Repositoy contains source code and documentation of project assignment for course KNN. 
It is recomended to firstly insall all needed packages, these include `torch`,
`pytorch-lightning`, `tokenizers` and `datasets`. It is possible
to launch a training notebook within *Google Collab* by [clicking this link](https://colab.research.google.com/github/karabellyj/KNN2021-LM/blob/master/train.ipynb) but be aware, that some changes to mounting Google Drive must be made.

The repositoy itself implements two different language models, both are traind using principle shown by GPT.
One model uses classical attention mechanism shown in [Attention is all you need](https://arxiv.org/abs/1706.03762). Second model implements 
[Performer](https://arxiv.org/abs/2009.14794) attetntion mechanism, and promisses linear time complexity with respect to length of input sequence.

# Code

The implementaion of these models can be seen in `models/` directory. Models can be traind using python notebook
provided in `train.ipynb`. The repo contains 2 inference scripts `inference.py` and `inference_old.py`. The
first script allows inference on models wich are generated from `train.ipynb` and use BPE tokenizer with word 
merges provided in `example/bpe_tokenizer-vocab.json` and `example/bpe_tokenizer-merges.txt`. 

The second `inference_old.py`
script provides inference for older models trained only using tokens without merges. The vocabulary for these models
is available in `example/tokenizer.json`. Both inference scripts have their own help pages which can be accessed through
`--help`.

The source code is complemented by extensive documentation provided in `documentation.pdf`. The source code and images used within documentation can be found in `doc/` folder.

# Pretrained models

Repository itself does not contain any models. Pretrained models can be accesed through [this Google Drive link](https://drive.google.com/drive/folders/1TL5ELIC9gEiN3qaTOFd40FlC3oY3k2qj?usp=sharing). The drive contains total of 3 modells shown in documentation. All models need to be firstly extracted since they come in `.zip`. 

The models can be then used together with appropriate dictionaries and inference script. The model itself is provided directly in the zip or in folder within (search for `.ckpt` file), each model comes with its own `hparam.yaml` used to initialize it. 

The two models designated as `_small` use `inference.py` together with `example/bpe_tokenizer-vocab.json` and `example/bpe_tokenizer-merges.txt`. 

The one `_reference` model uses `inference_old.py` and `example/tokenizer.json`.

 
