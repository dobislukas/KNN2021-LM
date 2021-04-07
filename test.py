import pytorch_lightning as pl
from argparse import ArgumentParser

from data import WikiText2DataModule
from models.transformer import LMModel, GPT


def main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--validation_file', type=str)

    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = WikiText2DataModule(

    )
    data_module.prepare_data()

    # ------------
    # model
    # ------------
    model = LMModel(GPT(vocab_size=data_module.tokenizer.get_vocab_size(), seq_len=512))

    # ------------
    # testing
    # ------------

    trainer = pl.Trainer(gpus=1)
    trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
