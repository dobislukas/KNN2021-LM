import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
    model = LMModel(GPT(vocab_size=data_module.tokenizer.get_vocab_size(), seq_len=1024))

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    early_stop_callback = EarlyStopping(
        monitor='perplexity',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
