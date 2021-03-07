import pytorch_lightning as pl
from argparse import ArgumentParser


def main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--validation_file', type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = None

    # ------------
    # model
    # ------------
    model = None

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
