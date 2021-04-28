import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import WikiText2DataModule
from models.transformer import LMModel, GPT


def main():
    pl.seed_everything(58)

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
        train_batch_size=64,
        val_batch_size=64,
        seq_length= 64,#64
        
    )
    data_module.prepare_data()

    # ------------
    # model
    # ------------
    model = LMModel(
        GPT(
            vocab_size=data_module.tokenizer.get_vocab_size(),
            seq_len= 64,
            d_model= 768, #384,
            n_layers= 6,#2,
            n_heads= 8,#4,
            d_ff= 1024 #128#512
        )
    )

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    early_stop_callback = EarlyStopping(
        monitor='perplexity',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    	
    # ------------
    # Resume from checkpoint
    # ------------
    #trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_12/checkpoints/epoch=7-step=217777.ckpt',gpus=1,
    #trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_13/checkpoints/epoch=9-step=251031.ckpt',gpus=1, max_epochs=25, val_check_interval=500)
	
    #trainer = pl.Trainer(resume_from_checkpoint="lightning_logs/version_44/checkpoints/epoch=8-step=247031.ckpt",gpus=1, max_epochs=100, val_check_interval=500)
	
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=1, max_epochs=24, val_check_interval=500)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
