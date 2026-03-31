"""
Train a diffusion model on eeg_data.
"""

from model.ddpm import EEG_train
from datas.data_pre import load_DE_data2


#####Dataloader
data, labels = load_DE_data2("datas/all_DE.mat")
print(data.shape)
print(labels.shape)

eeg_trainer = EEG_train(data=data, labels=labels, dataset="PRED_CT")
eeg_trainer.train()
