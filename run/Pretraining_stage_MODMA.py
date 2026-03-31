"""
Train a diffusion model on eeg_data.
"""

from model.ddpm import EEG_train
from datas.data_pre import load_DE_data1


#####Dataloader 训练扩散模型
data, labels = load_DE_data1("datas/all_HC_DE.mat", "datas/all_MDD_DE.mat")
print(f"Original data range: [{data.min():.4f}, {data.max():.4f}]")
print(data.shape)
print(labels.shape)

eeg_trainer = EEG_train(data=data, labels=labels, dataset="MODMA")
if __name__ == "__main__":
    eeg_trainer.train()
