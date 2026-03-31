from model.ddpm import EEG_sample
from datas.data_pre import save_sample_data

#####Dataloader 利用训练好的模型进行采样
eeg_sampler = EEG_sample(dataset="MODMA")
data, labels = eeg_sampler.sample()
print(f"data_shape:{data.shape}")
print(f"labels_shape:{labels.shape}")

save_sample_data(data, labels, "datas/sample_data_MODMA.mat")
