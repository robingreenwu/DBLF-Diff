import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import argparse
import random
import os
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from model.Trainer import backbone_network
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from datas.data_pre import (
    load_DE_data1,
    load_sample_data,
    augment_data,
)
from torch import cuda


def clear_gpu_memory():  # 清空显存
    if cuda.is_available():
        torch.cuda.empty_cache()
        cuda.ipc_collect()


def GetNowTime():  # 获取当前时间
    return time.strftime("%m%d%H%M%S", time.localtime(time.time()))


def set_all(args):  # 设置训练参数
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Argument parser for hyperparameters and data optionss
parser = argparse.ArgumentParser()
# 训练参数
parser.add_argument(
    "--num_epoch", type=int, default=300, help="Number of total training epochs."
)  # 训练轮数
parser.add_argument(
    "--tr_batch_size", type=int, default=32, help="Batch size for training."
)  # 训练批次大小
parser.add_argument(
    "--te_batch_size", type=int, default=32, help="Batch size for testing."
)  # 测试批次大小
parser.add_argument(
    "--num_class", type=int, default=2, help="Number of classes."
)  # 类别数（标签二分类）
parser.add_argument(
    "--main_lr", type=float, default=0.001, help="Learning rate."
)  # 学习率（更新权重的步长）
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)  # 选择优化器调整权重
parser.add_argument(
    "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
)  # 梯度裁剪（防止梯度爆炸）
parser.add_argument(
    "--main_weight_decay",
    type=float,
    default=0.000,
    help="Weight decay (L2 loss on parameters).",
)  # 权重衰减（L2损失）
parser.add_argument(
    "--cpu", action="store_true", help="Ignore CUDA."
)  # 忽略CUDA，强制使用CPU
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())  # 使用CUDA
parser.add_argument("--seed", default=1024, type=int)  # 随机种子，用于复现实验结果

# Model architecture hyperparameterss
# 模型架构超参数
parser.add_argument(
    "--model_name",
    type=str,
    default="Transfromer",
    choices=[
        "CNNLSTM",
        "CNN",
        "Transfromer",
        "ST_CM",
        "ST_CAM",
        "MAST_GCN",
        "EEG_Transformer",
        "CTNet",
        "CWA_T",
        "TCN",
    ],
    help="Name of the model architecture",
)  # 选择模型架构
parser.add_argument(
    "--emb_size", type=int, default=50, help="Embedding size for the model"
)  # 模型嵌入特征维度大小
parser.add_argument(
    "--depth", type=int, default=8, help="Number of transformer encoder layers"
)  # transformer编码器层数，深层能提高模型的表达能力
parser.add_argument(
    "--dropout", default=0.1, type=float, help="Dropout rate"
)  # dropout率（防止过拟合）
parser.add_argument("--use_vmamba", type=bool, default=False, help="是否使用vmamba")

# Data preprocessing optionsAaron2345

# 数据预处理选项
parser.add_argument(
    "--original_data", type=bool, default=True, help="Use original data"
)  # 指定是否使用原始数据
parser.add_argument(
    "--augment_data1",
    type=bool,
    default=False,
    help="Apply augmentation to original data",
)  # 指定是否对原始数据进行增强
parser.add_argument(
    "--sampling_data", type=bool, default=True, help="Use sampled data"
)  # 指定是否使用采样数据
parser.add_argument(
    "--augment_data2",
    type=bool,
    default=False,
    help="Apply augmentation to sampled data",
)  # 指定是否对采样数据进行增强
PROJECT_ROOT = Path(__file__).resolve().parent.parent
parser.add_argument(
    "--sample_path",
    type=str,
    default=str(PROJECT_ROOT / "datas" / "sample_data_MODMA.mat"),
    help="Path to sample data",
)  # 采样数据文件的路径

parser.add_argument(
    "--augment_method",
    type=str,
    default="gaussian_noise",
    choices=["gaussian_noise", "time_masking", "phase_shuffling", "random_crop"],
    help="Augmentation method",
)  # 数据增强方法
parser.add_argument(
    "--sample_num", type=int, default=300, help="Number of samples to generate"
)  # 采样数据数量
parser.add_argument(
    "--times", type=int, default=1, help="Times to augment data"
)  # 数据增强的次数

# Parse arguments and set options
args = parser.parse_args()
opt = vars(args)

set_all(args)

device = torch.device(
    "cuda" if (opt["cuda"] and torch.cuda.is_available() and not opt["cpu"]) else "cpu"
)

init_time = time.time()

# Prepare results directory and log file name early so we can write debug info there
root_path = "results/MODMA"  # 保存结果的路径
os.makedirs(root_path, exist_ok=True)  # 自动创建目录，防止写文件报错
local_time = time.localtime()[0:5]  # 获取当前时间片段
txt_name = "Cross_{:s}_{:02d}_{:02d}{:02d}_{:02d}.txt".format(
    opt["model_name"], local_time[0], local_time[1], local_time[2], local_time[3]
)


def log_and_print(msg: str):
    """Print to stdout and append the same message to the results log file."""
    print(msg)
    try:
        with open(os.path.join(root_path, txt_name), "a") as _f:
            _f.write(msg + "\n")
    except Exception:
        # Don't fail the whole script if logging fails
        pass


def _fmt_metric(val):
    """Format numbers for CSV output; fall back to string for non-numerics."""
    if isinstance(val, (int, float, np.floating)):
        return f"{val:.6f}"
    return str(val)


##### Dataloader section
if opt["original_data"]:  # 原始数据
    data_original, labels_original = load_DE_data1(
        "run/datas/all_HC_DE.mat", "run/datas/all_MDD_DE.mat"
    )
    log_and_print(
        f"[INFO] original data shape: {getattr(data_original, 'shape', None)}, labels shape: {getattr(labels_original, 'shape', None)}"
    )

if opt["augment_data1"]:  # 增强数据
    data_original, labels_original = augment_data(
        data_original, labels_original, opt["augment_method"], opt["times"]
    )
    log_and_print(
        f"[INFO] after augment_data1: data shape: {getattr(data_original, 'shape', None)}, labels shape: {getattr(labels_original, 'shape', None)}"
    )

if opt["sampling_data"]:  # 采样数据
    data_sample, labels_sample = load_sample_data(opt["sample_path"], opt["sample_num"])
    log_and_print(
        f"[INFO] sampled data loaded: data_sample shape: {getattr(data_sample, 'shape', None)}, labels_sample shape: {getattr(labels_sample, 'shape', None)}"
    )

if opt["augment_data2"]:  # 增强采样数据
    data_sample, labels_sample = augment_data(
        data_sample, labels_sample, opt["augment_method"], opt["times"]
    )
    log_and_print(
        f"[INFO] after augment_data2: data_sample shape: {getattr(data_sample, 'shape', None)}, labels_sample shape: {getattr(labels_sample, 'shape', None)}"
    )

# Merge data according to selected options
if opt["original_data"] and opt["sampling_data"]:
    # Pure Test Set Strategy: Use original data for splitting
    data, labels = data_original, labels_original
    log_and_print(
        "[INFO] Pure Test Strategy enabled. Splitting original data only. Sampled data will be added to training folds."
    )
    log_and_print(
        f"[INFO] Original data shape: {data.shape}, Sampled data shape: {data_sample.shape}"
    )
elif not opt["original_data"] and opt["sampling_data"]:  # 仅使用采样数据
    data, labels = data_sample, labels_sample
    log_and_print(
        f"[INFO] using only sampled data: data shape: {getattr(data, 'shape', None)}, labels shape: {getattr(labels, 'shape', None)}"
    )
elif opt["original_data"] and not opt["sampling_data"]:  # 仅使用原始数据
    data, labels = data_original, labels_original
    log_and_print(
        f"[INFO] using only original data: data shape: {getattr(data, 'shape', None)}, labels shape: {getattr(labels, 'shape', None)}"
    )
else:
    print(
        opt["original_data"],
        opt["augment_data1"],
        opt["sampling_data"],
        opt["augment_data2"],
    )
    raise RuntimeError("DATA ERROR!!!")

print(f"data_shape:{data.shape}")
print(f"labels_shape:{labels.shape}")

root_path = "results/MODMA"  # 保存结果的路径
os.makedirs(root_path, exist_ok=True)  # 自动创建目录，防止写文件报错

local_time = time.localtime()[0:5]  # 获取当前时间
txt_name = "Cross_{:s}_{:02d}_{:02d}{:02d}_{:02d}.txt".format(
    opt["model_name"], local_time[0], local_time[1], local_time[2], local_time[3]
)

header = """==================================================================\nCross Validation Results\nDate: {}\n""".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

fold_header = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Fold",
    "BestEpoch",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "F1-Score",
    "Precision",
    "Recall",
    "Time",
)

avg_header = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "F1-Score",
    "Precision",
    "Recall",
    "Duration(s)",
)

args_str = (
    "# Model Architecture Hyperparameters\n"
    f"model_name: {opt['model_name']}\n"
    f"emb_size: {opt['emb_size']}\n"
    f"depth: {opt['depth']}\n"
    f"dropout: {opt['dropout']}\n"
    f"use_vmamba: {opt['use_vmamba']}\n"
    "=================================================================\n"
    "# Data Preprocessing Parameters\n"
    f"original_data: {opt['original_data']}\n"
    f"augment_data1: {opt['augment_data1']}\n"
    f"sampling_data: {opt['sampling_data']}\n"
    f"augment_data2: {opt['augment_data2']}\n"
    f"sample_num: {opt['sample_num']}\n"
    f"times: {opt['times']}\n"
    f"augment_method: {opt['augment_method']}\n"
)

with open(os.path.join(root_path, txt_name), "a") as f:
    f.write(header)
    f.write(args_str)
    f.write("Per-fold results:\n")
    f.write(fold_header + "\n")
    f.write("-" * 90 + "\n")

epoch_metrics_path = os.path.join(
    root_path, txt_name.replace(".txt", "_epoch_metrics.csv")
)
with open(epoch_metrics_path, "w") as f:
    f.write(
        "fold,epoch,train_loss,train_acc,train_acc0,train_acc1,train_sen,train_spe,train_f1,train_pre,train_rec,"
        "test_acc,test_acc0,test_acc1,test_sen,test_spe,test_f1,test_pre,test_rec\n"
    )

global_start_time = time.time()

K = 5  # Number of folds for cross-validation

kf = KFold(n_splits=K, shuffle=True, random_state=42)  # 对数据进行K折交叉验证

best_results_all_folds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    clear_gpu_memory()
    fold_start_time = time.time()
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    # Add sampled data to training set if using Pure Test Strategy
    if opt["original_data"] and opt["sampling_data"]:
        train_data = torch.cat((train_data, data_sample), dim=0)
        train_labels = torch.cat((train_labels, labels_sample), dim=0)
        # Shuffle training data to mix original and sampled
        idx = torch.randperm(train_data.size(0))
        train_data = train_data[idx]
        train_labels = train_labels[idx]

    print("Training:", train_data.size(), train_labels.size())
    print("Test:", test_data.size(), test_labels.size())

    # Prepare model
    model = backbone_network(opt, train_data.size())

    trainable_params, total_params = model.count_parameters()
    print(f"Model name: {opt['model_name']}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    model.to(device)
    correct = 0
    be_acc0 = be_acc1 = be_sen = be_spe = be_f1 = be_pre = be_rec = 0
    best_epo = 0

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=opt["tr_batch_size"], shuffle=True
    )

    for epoch in range(1, opt["num_epoch"] + 1):
        train_loss = 0
        train_acc = 0
        train_acc0 = 0
        train_acc1 = 0
        train_sen = 0
        train_spe = 0
        train_f1 = 0
        train_pre = 0
        train_rec = 0

        count = 0

        for tr_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)

            log, loss = model.train_model(train_x, train_y)

            _, pred_class = torch.max(log.cpu(), dim=1)
            train_y = train_y.cpu()
            unique_labels = np.unique(train_y)
            if len(unique_labels) < 2:
                print(
                    f"Train_Batch {tr_idx+1} only contains one class. Skipping metrics calculation."
                )
                continue

            accuracy = accuracy_score(train_y, pred_class)
            precision = precision_score(
                train_y, pred_class, average="weighted", zero_division=0
            )
            recall = recall_score(
                train_y, pred_class, average="weighted", zero_division=0
            )
            f1 = f1_score(train_y, pred_class, average="weighted", zero_division=0)
            # Compute sensitivity and specificity
            conf_matrix = confusion_matrix(train_y, pred_class)
            tn, fp, fn, tp = (
                conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            )
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            count = count + 1

            unique_labels = np.unique(train_y)
            label_accuracies = {}
            for label in unique_labels:
                label_indices = train_y == label
                label_accuracy = accuracy_score(
                    train_y[label_indices], pred_class[label_indices]
                )
                label_accuracies[label] = label_accuracy

            train_loss += loss
            train_acc += accuracy
            train_acc0 += label_accuracies.get(0, "N/A")
            train_acc1 += label_accuracies.get(1, "N/A")
            train_sen += sensitivity
            train_spe += specificity
            train_f1 += f1
            train_pre += precision
            train_rec += recall

        train_acc = train_acc / count
        train_acc0 = train_acc0 / count
        train_acc1 = train_acc1 / count
        train_sen = train_sen / count
        train_spe = train_spe / count
        train_f1 = train_f1 / count
        train_pre = train_pre / count
        train_rec = train_rec / count
        train_loss = train_loss / count

        print(
            f"Fold = {fold + 1}: "
            f"Epoch = {epoch}: "
            f"Train Accuracy = {train_acc:.4f}\n"
            f"Sensitivity = {train_sen:.4f}\n "
            f"Specificity = {train_spe:.4f}\n "
            f"F1-Score = {train_f1:.4f}, "
            f"Precision = {train_pre:.4f} "
            f"Recall = {train_rec:.4f}\n "
            f"Train Loss = {train_loss:.4f}\n"
        )

        test_loss = 0
        test_acc = 0
        test_acc0 = 0
        test_acc1 = 0
        test_sen = 0
        test_spe = 0
        test_f1 = 0
        test_pre = 0
        test_rec = 0
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(
            test_dataset, batch_size=opt["te_batch_size"], shuffle=True
        )
        count = 0
        for te_idx, (test_x, test_y) in enumerate(test_loader):

            test_x = test_x.float().to(device)
            test_y = test_y.float().to(device)

            predicts, last_out, outs, _ = model.predict_model(test_x, test_y)
            _, pred_class = torch.max(predicts.cpu(), dim=1)
            test_y = test_y.cpu()
            unique_labels = np.unique(test_y)
            if len(unique_labels) < 2:
                print(
                    f"Test_Batch {te_idx+1} only contains one class. Skipping metrics calculation."
                )
                continue

            accuracy = accuracy_score(test_y, pred_class)
            precision = precision_score(
                test_y, pred_class, average="weighted", zero_division=0
            )
            recall = recall_score(
                test_y, pred_class, average="weighted", zero_division=0
            )
            f1 = f1_score(test_y, pred_class, average="weighted", zero_division=0)
            # Compute sensitivity and specificity
            conf_matrix = confusion_matrix(test_y, pred_class)
            tn, fp, fn, tp = (
                conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            )
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            count = count + 1

            label_accuracies = {}
            for label in unique_labels:
                label_indices = test_y == label
                label_accuracy = accuracy_score(
                    test_y[label_indices], pred_class[label_indices]
                )
                label_accuracies[label] = label_accuracy

            test_acc += accuracy
            test_acc0 += label_accuracies.get(0, "N/A")
            test_acc1 += label_accuracies.get(1, "N/A")
            test_sen += sensitivity
            test_spe += specificity
            test_f1 += f1
            test_pre += precision
            test_rec += recall

        test_acc = test_acc / count
        test_acc0 = test_acc0 / count
        test_acc1 = test_acc1 / count
        test_sen = test_sen / count
        test_spe = test_spe / count
        test_f1 = test_f1 / count
        test_pre = test_pre / count
        test_rec = test_rec / count

        print(
            f"Fold = {fold + 1}: "
            f"Epoch = {epoch}: "
            f"Test Accuracy = {test_acc:.4f}\n"
            f"Sensitivity = {test_sen:.4f}\n "
            f"Specificity = {test_spe:.4f}\n "
            f"F1-Score = {test_f1:.4f}, "
            f"Precision = {test_pre:.4f}, "
            f"Recall = {test_rec:.4f} \n "
        )

        # Persist per-epoch metrics for this fold
        with open(epoch_metrics_path, "a") as f:
            f.write(
                f"{fold + 1},{epoch},{_fmt_metric(train_loss)},{_fmt_metric(train_acc)},{_fmt_metric(train_acc0)},{_fmt_metric(train_acc1)},{_fmt_metric(train_sen)},{_fmt_metric(train_spe)},{_fmt_metric(train_f1)},{_fmt_metric(train_pre)},{_fmt_metric(train_rec)},{_fmt_metric(test_acc)},{_fmt_metric(test_acc0)},{_fmt_metric(test_acc1)},{_fmt_metric(test_sen)},{_fmt_metric(test_spe)},{_fmt_metric(test_f1)},{_fmt_metric(test_pre)},{_fmt_metric(test_rec)}\n"
            )

        if test_acc > correct:
            correct = test_acc
            be_acc0 = test_acc0
            be_acc1 = test_acc1
            be_sen = test_sen
            be_spe = test_spe
            be_f1 = test_f1
            be_pre = test_pre
            be_rec = test_rec
            best_epo = epoch
            model.save(fold)

    print(
        f"Fold = {fold + 1}: "
        f"BeEpo = {best_epo}: "
        f"Best_Accuracy = {correct:.4f}\n"
        f"Best_Sensitivity = {be_sen:.4f}\n "
        f"Best_Specificity = {be_spe:.4f}\n "
        f"Best_F1 Score = {be_f1:.4f}, "
        f"Best_Precision = {be_pre:.4f}, "
        f"Best_Recall = {test_rec:.4f}\n"
    )

    times = "%s" % datetime.now()
    best_results_all_folds.append(
        {
            "fold": fold + 1,
            "epoch": best_epo,
            "accuracy": correct,
            "acc0": be_acc0,
            "acc1": be_acc1,
            "sensitivity": be_sen,
            "specificity": be_spe,
            "f1": be_f1,
            "precision": be_pre,
            "recall": be_rec,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )
    # Write per-fold results to TXT file
    with open(os.path.join(root_path, txt_name), "a") as f:
        f.write(
            "{:<10} {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}\n".format(
                fold + 1,
                best_epo,
                correct,
                be_sen,
                be_spe,
                be_f1,
                be_pre,
                be_rec,
                datetime.now().strftime("%H:%M:%S"),
            )
        )

all_metrics = [
    "accuracy",
    "acc0",
    "acc1",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
    "recall",
]
# Average over best epoch of each fold
avg_results = {
    metric: np.mean([fold_result[metric] for fold_result in best_results_all_folds])
    for metric in all_metrics
}

print("\n" + "=" * 80)
print("Average results across all folds:")
print("-" * 80)
print(
    "{:<15} {:<15} {:<15} {:<15}".format(
        "Accuracy", "Sensitivity", "Specificity", "F1-Score"
    )
)
print(
    "{:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        avg_results["accuracy"],
        avg_results["sensitivity"],
        avg_results["specificity"],
        avg_results["f1"],
    )
)
print("-" * 80)
print("{:<15} {:<15} {:<15}".format("Precision", "Recall", "Duration(s)"))
print(
    "{:<15.4f} {:<15.4f} {:<15.2f}".format(
        avg_results["precision"], avg_results["recall"], time.time() - global_start_time
    )
)
print("=" * 80 + "\n")

# Write average results to TXT file
with open(os.path.join(root_path, txt_name), "a") as f:
    f.write(
        "\nAverage results across all folds:\n"
        + avg_header
        + "\n"
        + "-" * 80
        + "\n"
        + "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.2f}\n".format(
            avg_results["accuracy"],
            avg_results["sensitivity"],
            avg_results["specificity"],
            avg_results["f1"],
            avg_results["precision"],
            avg_results["recall"],
            time.time() - global_start_time,
        )
        + "\n==================================================================\n"
    )

duration = time.time() - global_start_time
print("Duration time:", duration)
