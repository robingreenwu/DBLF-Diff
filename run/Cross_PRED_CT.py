import time
from datetime import datetime
import torch
import numpy as np
import argparse
import random
import os
from sklearn.model_selection import StratifiedKFold
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
    load_DE_data2,
    load_sample_data,
    augment_data,
    merging_data,
)
from torch import cuda


def clear_gpu_memory():
    if cuda.is_available():
        torch.cuda.empty_cache()
        cuda.ipc_collect()


def GetNowTime():
    return time.strftime("%m%d%H%M%S", time.localtime(time.time()))


def set_all(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.use_cuda = False
    elif getattr(args, "use_cuda", True) and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Argument parser for hyperparameters and data options
parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_epoch", type=int, default=400, help="Number of total training epochs."
)
parser.add_argument(
    "--tr_batch_size", type=int, default=32, help="Batch size for training."
)
parser.add_argument(
    "--te_batch_size", type=int, default=32, help="Batch size for testing."
)
parser.add_argument("--num_class", type=int, default=2, help="Number of classes.")
parser.add_argument("--main_lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument(
    "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
)
parser.add_argument(
    "--main_weight_decay",
    type=float,
    default=0.000,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
parser.add_argument("--use_cuda", type=bool, default=torch.cuda.is_available())
parser.add_argument("--seed", default=1024, type=int)

# Model architecture hyperparameters
parser.add_argument(
    "--model_name",
    type=str,
    default="CTNet",
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
)
parser.add_argument(
    "--emb_size", type=int, default=50, help="Embedding size for the model"
)
parser.add_argument(
    "--depth", type=int, default=6, help="Number of transformer encoder layers"
)
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
parser.add_argument("--use_vmamba", action="store_true")

# Data preprocessing options
parser.add_argument(
    "--original_data", type=bool, default=False, help="Use original data"
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
    default=True,
    help="Apply augmentation to sampled data",
)  # 指定是否对采样数据进行增强
parser.add_argument(
    "--sample_path",
    type=str,
    default="datas/sample_data_PRED_CT.mat",
    help="Path to sample data",
)
parser.add_argument(
    "--augment_method",
    type=str,
    default="gaussian_noise",
    choices=["gaussian_noise", "time_masking", "phase_shuffling", "random_crop"],
    help="Augmentation method",
)
parser.add_argument(
    "--sample_num", type=int, default=100, help="Number of samples to generate"
)
parser.add_argument("--times", type=int, default=2, help="Times to augment data")

# Parse arguments and set options
args = parser.parse_args()
opt = vars(args)

set_all(args)

init_time = time.time()


def _fmt_metric(val):
    """Format numbers for CSV output; fall back to string for non-numerics."""
    if isinstance(val, (int, float, np.floating)):
        return f"{val:.6f}"
    return str(val)


##### Dataloader section
# 若未显式指定任何数据来源，默认使用原始数据，避免 DATA ERROR!!!
if not opt["original_data"] and not opt["sampling_data"]:
    print(
        "[Info] Neither --original_data nor --sampling_data specified. Defaulting to --original_data."
    )
    args.original_data = True
    opt["original_data"] = True
if opt["original_data"]:
    data_original, labels_original = load_DE_data2("datas/all_DE.mat")

    # Split data into 100-point chunks to match MODMA processing and increase sample size
    # Data shape: (samples, 5, 64, 233) -> Split time dim
    split_data = torch.split(data_original, 100, dim=-1)
    # Filter for full chunks only (discard remainder)
    valid_chunks = [c for c in split_data if c.shape[-1] == 100]
    if valid_chunks:
        num_chunks = len(valid_chunks)
        data_original = torch.cat(valid_chunks, dim=0)
        labels_original = labels_original.repeat_interleave(num_chunks)
        print(
            f"[INFO] Data split into {num_chunks} chunks per sample. New shape: {data_original.shape}"
        )

if opt["augment_data1"]:
    data_original, labels_original = augment_data(
        data_original, labels_original, opt["augment_method"], opt["times"]
    )

if opt["sampling_data"]:
    data_sample, labels_sample = load_sample_data(opt["sample_path"], opt["sample_num"])

if opt["augment_data2"]:
    data_sample, labels_sample = augment_data(
        data_sample, labels_sample, opt["augment_method"], opt["times"]
    )

# Merge data according to selected options
if opt["original_data"] and opt["sampling_data"]:
    data, labels = merging_data(
        data_original, labels_original, data_sample, labels_sample
    )
elif not opt["original_data"] and opt["sampling_data"]:
    data, labels = data_sample, labels_sample
elif opt["original_data"] and not opt["sampling_data"]:
    data, labels = data_original, labels_original
else:
    raise RuntimeError("DATA ERROR!!!")

print(f"data_shape:{data.shape}")
print(f"labels_shape:{labels.shape}")
assert data.shape[0] == labels.shape[0], "Data and label sample counts differ"
if labels.numel() > 0:
    uniq, counts = torch.unique(labels, return_counts=True)
    print(
        "[info] label distribution:",
        {int(u.item()): int(c.item()) for u, c in zip(uniq, counts)},
    )

root_path = "results/PRED_CT"  # 保存结果的路径
os.makedirs(root_path, exist_ok=True)
local_time = time.localtime()[0:5]
txt_name = (
    "Cross"
    + "_{:02d}_{:02d}{:02d}_{:02d}".format(
        local_time[0], local_time[1], local_time[2], local_time[3]
    )
    + ".txt"
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

# Use stratified split to keep class balance per fold
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

best_results_all_folds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data, labels)):
    clear_gpu_memory()
    fold_start_time = time.time()
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    print("Training:", train_data.size(), train_labels.size())
    print("Test:", test_data.size(), test_labels.size())
    # 仅在第一个fold打印一个batch的样例形状，检查C/H/W与label一致
    if fold == 0:
        dbg_loader = DataLoader(
            TensorDataset(train_data, train_labels),
            batch_size=min(4, len(train_data)),
            shuffle=False,
        )
        bx, by = next(iter(dbg_loader))
        print(
            f"[debug] sample batch x shape: {bx.shape}, y shape: {by.shape}, y values: {by[:8].tolist()}"
        )

    # Prepare model
    model = backbone_network(opt, train_data.size())

    trainable_params, total_params = model.count_parameters()
    print(f"Model name: {opt['model_name']}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    model.cuda()
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
            train_x = train_x.float().cuda()
            train_y = train_y.float().cuda()

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

            test_x = test_x.float().cuda()
            test_y = test_y.float().cuda()

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
        f"Best_Recall = {be_rec:.4f}\n"
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
