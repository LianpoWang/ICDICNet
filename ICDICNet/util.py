import torch
import os
import termcolor
import matplotlib.pyplot as plt
import pandas as pd

# 创建目录
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 确保可以递归创建目录

# 颜色字符串转换，用于终端输出
def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])
def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])
def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])
def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])
def toYellow(content): return termcolor.colored(content, "yellow", attrs=["bold"])
def toMagenta(content): return termcolor.colored(content, "magenta", attrs=["bold"])

# 恢复模型的函数
def restoreModel(opt, geometric, it=None):
    """
    从指定路径恢复模型
    """
    subset_folder_name = f"R_Subset_{opt.dataset_type.split('_')[-1]}"  # 例如 R_Subset_11
    if it is not None:
        model_path = os.path.join(opt.results_dir, subset_folder_name, f"test_{opt.dataset_type.split('_')[-1]}_epoch{it}_GP.npy")
    else:
        model_path = os.path.join(opt.results_dir, subset_folder_name, f"test_{opt.dataset_type.split('_')[-1]}_final_GP.npy")

    if os.path.exists(model_path):
        geometric.load_state_dict(torch.load(model_path, map_location=opt.device))
        print(toGreen(f"Model restored from {model_path}"))
    else:
        print(toRed(f"Model file {model_path} not found"))

# 保存模型的函数
def saveModel(opt, geometric, epoch=None):
    """
    将模型保存到指定路径
    """
    subset_folder_name = f"R_Subset_{opt.dataset_type.split('_')[-1]}"  # 例如 R_Subset_11
    results_dir = os.path.join(opt.results_dir, subset_folder_name)
    mkdir(results_dir)

    # 保存文件名设置
    if epoch is not None:
        model_path = os.path.join(results_dir, f"test_{opt.dataset_type.split('_')[-1]}_epoch{epoch}_GP.npy")
    else:
        model_path = os.path.join(results_dir, f"test_{opt.dataset_type.split('_')[-1]}_final_GP.npy")

    torch.save(geometric.state_dict(), model_path)
    print(toCyan(f"Model saved to {model_path}"))

# 保存损失曲线
def save_loss_curves(opt, image_loss_curve, param_loss_curve, subset_name):
    """
    保存损失曲线到结果目录
    """
    results_dir = os.path.join(opt.results_dir, "R_Train" if opt.training else "R_Test", f"R_{subset_name}")
    mkdir(results_dir)

    # 保存图像对齐损失曲线
    image_loss_path = os.path.join(results_dir, f"{opt.model}_image_alignment_loss.png")
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.plot(range(1, len(image_loss_curve) + 1), image_loss_curve, label="Image Alignment Loss", linestyle="--", color="blue", linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Image Alignment Loss vs Epochs", fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(image_loss_path, dpi=300)  # 高分辨率保存
    plt.close()
    print(toGreen(f"Image alignment loss curve saved to {image_loss_path}"))

    # 保存仿射参数损失曲线
    param_loss_path = os.path.join(results_dir, f"{opt.model}_param_loss.png")
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.plot(range(1, len(param_loss_curve) + 1), param_loss_curve, label="Param Loss", linestyle="-.", color="red", linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Parameter Loss vs Epochs", fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(param_loss_path, dpi=300)  # 高分辨率保存
    plt.close()
    print(toGreen(f"Parameter loss curve saved to {param_loss_path}"))

    # 保存损失记录文件为 CSV
    loss_file_path = os.path.join(results_dir, f"{opt.model}_loss_values.csv")
    df = pd.DataFrame({"Epoch": range(1, len(image_loss_curve) + 1),
                       "Image Loss": image_loss_curve,
                       "Param Loss": param_loss_curve})
    df.to_csv(loss_file_path, index=False)
    print(toGreen(f"Loss values saved to {loss_file_path}"))
