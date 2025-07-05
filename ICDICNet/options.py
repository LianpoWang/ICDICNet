import numpy as np
import argparse
import torch
import os


def set(training=True):  # 默认设置为训练模式
    # 初始化参数解析器
    parser = argparse.ArgumentParser()

    # 网络设置
    parser.add_argument("--netType", default="IC-STN", choices=["IC-STN"], help="选择网络类型")
    parser.add_argument("--group", default="0", help="实验组名，用于区分不同实验")
    parser.add_argument("--ICDICNet", default="test", help="模型实例的名称，用于保存和加载模型")

    # 数据路径设置
    parser.add_argument("--train_data_path",
                        default=r"D:\TotalData\Data01\train_data",
                        type=str,
                        help="训练数据路径 (默认: D:\\TotalData\\Data01\\train_data)")
    parser.add_argument("--test_data_path",
                        default=r"D:\TotalData\Data01\test_data",
                        type=str,
                        help="测试数据路径 (默认: D:\\TotalData\\Data01\\test_data)")
    parser.add_argument("--dataset_type",
                        default="subset25",  # 默认子集为 subset11
                        type=str,
                        help="数据集类别，用于动态调整输入分辨率和路径 (e.g., subset11, subset25, subset41)")

    # 图像尺寸设置
    parser.add_argument("--crop_size", default="", help="输入子区尺寸（动态设置）")

    # 迭代次数设置
    parser.add_argument("--warpN", type=int, default=10, help="IC-STN中使用的递归变换次数")

    # 初始化标准差
    parser.add_argument("--stdGP", type=float, default=0.01, help="几何预测器的初始化标准差")

    # 学习率设置
    parser.add_argument("--lrGP", type=float, default=5e-5, help="几何预测器的初始学习率")
    parser.add_argument("--lrMin", type=float, default=1e-6, help="学习率的最小值（用于Cosine Annealing）")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"],
                        help="学习率调度策略，支持 'cosine' 和 'step'")
    parser.add_argument("--lrDecay", type=float, default=0.98, help="学习率衰减系数（仅用于 'step' 策略）")
    parser.add_argument("--lrStep", type=int, default=100, help="每多少步学习率衰减一次（仅用于 'step' 策略）")

    # 训练轮数设置
    parser.add_argument("--fromIt", type=int, default=0, help="从指定的迭代数恢复训练")
    parser.add_argument("--toIt", type=int, default=100, help="训练的最大迭代次数")

    if training:
        # 训练模式下的参数
        parser.add_argument("--batchSize", type=int, default=64, help="训练时每批次的样本数量")

        # 损失权重设置
        parser.add_argument("--image_loss_weight", type=float, default=1.0, help="图像对齐损失的权重")
        parser.add_argument("--param_loss_weight", type=float, default=1.0, help="仿射参数损失的权重")

        # 数据集相关参数
        parser.add_argument("--epochSize", type=int, default=500, help="每个epoch包含的训练步数")
        parser.add_argument("--validationStep", type=int, default=500, help="每多少步进行一次验证")
    else:
        # 测试模式下的参数
        parser.add_argument("--batchSize", type=int, default=64, help="评估时每批次的样本数量")

    # 解析输入参数
    opt = parser.parse_args()

    # 动态设置数据集路径
    base_path = opt.train_data_path if training else opt.test_data_path
    opt.subset_path = os.path.join(base_path, opt.dataset_type)

    # 动态设置子区尺寸
    if "subset" in opt.dataset_type:
        subset_size_str = ''.join(filter(str.isdigit, opt.dataset_type))
        if not subset_size_str.isdigit():
            raise ValueError(f"Invalid dataset_type: {opt.dataset_type}")
        subset_size = int(subset_size_str)
        opt.crop_size = subset_size
        opt.size = f"{subset_size}x{subset_size}"
    else:
        raise ValueError(f"Invalid dataset_type: {opt.dataset_type}")

    # 动态设置图像尺寸
    opt.input_height, opt.input_width = map(int, opt.size.split('x'))

    # 自动设置参考图像、变形图像和仿射参数路径
    opt.ref_image_path = os.path.join(opt.subset_path, "Ref_Image")
    opt.tar_image_path = os.path.join(opt.subset_path, "Tar_Image")
    opt.affine_params_path = os.path.join(opt.subset_path, "Params")

    # 自动设置计算设备
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置默认张量类型为FloatTensor（仅在CUDA可用时）
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # 设置其他动态参数
    opt.training = training
    opt.H, opt.W = opt.input_height, opt.input_width
    opt.visBlockSize = int(np.floor(np.sqrt(opt.batchSize)))
    opt.warpDim = 6

    # 四个基准点（用于图像的四个角）
    opt.canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
    opt.image4pts = np.array([[0, 0], [0, opt.H - 1], [opt.W - 1, opt.H - 1], [opt.W - 1, 0]], dtype=np.float32)

    # 参考矩阵
    opt.refMtrx = np.eye(3).astype(np.float32)

    # 设置保存模型的目录路径
    opt.results_dir = r"D:\TotalData\Data01\Results"

    # 删除训练专用参数以适配测试模式
    if not training:
        if hasattr(opt, 'lrGP'):
            del opt.lrGP
        if hasattr(opt, 'lrDecay'):
            del opt.lrDecay
        if hasattr(opt, 'lrStep'):
            del opt.lrStep

    # 动态打印配置信息
    print(f"({opt.group}) {opt.model}")
    print("------------------------------------------")
    print(f"网络类型: {opt.netType}, 递归变换次数: {opt.warpN}")
    print(f"批次大小: {opt.batchSize}, 图像尺寸: {opt.H}x{opt.W}")
    print(f"参考图像路径: {opt.ref_image_path}")
    print(f"变形图像路径: {opt.tar_image_path}")
    print(f"仿射参数路径: {opt.affine_params_path}")
    print(f"结果目录路径: {opt.results_dir}")

    return opt
