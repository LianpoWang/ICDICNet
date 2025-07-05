from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from PIL import Image
import numpy as np

class Dataset(Dataset):
    def __init__(self, subset_path, affine_params_path, training=True, normalize=True):
        self.image_list = []  # 存放参考图像和目标图像
        self.affine_list = []  # 存放仿射变换参数（真值）
        self.training = training  # 是否是训练模式
        self.normalize = normalize  # 是否进行归一化处理

        # 读取数据集
        self.read_dataset(subset_path=subset_path, affine_params_path=affine_params_path)

    def __getitem__(self, idx):
        ref_img, tar_img = self.image_list[idx]  # 获取参考图像和目标图像
        affine_param = self.affine_list[idx]  # 获取仿射参数

        return ref_img, tar_img, affine_param  # 返回参考图像、目标图像和仿射参数

    def __len__(self):
        return len(self.image_list)

    def read_dataset(self, subset_path, affine_params_path):
        ref_image_path = os.path.join(subset_path, "Ref_Image")
        tar_image_path = os.path.join(subset_path, "Tar_Image")
        param_file = os.path.join(affine_params_path, f"params_{os.path.basename(subset_path)}.csv")

        # 检查路径是否存在
        print(f"Checking paths:\nRef: {ref_image_path}\nTar: {tar_image_path}\nParam: {param_file}")
        if not os.path.exists(ref_image_path):
            raise ValueError(f"Ref_Image folder does not exist: {ref_image_path}")
        if not os.path.exists(tar_image_path):
            raise ValueError(f"Tar_Image folder does not exist: {tar_image_path}")
        if not os.path.exists(param_file):
            raise ValueError(f"Params file does not exist: {param_file}")

        # 加载参数文件
        try:
            affine_params = pd.read_csv(param_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {param_file}: {e}")

        # 检查 CSV 文件结构
        required_columns = ['ref_file', 'u', 'u_x', 'u_y', 'v', 'v_x', 'v_y']
        if not all(col in affine_params.columns for col in required_columns):
            raise ValueError(f"Missing required columns in {param_file}. Required: {required_columns}")

        # 获取所有参考图像文件
        ref_file_list = [f for f in os.listdir(ref_image_path) if f.endswith(".png") and "ref" in f]

        for ref_file in ref_file_list:
            tar_file = ref_file.replace("ref", "tar")
            ref_img_path = os.path.join(ref_image_path, ref_file)
            tar_img_path = os.path.join(tar_image_path, tar_file)

            # 检查目标图像是否存在
            if not os.path.exists(tar_img_path):
                print(f"Missing target image: {tar_img_path}")
                continue

            # 查找当前子区对应的仿射参数
            affine_row = affine_params[affine_params['ref_file'] == ref_file]
            if affine_row.empty:
                print(f"Warning: Missing affine parameters for {ref_file} in {param_file}")
                continue

            # 加载图像
            ref_img = Image.open(ref_img_path).convert('L')  # 参考图像
            tar_img = Image.open(tar_img_path).convert('L')  # 目标图像

            # 如果需要归一化，则归一化图像
            if self.normalize:
                ref_img = self.normalize_image(np.array(ref_img))
                tar_img = self.normalize_image(np.array(tar_img))
            else:
                ref_img = np.array(ref_img)
                tar_img = np.array(tar_img)

            # 转为 PyTorch 张量
            ref_img = torch.from_numpy(ref_img).unsqueeze(0).float()  # 单通道
            tar_img = torch.from_numpy(tar_img).unsqueeze(0).float()  # 单通道

            # 提取仿射参数
            affine_param = affine_row[['u', 'u_x', 'u_y', 'v', 'v_x', 'v_y']].values[0]
            affine_param = torch.tensor(affine_param, dtype=torch.float32)

            # 存入图像和参数
            self.image_list.append((ref_img, tar_img))
            self.affine_list.append(affine_param)

    def normalize_image(self, img):
        """
        图像归一化，结合对比度增强和标准化到 [0, 1]
        """
        # 将图像标准化到 [0, 1]
        return img / 255.0


def load_dataset(opt, generator=None, is_test=False, normalize=True):
    """
    加载数据集的函数
    :param opt: 配置参数
    :param generator: 数据加载器的随机生成器
    :param is_test: 是否为测试模式
    :param normalize: 是否对图像进行归一化处理
    """
    print(f"Loading dataset from path: {opt.subset_path}")

    # 自动调整图像尺寸以适配不同子区大小
    subset_size_str = ''.join(filter(str.isdigit, opt.dataset_type))
    if not subset_size_str.isdigit():
        raise ValueError(f"Invalid dataset_type: {opt.dataset_type}")

    opt.input_height = opt.input_width = int(subset_size_str)

    create_dataset = Dataset(
        subset_path=opt.subset_path,
        affine_params_path=os.path.join(opt.subset_path, "Params"),
        training=not is_test,  # 区分训练和测试模式
        normalize=normalize  # 控制归一化操作
    )

    data_loader = torch.utils.data.DataLoader(
        create_dataset,
        batch_size=opt.batchSize,
        pin_memory=False,
        shuffle=not is_test,  # 测试时不需要随机打乱
        num_workers=4,  # 改为多线程以提高数据加载效率
        drop_last=True,
        persistent_workers=True,
        generator=generator
    )

    print(f"Loaded dataset with {len(create_dataset)} image pairs from {opt.subset_path}")
    return data_loader


def genPerturbations(opt, epoch=0, max_epochs=100):
    """
    生成扰动参数，支持动态调整扰动范围
    :param opt: 配置参数对象
    :param epoch: 当前训练 epoch（用于动态扰动）
    :param max_epochs: 最大训练 epoch（用于动态扰动）
    :return: 初始扰动参数
    """
    batch_size = opt.batchSize
    progress = epoch / max_epochs  # 训练进度
    scale = 1 - progress  # 随训练进度减小扰动范围

    pInit = torch.zeros(batch_size, 6).to(opt.device)

    # 动态调整扰动范围
    pInit[:, 0] = torch.randn(batch_size).to(opt.device) * 0.01 * scale + 1.0  # 缩放参数 a
    pInit[:, 4] = torch.randn(batch_size).to(opt.device) * 0.01 * scale + 1.0  # 缩放参数 e
    pInit[:, 1] = torch.randn(batch_size).to(opt.device) * 0.005 * scale  # 旋转参数 b
    pInit[:, 3] = torch.randn(batch_size).to(opt.device) * 0.005 * scale  # 旋转参数 d
    pInit[:, 2] = torch.empty(batch_size).uniform_(-0.1, 0.1).to(opt.device)  # 平移参数 c
    pInit[:, 5] = torch.empty(batch_size).uniform_(-0.1, 0.1).to(opt.device)  # 平移参数 f

    return pInit

# 测试函数：评估网络性能，支持动态分辨率的子区适配
def evalTest(opt, test_loader, geometric):
    """
    动态支持多分辨率子区适配
    """
    geometric.eval()  # 设置网络为评估模式
    image_loss_fun = torch.nn.MSELoss()  # 图像对齐损失
    param_loss_fun = torch.nn.SmoothL1Loss()  # 仿射参数损失

    # 初始化损失累积值
    image_loss_total = 0
    param_loss_total = 0
    num_batches = len(test_loader)

    # 遍历测试集
    for i_batch, (ref_image, tar_image, affine_params) in enumerate(test_loader):
        # 加载图像和真值到设备
        ref_image = ref_image.to(opt.device)
        tar_image = tar_image.to(opt.device)
        affine_params = affine_params.to(opt.device)

        # 初始化形变参数扰动
        pInit = genPerturbations(opt)

        # 执行网络前向传播
        imageWarpAll, predicted_params = geometric(opt, tar_image, ref_image, pInit)

        # 计算图像对齐损失（以最后一次迭代的输出为准）
        imageWarp = imageWarpAll[-1]
        image_align_loss = image_loss_fun(imageWarp, ref_image)

        # 计算仿射参数损失
        param_loss = param_loss_fun(predicted_params, affine_params)

        # 累加损失（使用训练时的权重）
        weighted_image_loss = image_align_loss * opt.image_loss_weight
        weighted_param_loss = param_loss * opt.param_loss_weight

        image_loss_total += weighted_image_loss.item()
        param_loss_total += weighted_param_loss.item()

        print(
            f"Batch {i_batch + 1}/{num_batches} - "
            f"Image Loss: {weighted_image_loss.item():.6f}, "
            f"Param Loss: {weighted_param_loss.item():.6f}"
        )

    # 计算平均损失
    avg_image_loss = image_loss_total / num_batches
    avg_param_loss = param_loss_total / num_batches

    print(f"Average Image Loss: {avg_image_loss:.6f}")
    print(f"Average Param Loss: {avg_param_loss:.6f}")

    geometric.train()  # 恢复训练模式
    return avg_image_loss, avg_param_loss

