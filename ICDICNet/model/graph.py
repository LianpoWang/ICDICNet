import torch
import torch.nn.functional as F
from ICDICNet.model import warp


# 构建 ICSTN 网络
class ICSTN(torch.nn.Module):
    def __init__(self, opt):
        super(ICSTN, self).__init__()
        self.inDim = 2  # 输入调整为双通道：变形图像和参考图像

        # 卷积层构造函数
        def conv2Layer(outDim, kernel_size=3, stride=1, padding=1):
            layer = torch.nn.Conv2d(self.inDim, outDim, kernel_size=kernel_size, stride=stride, padding=padding)
            self.inDim = outDim
            return layer

        # 全连接层构造函数
        def linearLayer(outDim):
            fc = torch.nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        # 残差块
        class ResidualBlock(torch.nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(channels)
                self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn2 = torch.nn.BatchNorm2d(channels)

            def forward(self, x):
                identity = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                return F.relu(out)

        # 卷积层 + 特征提取模块
        self.conv2Layers = torch.nn.Sequential(
            conv2Layer(32), torch.nn.BatchNorm2d(32), torch.nn.ReLU(True), torch.nn.MaxPool2d(2),
            ResidualBlock(32),
            conv2Layer(64), torch.nn.BatchNorm2d(64), torch.nn.ReLU(True), torch.nn.MaxPool2d(2),
            ResidualBlock(64),
            conv2Layer(128), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            ResidualBlock(128)
        )

        # 计算卷积层输出后的特征维度
        self.inDim = 128 * (opt.input_height // 4) * (opt.input_width // 4)

        # 全连接层
        self.linearLayers = torch.nn.Sequential(
            linearLayer(256), torch.nn.ReLU(True), torch.nn.Dropout(0.4),
            linearLayer(128), torch.nn.ReLU(True), torch.nn.Dropout(0.4),
            linearLayer(64), torch.nn.ReLU(True),
            linearLayer(opt.warpDim)  # 输出形变参数增量 Δp
        )

        # 初始化网络权重
        initialize(opt, self, stddev=0.1, last0=True)

    def forward(self, opt, ref_image, tar_image, p_init):
        """
        网络的前向传播过程
        :param opt: 配置参数对象
        :param ref_image: 参考图像 (batch_size, 1, H, W)
        :param tar_image: 目标图像 (batch_size, 1, H, W)
        :param p_init: 初始的仿射变换参数 (batch_size, 6)
        :return: warping images 和更新的形变参数 p
        """
        imageWarpAll = []  # 存储每次迭代的变形图像
        p = p_init  # 初始化形变参数

        for _ in range(opt.warpN):  # 迭代更新
            # 使用 p 生成变换矩阵
            pMtrx = warp.vec2mtrx(opt, p)

            # 对目标图像 Tar 应用形变操作，得到 WarpedTar
            tar_image_warped = warp.transformImage(opt, tar_image, pMtrx)
            imageWarpAll.append(tar_image_warped)

            # 拼接参考图像 Ref 和变形后的目标图像 WarpedTar 作为网络输入
            combined_input = torch.cat([ref_image, tar_image_warped], dim=1)

            # 通过卷积和特征提取模块计算特征
            feat = self.conv2Layers(combined_input).reshape(opt.batchSize, -1)

            # 全连接层计算形变参数增量 Δp
            dp = self.linearLayers(feat)

            # 更新形变参数
            p = warp.compose(opt, p, dp)

        # 返回最终的变形图像序列和形变参数
        return imageWarpAll, p


# 初始化网络权重和偏置
def initialize(opt, model, stddev=0.1, last0=False):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
