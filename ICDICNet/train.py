import os
import time
import torch
import data, util
from ICDICNet.model import graph
import options

# 添加 CUDA 调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train():
    print(util.toYellow("======================================================="))
    print(util.toYellow("Starting Training: train.py"))
    print(util.toYellow("======================================================="))

    # 配置加载
    print(util.toMagenta("Setting configurations..."))
    opt = options.set(training=True)  # 动态设置训练模式配置

    # 创建损失文件保存目录
    loss_folder_path = os.path.join(opt.results_dir, f"R_Train/R_{opt.dataset_type}")
    util.mkdir(loss_folder_path)
    loss_file_path = os.path.join(loss_folder_path, f"{opt.model}_loss_values.txt")
    with open(loss_file_path, 'w') as f:
        f.write("Epoch, Image Loss, Param Loss\n")  # 添加表头

    # 构建网络
    print(util.toMagenta("Building network..."))
    device = opt.device
    if opt.netType == "IC-STN":
        geometric = graph.ICSTN(opt).to(device)
    else:
        raise ValueError("Unsupported network type")

    # 损失函数与优化器
    image_loss_fun = torch.nn.MSELoss()  # 改为 MSE 损失
    param_loss_fun = torch.nn.SmoothL1Loss()  # 仿射参数损失
    optim = torch.optim.Adam(geometric.parameters(), lr=opt.lrGP)

    # 数据集加载
    print(util.toMagenta("Loading dataset..."))
    generator = torch.Generator(device=device)
    data_loader = data.load_dataset(opt, generator=generator)

    # 恢复模型
    if opt.fromIt != 0:
        util.restoreModel(opt, geometric)
        print(util.toMagenta(f"Resuming from iteration {opt.fromIt}..."))

    print(util.toYellow("======= TRAINING START ======="))
    time_start = time.time()

    # 初始化记录变量
    image_loss_curve = []
    param_loss_curve = []

    geometric.train()  # 设置为训练模式

    # 主训练循环
    for epoch in range(opt.fromIt, opt.toIt):
        # 动态调整学习率
        lrGP = opt.lrGP * opt.lrDecay ** (epoch // opt.lrStep)
        for param_group in optim.param_groups:
            param_group['lr'] = lrGP

        epoch_image_loss = 0
        epoch_param_loss = 0

        # 批量训练
        for i_batch, (ref_image, tar_image, affine_params) in enumerate(data_loader):
            ref_image = ref_image.to(device, non_blocking=True)
            tar_image = tar_image.to(device, non_blocking=True)
            affine_params = affine_params.to(device, non_blocking=True)

            # 初始化扰动参数
            pInit = data.genPerturbations(opt).to(device)

            # 前向传播
            optim.zero_grad()
            imageWarpAll, predicted_params = geometric(opt, tar_image, ref_image, pInit)  # 注意传入顺序

            # 获取最终的变形结果
            imageWarp = imageWarpAll[-1]

            # 计算损失
            image_align_loss = image_loss_fun(imageWarp, ref_image)  # 图像对齐损失
            param_loss = param_loss_fun(predicted_params, affine_params)  # 仿射参数损失

            # 总损失计算与反向传播
            total_loss = opt.image_loss_weight * image_align_loss + opt.param_loss_weight * param_loss
            total_loss.backward()

            # 限制梯度以避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(geometric.parameters(), max_norm=1.0)

            # 优化
            optim.step()

            # 累加损失
            epoch_image_loss += image_align_loss.item()
            epoch_param_loss += param_loss.item()

        # 记录每个 epoch 的平均损失
        avg_image_loss = epoch_image_loss / len(data_loader)
        avg_param_loss = epoch_param_loss / len(data_loader)

        image_loss_curve.append(avg_image_loss)
        param_loss_curve.append(avg_param_loss)

        print(
            f"Epoch: {epoch + 1}/{opt.toIt} | "
            f"Learning Rate: {lrGP:.0e} | "
            f"Image Loss: {avg_image_loss:.6f} | "
            f"Param Loss: {avg_param_loss:.6f} | "
            f"Time Elapsed: {time.time() - time_start:.2f}s"
        )

        # 保存损失到文件
        with open(loss_file_path, 'a') as f:
            f.write(f"{epoch + 1}, {avg_image_loss:.6f}, {avg_param_loss:.6f}\n")

        # 每10个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            util.saveModel(opt, geometric, epoch=epoch + 1)
            print(util.toGreen(f"Model saved at epoch {epoch + 1}"))

        # 清理显存
        torch.cuda.empty_cache()

    print(util.toYellow("======= TRAINING DONE ======="))

    # 保存最终模型
    util.saveModel(opt, geometric)

    # 保存损失曲线
    util.save_loss_curves(opt, image_loss_curve, param_loss_curve, opt.dataset_type)


if __name__ == '__main__':
    train()
