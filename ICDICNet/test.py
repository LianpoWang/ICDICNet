# deepDIC/scripts/test.py

import os
import math
import torch
import data, util
from ICDICNet.model import graph
import options

# …（前面代码保持不变）…

def evaluate_model(opt, geometric, test_loader):
    """
    测试模型并计算 MSE、RMSE、Max RMSE
    """
    print(util.toMagenta("Evaluating IC-DICNet..."))

    # 定义损失函数（这里 image_loss_fun 产生的是 MSE）
    image_loss_fun = torch.nn.MSELoss(reduction='mean')

    image_losses = []  # 存放每个 batch 的 MSE
    rmse_values  = []  # 存放每个 batch 的 RMSE

    for i_batch, (ref_image, tar_image, affine_params) in enumerate(test_loader):
        ref_image = ref_image.to(opt.device, non_blocking=True)
        tar_image = tar_image.to(opt.device, non_blocking=True)

        # 初始化扰动参数
        pInit = data.genPerturbations(opt).to(opt.device)

        # 前向传播
        with torch.no_grad():
            imageWarpAll, _ = geometric(opt, tar_image, ref_image, pInit)

        # 取最后一次迭代的变形图像
        imageWarp = imageWarpAll[-1]

        # 计算本 batch 的 MSE
        mse = image_loss_fun(imageWarp, ref_image).item()
        rmse = math.sqrt(mse)

        image_losses.append(mse)
        rmse_values.append(rmse)

        # 简化进度打印
        if (i_batch + 1) % max(1, len(test_loader)//10) == 0 or (i_batch + 1) == len(test_loader):
            print(f"Batch {i_batch+1}/{len(test_loader)} | MSE: {mse:.6f} | RMSE: {rmse:.6f}")

    # 计算三种指标
    avg_mse     = sum(image_losses) / len(image_losses)
    avg_rmse    = math.sqrt(avg_mse)
    max_rmse    = max(rmse_values)

    print(util.toYellow(f"\nAverage MSE:  {avg_mse:.6f}"))
    print(util.toYellow(f"Average RMSE: {avg_rmse:.6f}"))
    print(util.toYellow(f"Max RMSE:     {max_rmse:.6f}\n"))

    return avg_mse, avg_rmse, max_rmse


def save_evaluation_results(opt, subset_name, avg_mse, avg_rmse, max_rmse):
    """
    保存评估结果到指定的目录
    """
    results_dir = os.path.join(opt.results_dir, f"R_Test/R_{subset_name}")
    util.mkdir(results_dir)

    fn = os.path.join(results_dir, f"{opt.model}_evaluation_results.txt")
    with open(fn, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Average MSE:  {avg_mse:.6f}\n")
        f.write(f"Average RMSE: {avg_rmse:.6f}\n")
        f.write(f"Max RMSE:     {max_rmse:.6f}\n")

    print(util.toGreen(f"Evaluation results saved to {fn}"))


def test():
    opt = options.set(training=False)
    # …（加载模型、test_loader）…

    # 测试并获取三个指标
    avg_mse, avg_rmse, max_rmse = evaluate_model(opt, geometric, test_loader)

    # 保存到文件
    save_evaluation_results(opt, opt.dataset_type, avg_mse, avg_rmse, max_rmse)

if __name__ == '__main__':
    test()
