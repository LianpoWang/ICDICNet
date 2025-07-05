import numpy as np
import os
import cv2
from multiprocessing import Pool
from functools import partial

# === 用户配置区 ===
config = {
    "train_ref_path": r"D:\work-one\Procedure\BSpeckleRender\Train_References",  # 原始参考图像路径
    "output_base": r"D:\work-one\TotalData\dataset\BigData\data1",  # 生成数据存储根目录
    "num_processes": 4,  # 并行进程数
    "total_images": 200 # 总图像数量
}


# ==================

def gaussian_smoothing1(x, y):
    # 参数设置
    tx, ty = np.random.uniform(-0.1, 0.1, 2)
    kx, ky = np.random.uniform(0.99, 1.01, 2)
    theta = np.random.uniform(-0.005, 0.005)
    rx, ry = np.random.uniform(-0.01, 0.01, 2)

    # 高斯核生成
    x0, x1 = np.random.randint(1, 255, 2)
    y0, y1 = np.random.randint(1, 255, 2)
    ux_g, uy_g = 0, 0

    for _ in range(2):
        Ax, Ay = np.random.uniform(0.001, 0.006, 2)
        sigma_x0, sigma_x1 = np.random.uniform(0.01, 0.05, 2)
        sigma_y0, sigma_y1 = np.random.uniform(0.01, 0.05, 2)

        p = -0.5 * ((x - x0) / sigma_x0) ** 2 - 0.5 * ((y - y0) / sigma_y0) ** 2
        q = -0.5 * ((x - x1) / sigma_x1) ** 2 - 0.5 * ((y - y1) / sigma_y1) ** 2
        ux_g += Ax * np.exp(p)
        uy_g += Ay * np.exp(q)

    # 几何变换
    u = np.cos(theta) * ((kx - 1) * x + rx * y + ux_g) + np.sin(theta) * (ry * x + (ky - 1) * y + uy_g) + tx
    v = -np.sin(theta) * ((kx - 1) * x + rx * y + ux_g) + np.cos(theta) * (ry * x + (ky - 1) * y + uy_g) + ty

    # 归一化
    u /= max(1, np.max(np.abs(u)))
    v /= max(1, np.max(np.abs(v)))

    return x + u, y + v, u, v

def validate_paths():
    """路径校验与目录创建"""
    required_paths = [
        config["train_ref_path"],
        os.path.join(config["output_base"], "img/ref"),
        os.path.join(config["output_base"], "Displacement/x")
    ]
    for path in required_paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"目录已创建: {path}")


def generate_deformed_image(img_idx):
    """处理单个图像"""
    try:
        # 输入路径
        ref_path = os.path.join(config["train_ref_path"], f"Ref{img_idx}.tif")
        if not os.path.exists(ref_path):
            print(f"警告: 缺失文件 {ref_path}")
            return

        # 输出路径
        save_paths = {
            'ref': os.path.join(config["output_base"], f"img/ref/Ref{img_idx}.jpg"),
            'def': os.path.join(config["output_base"], f"img/def/Def{img_idx}.jpg"),
            'x': os.path.join(config["output_base"], f"Displacement/x/Dispx{img_idx}.csv"),
            'y': os.path.join(config["output_base"], f"Displacement/y/Dispy{img_idx}.csv")
        }

        # 生成逻辑
        img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = img_ref.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        _, _, disp_x, disp_y = gaussian_smoothing1(xx, yy)

        # 边缘归零处理
        disp_x[:2, :] = disp_x[-2:, :] = disp_x[:, :2] = disp_x[:, -2:] = 0
        disp_y[:2, :] = disp_y[-2:, :] = disp_y[:, :2] = disp_y[:, -2:] = 0

        # 保存结果
        cv2.imwrite(save_paths['ref'], img_ref)
        cv2.imwrite(save_paths['def'], cv2.remap(img_ref, xx + disp_x, yy + disp_y, cv2.INTER_CUBIC))
        np.savetxt(save_paths['x'], disp_x, delimiter=',', fmt='%.3f')
        np.savetxt(save_paths['y'], disp_y, delimiter=',', fmt='%.3f')

    except Exception as e:
        print(f"处理图像 {img_idx} 时发生错误: {str(e)}")


if __name__ == '__main__':
    validate_paths()
    with Pool(processes=config["num_processes"]) as pool:
        pool.map(generate_deformed_image, range(1, config["total_images"] + 1))