import numpy as np
import os
import cv2

# === 用户配置区 ===
config = {
    "input_base": r"D:\work-one\TotalData\dataset\BigData\data1",  # data_generator的输出目录
    "output_base": r"D:\TotalData\dataset\SmallData",  # 子集存储根目录
    "subset_size": 41,
    "step": 1
}


# ==================

def validate_paths():
    """确保输出目录存在"""
    os.makedirs(os.path.join(config["output_base"], "Ref_Subset/Subset41"), exist_ok=True)
    os.makedirs(os.path.join(config["output_base"], "Def_Subset/Subset41"), exist_ok=True)


def extract_subsets(img_idx):
    """提取单个图像的子集"""
    try:
        # 输入路径
        ref_img = cv2.imread(os.path.join(config["input_base"], f"img/ref/Ref{img_idx}.jpg"), 0)
        def_img = cv2.imread(os.path.join(config["input_base"], f"img/def/Def{img_idx}.jpg"), 0)
        disp_x = np.loadtxt(os.path.join(config["input_base"], f"Displacement/x/Dispx{img_idx}.csv"), delimiter=',')
        disp_y = np.loadtxt(os.path.join(config["input_base"], f"Displacement/y/Dispy{img_idx}.csv"), delimiter=',')

        # 遍历POI
        height, width = ref_img.shape
        for y in range(0, height - config["subset_size"], config["step"]):
            for x in range(0, width - config["subset_size"], config["step"]):
                # 参考子集
                ref_subset = ref_img[y:y + config["subset_size"], x:x + config["subset_size"]]

                # 变形坐标计算
                dx = disp_x[y, x]
                dy = disp_y[y, x]
                x_def = int(round(x + dx))
                y_def = int(round(y + dy))

                # 边界检查
                if (0 <= x_def <= width - config["subset_size"] and
                        0 <= y_def <= height - config["subset_size"]):
                    def_subset = def_img[y_def:y_def + config["subset_size"], x_def:x_def + config["subset_size"]]

                    # 保存子集
                    cv2.imwrite(
                        os.path.join(config["output_base"], f"Ref_Subset/Subset41/Ref_{img_idx}_{x}_{y}.jpg"),
                        ref_subset
                    )
                    cv2.imwrite(
                        os.path.join(config["output_base"], f"Def_Subset/Subset41/Def_{img_idx}_{x}_{y}.jpg"),
                        def_subset
                    )
    except Exception as e:
        print(f"处理图像 {img_idx} 时发生错误: {str(e)}")


if __name__ == '__main__':
    validate_paths()
    for img_idx in range(1, 364):  # 根据实际图像数量调整
        extract_subsets(img_idx)