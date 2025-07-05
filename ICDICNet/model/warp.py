import torch
import numpy as np

# 仿射变换参数拟合
def fit(Xsrc, Xdst):
    """
    根据源点和目标点计算仿射变换矩阵。
    """
    ptsN = len(Xsrc)
    X, Y, U, V, O, I = Xsrc[:, 0], Xsrc[:, 1], Xdst[:, 0], Xdst[:, 1], np.zeros([ptsN]), np.ones([ptsN])
    A = np.concatenate((np.stack([X, Y, I, O, O, O], axis=1),
                        np.stack([O, O, O, X, Y, I], axis=1)), axis=0)
    b = np.concatenate((U, V), axis=0)
    p1, p2, p3, p4, p5, p6 = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    pMtrx = np.array([[p1, p2, p3], [p4, p5, p6], [0, 0, 1]], dtype=np.float32)
    return pMtrx

# compose 函数，支持高精度和稳定性
def compose(opt, p, dp):
    """
    根据公式 W(ξ; p) = W(ξ; p) · W⁻¹(ξ; Δp) 计算新的形变参数。
    """
    pMtrx = vec2mtrx(opt, p)  # 当前形变矩阵
    dpMtrx = vec2mtrx(opt, dp)  # 增量形变矩阵

    # 计算增量矩阵的逆
    dpMtrx_inv = torch.linalg.inv(dpMtrx)  # 使用高精度矩阵求逆
    pMtrxNew = torch.matmul(pMtrx, dpMtrx_inv)  # 组合更新

    # 对新矩阵进行归一化（避免数值不稳定）
    pMtrxNew = pMtrxNew / pMtrxNew[:, 2:3, 2:3]
    pNew = mtrx2vec(opt, pMtrxNew)  # 转换回参数形式
    return pNew

# 将变换参数转换为仿射矩阵
def vec2mtrx(opt, p):
    """
    将仿射参数向量转换为矩阵形式。
    """
    O = torch.zeros(opt.batchSize, dtype=torch.float32).to(opt.device)
    I = torch.ones(opt.batchSize, dtype=torch.float32).to(opt.device)

    p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
    pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                         torch.stack([p4, I + p5, p6], dim=-1),
                         torch.stack([O, O, I], dim=-1)], dim=1)
    return pMtrx

# 将仿射矩阵转换为变换参数
def mtrx2vec(opt, pMtrx):
    """
    将仿射矩阵转换回参数向量形式。
    """
    [row0, row1, _] = torch.unbind(pMtrx, dim=1)
    [e00, e01, e02] = torch.unbind(row0, dim=1)
    [e10, e11, e12] = torch.unbind(row1, dim=1)
    p = torch.stack([e00 - 1, e01, e02, e10, e11 - 1, e12], dim=1)
    return p

# 对图像应用仿射变换
def transformImage(opt, image, pMtrx):
    """
    根据仿射矩阵对图像进行变形。
    """
    refMtrx = torch.from_numpy(opt.refMtrx).to(opt.device)
    refMtrx = refMtrx.repeat(opt.batchSize, 1, 1)  # 扩展到 batch 大小
    transMtrx = torch.matmul(refMtrx, pMtrx)  # 计算总的变换矩阵

    # 构建网格并进行变换
    X, Y = np.meshgrid(np.linspace(-1, 1, opt.W), np.linspace(-1, 1, opt.H))
    X, Y = X.flatten(), Y.flatten()
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
    XYhom = np.tile(XYhom, [opt.batchSize, 1, 1]).astype(np.float32)
    XYhom = torch.from_numpy(XYhom).to(opt.device)

    XYwarpHom = torch.matmul(transMtrx, XYhom)  # 计算变换后的坐标
    XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom, dim=1)
    Xwarp = (XwarpHom / (ZwarpHom + 1e-8)).reshape(opt.batchSize, opt.H, opt.W)
    Ywarp = (YwarpHom / (ZwarpHom + 1e-8)).reshape(opt.batchSize, opt.H, opt.W)
    grid = torch.stack([Xwarp, Ywarp], dim=-1)

    # 使用高精度双三次插值
    imageWarp = torch.nn.functional.grid_sample(image, grid, mode="bicubic", align_corners=True)
    return imageWarp
