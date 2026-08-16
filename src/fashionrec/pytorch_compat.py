"""Compatibility helpers for RecBole with newer PyTorch/NumPy."""  # RecBole 与新版本 PyTorch/NumPy 的兼容辅助模块

import numpy as np  # 导入 NumPy 数值计算库
import torch  # 导入 PyTorch 深度学习框架


def patch_numpy_for_recbole() -> None:  # 为 RecBole 恢复 NumPy 2.x 中已移除的类型别名
    """Restore NumPy aliases removed in 2.x that RecBole 1.2.1 still uses."""  # 恢复 RecBole 1.2.1 仍依赖的 NumPy 别名
    if getattr(patch_numpy_for_recbole, "_applied", False):  # 若补丁已应用则跳过
        return  # 直接返回避免重复打补丁

    np.float = np.float64  # 将 np.float 映射到 float64
    np.int = np.int64  # 将 np.int 映射到 int64
    np.long = np.int64  # 将 np.long 映射到 int64（NumPy 2.x 已移除）
    np.bool = np.bool_  # 将 np.bool 映射到 bool_
    np.object = object  # 将 np.object 映射到内置 object
    np.str = str  # 将 np.str 映射到内置 str
    patch_numpy_for_recbole._applied = True  # 标记 NumPy 补丁已应用


def patch_torch_load_for_recbole() -> None:  # 修补 torch.load 以兼容 RecBole 检查点加载
    """RecBole 检查点含优化器/配置对象；PyTorch 2.6+ 默认 weights_only=True 会导致加载失败。"""  # 函数说明文档
    if getattr(torch.load, "_recbole_compat_patch_applied", False):  # 若 torch.load 补丁已应用则跳过
        return  # 直接返回避免重复打补丁

    original_load = torch.load  # 保存原始 torch.load 函数引用

    def _patched_load(*args, **kwargs):  # 定义包装后的 load 函数
        kwargs.setdefault("weights_only", False)  # 默认允许加载非张量对象
        return original_load(*args, **kwargs)  # 调用原始 load 并返回结果

    _patched_load._recbole_compat_patch_applied = True  # 标记包装函数已打补丁
    torch.load = _patched_load  # 用包装函数替换全局 torch.load


def patch_recbole_compat() -> None:  # 一次性应用所有 RecBole 兼容补丁
    patch_numpy_for_recbole()  # 应用 NumPy 别名补丁
    patch_torch_load_for_recbole()  # 应用 torch.load 补丁
