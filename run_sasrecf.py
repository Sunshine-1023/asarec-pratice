"""Convenience entrypoint for training SASRecF with default config."""  # 使用默认配置训练 SASRecF 的便捷入口脚本

from __future__ import annotations  # 启用延迟注解评估

import sys  # 导入系统模块以访问命令行参数

from run_sasrec import main  # 导入 SASRec 训练主函数


def _ensure_default_sasrecf_config() -> None:  # 在用户未指定时注入默认 sasrecf 配置
    """Inject sasrecf config when user does not pass --config."""  # 未传 --config 时自动追加 sasrecf 配置文件
    if "--config" in sys.argv:  # 若命令行已包含 --config 参数
        return  # 不修改参数直接返回
    sys.argv.extend(["--config", "configs/sasrecf.yaml"])  # 追加默认配置文件路径


if __name__ == "__main__":  # 脚本直接运行时
    _ensure_default_sasrecf_config()  # 确保使用 sasrecf 默认配置
    main()  # 调用 SASRec 训练主函数
