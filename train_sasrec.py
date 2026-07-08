"""Compatibility entrypoint for RecBole SASRec/SASRecF training."""  # 兼容入口：训练 SASRec / SASRecF

from run_sasrec import main  # 复用现有训练主逻辑


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用训练入口
