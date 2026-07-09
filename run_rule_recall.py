"""Step 4/6 (optional) — Export Popular / Category Popular / Item2Item recall CSV. Not required for offline_eval (computed on the fly). Use for debugging."""  # 步骤 4/6（可选）：导出规则三路召回 CSV，供调试使用

from src.recall.rule_recall_export import main  # 导入规则召回导出主函数


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用规则召回导出主函数
