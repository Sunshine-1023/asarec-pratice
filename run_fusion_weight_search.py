"""Step 5/6 — Valid-set grid search for weighted fusion channel weights. Writes outputs/evaluation/best_fusion_weights.json. Run before test offline_eval."""  # 步骤 5/6：在 valid 集上网格搜索融合通道权重

from src.evaluate.weight_search import main  # 导入融合权重搜索主函数


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用权重搜索主函数
