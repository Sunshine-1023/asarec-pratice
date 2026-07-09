"""Step 6/6 — Four-channel fusion + offline MAP@12 evaluation. Examples: python run_offline_eval.py --eval-split valid; python run_offline_eval.py --eval-split test --weights-json outputs/evaluation/best_fusion_weights.json"""  # 步骤 6/6：四路融合离线 MAP@12 评估

from src.evaluate.offline_eval import main  # 导入离线融合评估主函数


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用离线评估主函数
