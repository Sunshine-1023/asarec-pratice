"""Run-scoped artifact paths for reproducible experiments."""  # 单次实验隔离的产物路径

from __future__ import annotations  # 延迟注解

from dataclasses import dataclass  # 路径集合
from pathlib import Path  # 路径


@dataclass(frozen=True, slots=True)  # 不可变路径契约
class RunArtifacts:  # 一次实验的全部产物目录
    profile: str  # baseline / industrial 顶层命名空间
    run_id: str  # 唯一运行 ID
    root: Path  # outputs/runs/<profile>/<run_id>

    @classmethod
    def from_root(
        cls,
        output_root: str | Path,
        run_id: str,
        *,
        profile: str = "baseline",
    ) -> "RunArtifacts":  # 构造路径但不创建目录
        clean_run_id = str(run_id).strip()  # 清洗运行 ID
        if not clean_run_id:  # 禁止空目录名
            raise ValueError("run_id must not be empty")  # 抛错
        if clean_run_id in {".", ".."} or "/" in clean_run_id or "\\" in clean_run_id:
            raise ValueError("run_id must be a single path-safe name")
        clean_profile = str(profile).strip().lower()
        if clean_profile not in {"baseline", "industrial"}:
            raise ValueError("profile must be 'baseline' or 'industrial'")
        return cls(
            profile=clean_profile,
            run_id=clean_run_id,
            root=Path(output_root) / clean_profile / clean_run_id,
        )  # 返回路径集合

    @property
    def data(self) -> Path:  # 本次运行的处理后数据，避免覆盖全局 data/processed
        return self.root / "data"  # 目录

    @property
    def manifest(self) -> Path:  # 运行 manifest
        return self.root / "manifest.json"  # 文件路径

    @property
    def resolved_config(self) -> Path:  # 冻结后的配置
        return self.root / "resolved_config.json"  # 文件路径

    @property
    def checkpoints(self) -> Path:  # 模型权重目录
        return self.root / "checkpoints"  # 目录

    @property
    def recall(self) -> Path:  # 各通道召回目录
        return self.root / "recall"  # 目录

    @property
    def candidates(self) -> Path:  # 候选并集目录
        return self.root / "candidates"  # 目录

    @property
    def ranking(self) -> Path:  # 排序模型与特征目录
        return self.root / "ranking"  # 目录

    @property
    def evaluation(self) -> Path:  # 指标与报告目录
        return self.root / "evaluation"  # 目录

    @property
    def logs(self) -> Path:  # 日志目录
        return self.root / "logs"  # 目录

    def ensure_directories(self) -> None:  # 显式创建所有运行目录
        for path in (  # 遍历目录
            self.root,  # 根目录
            self.data,  # 处理后数据
            self.checkpoints,  # checkpoint
            self.recall,  # recall
            self.candidates,  # candidates
            self.ranking,  # ranking
            self.evaluation,  # evaluation
            self.logs,  # logs
        ):  # 目录结束
            path.mkdir(parents=True, exist_ok=True)  # 创建目录

    def recall_file(self, channel: str, split: str, suffix: str = ".csv") -> Path:  # 某通道召回文件
        return self.recall / f"{str(channel).strip().lower()}_{str(split).strip().lower()}{suffix}"  # 路径

    def candidate_file(self, split: str, suffix: str = ".csv") -> Path:  # 候选并集文件
        return self.candidates / f"{str(split).strip().lower()}{suffix}"  # 路径

    def ranking_table_file(self, split: str) -> Path:  # LambdaRank 训练/推理表
        return self.ranking / f"{str(split).strip().lower()}.parquet"

    def ranker_dir(self) -> Path:  # 学习排序 artifact
        return self.ranking / "lambdarank"

    def ranker_scored_file(self, split: str) -> Path:  # 打分后的候选
        return self.ranking / f"{str(split).strip().lower()}_scored.csv"

    def metrics_file(self, split: str) -> Path:  # 指标文件
        return self.evaluation / f"{str(split).strip().lower()}_metrics.json"  # 路径

    def selected_checkpoint_file(self, model: str) -> Path:
        return self.checkpoints / f"{str(model).strip().lower()}_selected.pth"

    def checkpoint_selection_file(self, model: str) -> Path:
        return self.evaluation / f"{str(model).strip().lower()}_checkpoint_selection.json"
