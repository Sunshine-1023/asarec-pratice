"""Resolved experiment context shared by every pipeline stage."""  # 全阶段共享的运行上下文

from __future__ import annotations  # 延迟注解

import hashlib  # 配置内容哈希
import json  # 冻结配置
from dataclasses import asdict, dataclass  # 上下文和配置序列化
from datetime import datetime, timezone  # 默认运行 ID
from pathlib import Path  # 路径

from fashionrec.experiment.artifacts import RunArtifacts  # 运行产物路径
from fashionrec.experiment.config import ExperimentConfig, load_experiment_config  # 实验配置


PIPELINE_PROFILES = ("baseline", "industrial")


class RunConfigurationConflictError(RuntimeError):
    """Raised when an existing run directory belongs to another configuration."""


def profile_for_config(config: ExperimentConfig) -> str:
    return "industrial" if config.ranking.enabled else "baseline"


def make_run_id(experiment_name: str, now: datetime | None = None) -> str:  # 生成稳定可读的运行 ID
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")  # 微秒避免碰撞
    return f"{str(experiment_name).strip()}_{stamp}"  # 名称与时间戳


@dataclass(frozen=True, slots=True)  # 不可变运行上下文
class RunContext:  # 一次正式流水线的共享上下文
    config: ExperimentConfig  # 已解析实验配置
    artifacts: RunArtifacts  # 本次产物路径
    strict: bool = True  # 正式实验默认禁止静默回退

    @property
    def run_id(self) -> str:  # 便捷访问运行 ID
        return self.artifacts.run_id  # 返回 ID

    @property
    def config_sha256(self) -> str:  # 配置文件哈希
        return hashlib.sha256(self.config.source_path.read_bytes()).hexdigest()  # SHA256

    def resolved_payload(self) -> dict[str, object]:  # 可序列化的冻结运行配置
        payload = asdict(self.config)  # 数据类转字典
        payload["source_path"] = str(self.config.source_path)  # Path 转字符串
        payload["run_id"] = self.run_id  # 运行 ID
        payload["profile"] = self.artifacts.profile
        payload["strict"] = self.strict  # 严格模式
        payload["config_sha256"] = self.config_sha256  # 配置哈希
        return json.loads(json.dumps(payload, default=str))  # 元组等转为 JSON 可写类型

    def initialize(self) -> None:  # 创建目录并冻结配置
        if self.artifacts.resolved_config.exists():
            try:
                existing = json.loads(self.artifacts.resolved_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunConfigurationConflictError(
                    f"cannot safely resume run with unreadable config: {self.artifacts.resolved_config}"
                ) from exc
            expected = self.config_sha256
            existing_hash = str(existing.get("config_sha256", ""))
            existing_profile = str(existing.get("profile", self.artifacts.profile))
            if existing_hash != expected or existing_profile != self.artifacts.profile:
                raise RunConfigurationConflictError(
                    "run-id already exists with a different profile or experiment config: "
                    f"{self.artifacts.root}"
                )
        elif self.artifacts.root.exists() and any(self.artifacts.root.iterdir()):
            raise RunConfigurationConflictError(
                f"run directory is non-empty but has no resolved config: {self.artifacts.root}"
            )
        self.artifacts.ensure_directories()  # 创建目录
        self.artifacts.resolved_config.write_text(  # 写配置
            json.dumps(self.resolved_payload(), ensure_ascii=False, indent=2),  # JSON
            encoding="utf-8",  # 编码
        )  # 写出结束

    def write_manifest(self, *, status: str, completed_steps: list[str]) -> None:  # 写运行状态清单
        self.artifacts.manifest.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "profile": self.artifacts.profile,
                    "status": str(status),
                    "strict": self.strict,
                    "config_sha256": self.config_sha256,
                    "completed_steps": list(completed_steps),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )



def create_run_context(  # 从配置创建一次运行上下文
    config_path: str | Path = "configs/baseline/experiment.yaml",  # 实验配置
    output_root: str | Path = "outputs/runs",  # 运行产物根目录
    run_id: str | None = None,  # 可指定运行 ID 以恢复
    profile: str | None = None,  # 显式 baseline / industrial；默认由配置推导
    strict: bool = True,  # 是否严格依赖
    initialize: bool = False,  # 是否立即创建目录
) -> RunContext:  # 返回上下文
    config = load_experiment_config(config_path)  # 加载配置
    configured_profile = profile_for_config(config)
    requested_profile = configured_profile if profile in {None, "auto"} else str(profile).strip().lower()
    if requested_profile not in PIPELINE_PROFILES:
        raise ValueError(f"profile must be one of {PIPELINE_PROFILES} or 'auto'")
    if requested_profile != configured_profile:
        raise ValueError(
            f"profile/config mismatch: profile={requested_profile}, "
            f"ranking.enabled={config.ranking.enabled} requires profile={configured_profile}"
        )
    resolved_run_id = run_id or make_run_id(config.experiment.name)  # 生成或复用 ID
    context = RunContext(  # 创建上下文
        config=config,  # 配置
        artifacts=RunArtifacts.from_root(output_root, resolved_run_id, profile=requested_profile),  # 路径
        strict=strict,  # 严格模式
    )  # 上下文结束
    if initialize:  # 调用方明确要求落盘
        context.initialize()  # 创建目录并冻结配置
    return context  # 返回上下文
