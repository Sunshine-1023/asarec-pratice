"""Streaming data snapshot manifest for reproducible experiments."""  # 可复现实验的流式数据快照清单

from __future__ import annotations  # 启用延迟注解

import csv  # 流式读取 TSV/CSV
import hashlib  # SHA256
import json  # 写出 JSON
import subprocess  # 只读获取 Git SHA
from datetime import datetime, timezone  # 生成时间
from pathlib import Path  # 路径类型
from typing import Any, Iterable  # 类型注解


CHUNK_SIZE = 1024 * 1024  # 流式哈希分块大小（1MB）
INTER_USER_COL = "user_id:token"  # RecBole 交互用户列
INTER_ITEM_COL = "item_id:token"  # RecBole 交互商品列
INTER_TIME_COL = "timestamp:float"  # RecBole 交互时间戳列


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:  # 流式计算文件 SHA256
    digest = hashlib.sha256()  # 创建哈希器
    with path.open("rb") as handle:  # 二进制打开，避免整文件读入内存
        while True:  # 循环读块
            chunk = handle.read(chunk_size)  # 读取一块
            if not chunk:  # 读完
                break  # 退出循环
            digest.update(chunk)  # 更新哈希
    return digest.hexdigest()  # 返回十六进制摘要


def read_git_sha(repo_root: Path | None = None) -> str | None:  # 只读获取当前提交 SHA
    cwd = repo_root or Path.cwd()  # 仓库根目录
    try:  # Git 可能不存在或不是仓库
        result = subprocess.run(  # 只读查询 HEAD
            ["git", "rev-parse", "HEAD"],  # 不修改 Git 状态
            cwd=cwd,  # 在仓库根目录执行
            check=False,  # 失败时不抛异常
            capture_output=True,  # 捕获输出
            text=True,  # 文本模式
        )  # 子进程结束
    except OSError:  # 系统没有 git 可执行文件
        return None  # 无法获取 SHA
    if result.returncode != 0:  # git 命令失败
        return None  # 返回空
    sha = result.stdout.strip()  # 去掉换行
    return sha or None  # 空字符串视为缺失


def _iso_from_unix(ts: float) -> str:  # Unix 秒时间戳转 UTC ISO 日期时间
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()  # 转为带时区的 ISO 字符串


def stream_inter_stats(path: Path) -> dict[str, Any]:  # 流式统计 RecBole .inter 文件
    n_rows = 0  # 数据行数（不含表头）
    users: set[str] = set()  # 去重用户
    items: set[str] = set()  # 去重商品
    min_ts: float | None = None  # 最小时间戳
    max_ts: float | None = None  # 最大时间戳
    with path.open("r", encoding="utf-8", newline="") as handle:  # 文本流式打开
        reader = csv.DictReader(handle, delimiter="\t")  # TSV 字典读取器
        if reader.fieldnames is None:  # 空文件或无表头
            raise ValueError(f"Missing header in {path}")  # 无法统计
        required = {INTER_USER_COL, INTER_ITEM_COL, INTER_TIME_COL}  # 必需列
        missing = required.difference(reader.fieldnames)  # 缺列
        if missing:  # 有缺列
            raise ValueError(f"{path} missing columns: {sorted(missing)}")  # 抛出缺列
        for row in reader:  # 逐行扫描，不把整表载入 pandas
            n_rows += 1  # 行数加一
            users.add(str(row[INTER_USER_COL]))  # 记录用户
            items.add(str(row[INTER_ITEM_COL]))  # 记录商品
            ts = float(row[INTER_TIME_COL])  # 解析时间戳
            if min_ts is None or ts < min_ts:  # 更新最小时间
                min_ts = ts  # 保存最小值
            if max_ts is None or ts > max_ts:  # 更新最大时间
                max_ts = ts  # 保存最大值
    return {  # 汇总统计
        "n_rows": n_rows,  # 行数
        "n_users": len(users),  # 用户数
        "n_items": len(items),  # 商品数
        "min_timestamp": min_ts,  # 最小 Unix 时间戳
        "max_timestamp": max_ts,  # 最大 Unix 时间戳
        "min_datetime": _iso_from_unix(min_ts) if min_ts is not None else None,  # 最小时间 ISO
        "max_datetime": _iso_from_unix(max_ts) if max_ts is not None else None,  # 最大时间 ISO
    }  # 统计字典结束


def describe_file(path: Path, collect_inter_stats: bool = False) -> dict[str, Any]:  # 描述单个数据文件
    resolved = path.resolve()  # 绝对路径
    info: dict[str, Any] = {  # 基础信息
        "path": str(path),  # 原始相对/给定路径
        "resolved_path": str(resolved),  # 绝对路径
        "exists": resolved.exists(),  # 是否存在
    }  # 基础信息结束
    if not resolved.exists() or not resolved.is_file():  # 不存在或不是文件
        return info  # 只返回存在性
    info["nbytes"] = resolved.stat().st_size  # 字节数
    info["sha256"] = sha256_file(resolved)  # 流式哈希
    if collect_inter_stats:  # 需要交互统计时
        info.update(stream_inter_stats(resolved))  # 合并行数/用户/时间范围
    return info  # 返回文件描述


def build_manifest(  # 组装实验数据清单
    files: dict[str, Path],  # 逻辑名到路径
    inter_files: Iterable[str] | None = None,  # 需要对交互做统计的逻辑名
    split_bounds: dict[str, Any] | None = None,  # train/valid/test 时间边界
    preprocess: dict[str, Any] | None = None,  # 预处理参数
    repo_root: Path | None = None,  # Git 仓库根目录
    generated_at: str | None = None,  # 可注入生成时间以便测试
) -> dict[str, Any]:  # 返回清单字典
    inter_names = set(inter_files or ())  # 需要统计的交互文件名
    file_payload = {  # 逐文件描述
        name: describe_file(path, collect_inter_stats=name in inter_names)  # 交互文件额外统计
        for name, path in files.items()  # 遍历全部文件
    }  # 文件描述结束
    return {  # 完整清单
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),  # 生成时间
        "git_sha": read_git_sha(repo_root),  # 只读 Git SHA
        "preprocess": preprocess or {},  # 预处理参数
        "split_bounds": split_bounds or {},  # 时间切分边界
        "files": file_payload,  # 各文件快照
    }  # 清单结束


def canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:  # 去掉生成时间后用于稳定性比较
    copied = json.loads(json.dumps(payload))  # 深拷贝为纯 JSON 类型
    copied.pop("generated_at", None)  # 生成时间允许变化
    return copied  # 返回可比较内容


def write_manifest(payload: dict[str, Any], output_path: Path) -> Path:  # 将清单写入 JSON
    output_path = Path(output_path)  # 规范化路径
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写 JSON
    return output_path  # 返回写出路径


def build_processed_hm_manifest(  # 针对当前 hm 处理后数据生成清单
    processed_dir: Path | None = None,  # data/processed 目录
    raw_transactions: Path | None = None,  # 原始或过滤后交易文件
    preprocess: dict[str, Any] | None = None,  # 预处理参数
    split_bounds: dict[str, Any] | None = None,  # 切分边界
    repo_root: Path | None = None,  # 仓库根
) -> dict[str, Any]:  # 返回清单
    processed_dir = processed_dir or Path("data/processed")  # 默认处理后目录
    hm_dir = processed_dir / "hm"  # hm 交互目录
    files = {  # 需要纳入快照的文件
        "raw_or_filtered_transactions": raw_transactions or Path("data/raw/filtered/transactions_train.csv"),  # 输入交易
        "hm.inter": hm_dir / "hm.inter",  # 全量交互
        "hm.train.inter": hm_dir / "hm.train.inter",  # 训练集
        "hm.valid.inter": hm_dir / "hm.valid.inter",  # 验证集
        "hm.test.inter": hm_dir / "hm.test.inter",  # 测试集
    }  # 文件映射结束
    inter_names = ["hm.inter", "hm.train.inter", "hm.valid.inter", "hm.test.inter"]  # 需要流式统计的交互
    return build_manifest(  # 组装清单
        files=files,  # 文件映射
        inter_files=inter_names,  # 交互统计
        split_bounds=split_bounds,  # 时间边界
        preprocess=preprocess,  # 预处理参数
        repo_root=repo_root,  # 仓库根
    )  # 清单组装结束
