"""Tests for streaming data manifests."""  # 数据清单测试

from __future__ import annotations  # 延迟注解

import json  # 检查清单中不引用全局旧路径
from pathlib import Path  # 路径

from fashionrec.industrial.data.manifest import (  # 清单工具
    SCHEMA_VERSION,  # 当前行级 schema
    build_manifest,  # 组装
    build_processed_hm_manifest,  # hm 处理后清单
    canonical_manifest,  # 去掉生成时间
    sha256_file,  # 哈希
    stream_inter_stats,  # 流式统计
    write_manifest,  # 写出
)  # 导入结束


def _write_inter(path: Path, rows: list[tuple[str, str, int]]) -> None:  # 写极小 TSV fixture
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    lines.extend(f"{user}\t{item}\t{ts}" for user, item, ts in rows)  # 数据行
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def test_sha256_is_stable(tmp_path: Path) -> None:  # 相同内容哈希稳定
    path = tmp_path / "a.tsv"  # 文件
    path.write_text("hello\n", encoding="utf-8")  # 写入
    first = sha256_file(path)  # 第一次
    second = sha256_file(path)  # 第二次
    assert first == second  # 必须相同
    assert len(first) == 64  # SHA256 十六进制长度


def test_stream_inter_stats_counts_and_time_range(tmp_path: Path) -> None:  # 行数、用户数、时间范围
    path = tmp_path / "tiny.inter"  # fixture
    _write_inter(  # 三行两条用户两个商品
        path,  # 路径
        [  # 行
            ("u1", "010", 1_600_000_000),  # 用户1
            ("u1", "011", 1_600_000_100),  # 用户1 第二件
            ("u2", "010", 1_600_000_200),  # 用户2
        ],  # 行结束
    )  # 写出结束
    stats = stream_inter_stats(path)  # 流式统计
    assert stats["n_rows"] == 3  # 三行
    assert stats["n_users"] == 2  # 两用户
    assert stats["n_items"] == 2  # 两商品
    assert stats["min_timestamp"] == 1_600_000_000  # 最小时间
    assert stats["max_timestamp"] == 1_600_000_200  # 最大时间


def test_manifest_stable_except_generated_at(tmp_path: Path) -> None:  # 同一输入除生成时间外一致
    train = tmp_path / "train.inter"  # 训练
    valid = tmp_path / "valid.inter"  # 验证
    _write_inter(train, [("u1", "1", 10), ("u2", "2", 20)])  # 训练行
    _write_inter(valid, [("u1", "3", 30)])  # 验证行
    files = {"hm.train.inter": train, "hm.valid.inter": valid}  # 文件映射
    bounds = {"valid_start": "2020-09-09", "test_start": "2020-09-16"}  # 边界
    preprocess = {"weeks": 6, "min_user_purchases": 5}  # 预处理
    first = build_manifest(files, inter_files=files, split_bounds=bounds, preprocess=preprocess, repo_root=tmp_path)  # 第一次
    second = build_manifest(files, inter_files=files, split_bounds=bounds, preprocess=preprocess, repo_root=tmp_path)  # 第二次
    assert first["generated_at"] != ""  # 有生成时间
    assert canonical_manifest(first) == canonical_manifest(second)  # 去掉时间后完全一致
    out = write_manifest(first, tmp_path / "manifest.json")  # 写出
    assert out.exists()  # 文件存在
    assert "sha256" in first["files"]["hm.train.inter"]  # 训练文件有哈希
    assert first["files"]["hm.train.inter"]["n_rows"] == 2  # 训练两行
    assert first["split_bounds"]["valid_start"] == "2020-09-09"  # 边界写入
    assert first["schema_version"] == SCHEMA_VERSION  # 语义版本
    assert first["dataset_version"] == second["dataset_version"]  # 协议版本稳定
    assert len(first["dataset_version"]) == 12  # 短哈希


def test_processed_hm_manifest_records_true_raw_separately(tmp_path: Path) -> None:  # raw 与本次输入分开记
    processed = tmp_path / "run-data"  # 模拟 outputs/runs/<id>/data
    hm = processed / "hm"  # hm 目录
    hm.mkdir(parents=True)  # 创建
    raw = tmp_path / "transactions_train.csv"  # 真正 raw
    filtered = tmp_path / "filtered_transactions.csv"  # 本 run filtered
    raw.write_text("customer_id,article_id,t_dat\na,1,2020-09-22\n", encoding="utf-8")  # raw 内容
    filtered.write_text("customer_id,article_id,t_dat\na,1,2020-09-22\n", encoding="utf-8")  # 相同内容不同路径
    _write_inter(hm / "hm.inter", [("u1", "1", 10)])  # 全量
    _write_inter(hm / "hm.train.inter", [("u1", "1", 10)])  # 训练
    _write_inter(hm / "hm.model_train.inter", [("u1", "1", 10)])  # 模型训练
    _write_inter(hm / "hm.valid.inter", [("u1", "2", 20)])  # 验证
    _write_inter(hm / "hm.test.inter", [("u1", "3", 30)])  # 测试
    preprocess = {"weeks": 6, "with_filter": True, "processed_dir": str(processed)}  # 协议
    bounds = {"valid_start": "2020-09-09"}  # 边界
    first = build_processed_hm_manifest(  # 第一次
        processed_dir=processed,  # 本 run 目录
        raw_transactions=filtered,  # 本次实际喂给 preprocess 的文件
        true_raw_transactions=raw,  # 真正 raw
        preprocess=preprocess,  # 参数
        split_bounds=bounds,  # 边界
        repo_root=tmp_path,  # 无 git
    )  # 第一次结束
    second = build_processed_hm_manifest(  # 第二次
        processed_dir=processed,
        raw_transactions=filtered,
        true_raw_transactions=raw,
        preprocess=preprocess,
        split_bounds=bounds,
        repo_root=tmp_path,
    )  # 第二次结束
    assert first["raw_transactions"]["path"] == str(raw)  # raw 单独记录
    assert first["files"]["run_transactions_input"]["path"] == str(filtered)  # 本次输入是 filtered
    assert first["files"]["hm.train.inter"]["n_users"] == 1  # 用户数
    assert first["files"]["hm.train.inter"]["n_items"] == 1  # SKU 数
    assert canonical_manifest(first) == canonical_manifest(second)  # 去掉 generated_at 后完全一致
    assert first["schema_version"] == SCHEMA_VERSION  # schema
    assert "data/raw/filtered" not in json.dumps(first)  # 不引用全局旧 filtered
