"""Tests for streaming data manifests."""  # 数据清单测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

from src.data.manifest import (  # 清单工具
    build_manifest,  # 组装
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
