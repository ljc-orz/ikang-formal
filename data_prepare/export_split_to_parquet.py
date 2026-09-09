#!/usr/bin/env python3
"""将划分后的 MySQL 数据表流式导出为固定字段结构的 Parquet 文件。"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pymysql
from pymysql.cursors import SSDictCursor

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "datas",
    "charset": "utf8mb4",
    "unix_socket": "/DaTa/mysql/mysql.sock",
    "local_infile": True,
    "autocommit": False,
}

DEFAULT_TABLE = "result_merged_wide_split_50000"
DEFAULT_OUTPUT = "result_merged_wide_split_50000.parquet"
DEFAULT_BATCH_SIZE = 5_000

FIELDS = [
    "uid",
    "id",
    "hospid",
    "usex",
    "examage",
    "image_path_1",
    "image_path_2",
    "result_alt",
    "result_bmi",
    "result_fbg",
    "result_hb",
    "result_hba1c",
    "result_hct",
    "result_hdl_c",
    "result_rbc",
    "result_scr",
    "result_tg",
    "result_wbc",
    "split",
]

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("uid", pa.string()),
        pa.field("id", pa.string()),
        pa.field("hospid", pa.int64()),
        pa.field("usex", pa.string()),
        pa.field("examage", pa.float64()),
        pa.field("image_path_1", pa.string()),
        pa.field("image_path_2", pa.string()),
        pa.field("result_alt", pa.int8()),
        pa.field("result_bmi", pa.int8()),
        pa.field("result_fbg", pa.int8()),
        pa.field("result_hb", pa.int8()),
        pa.field("result_hba1c", pa.int8()),
        pa.field("result_hct", pa.int8()),
        pa.field("result_hdl_c", pa.int8()),
        pa.field("result_rbc", pa.int8()),
        pa.field("result_scr", pa.int8()),
        pa.field("result_tg", pa.int8()),
        pa.field("result_wbc", pa.int8()),
        pa.field("split", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"要导出的 MySQL 表名（默认：{DEFAULT_TABLE}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"输出 Parquet 路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每批读取行数（默认：{DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=50_000,
        help="期望导出的行数；设为 0 则不校验（默认：50000）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出文件",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.table):
        raise ValueError(f"非法 MySQL 表名：{args.table!r}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.expected_rows < 0:
        raise ValueError("--expected-rows 不能小于 0")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{args.output}。如需覆盖，请增加 --overwrite。"
        )


def iter_mysql_batches(
    conn: pymysql.connections.Connection,
    table: str,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """使用服务端游标逐批读取，避免一次性载入全部结果。"""
    field_sql = ", ".join(f"`{field}`" for field in FIELDS)
    sql = f"SELECT {field_sql} FROM `{table}` ORDER BY uid, id"
    with conn.cursor() as cursor:
        cursor.execute(sql)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield list(rows)


def export_parquet(
    table: str,
    output: Path,
    batch_size: int,
    expected_rows: int,
) -> int:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    conn = pymysql.connect(cursorclass=SSDictCursor, **DB_CONFIG)
    writer: pq.ParquetWriter | None = None
    exported_rows = 0
    try:
        writer = pq.ParquetWriter(
            temp_path,
            PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=["usex", "split"],
            write_statistics=True,
        )
        for rows in iter_mysql_batches(conn, table, batch_size):
            arrow_table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
            writer.write_table(arrow_table, row_group_size=batch_size)
            exported_rows += len(rows)
            print(f"\r已导出 {exported_rows} 行", end="", flush=True)
        print()

        writer.close()
        writer = None

        if expected_rows and exported_rows != expected_rows:
            raise RuntimeError(
                f"导出行数为 {exported_rows}，不等于期望的 {expected_rows}；"
                "临时文件不会作为最终结果保留"
            )

        metadata = pq.read_metadata(temp_path)
        if metadata.num_rows != exported_rows:
            raise RuntimeError(
                f"Parquet 元数据行数 {metadata.num_rows} 与读取行数 {exported_rows} 不一致"
            )

        os.replace(temp_path, output)
        return exported_rows
    except Exception:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()


def print_verification(output: Path) -> None:
    parquet_file = pq.ParquetFile(output)
    first_record = parquet_file.read_row_group(0).slice(0, 1).to_pylist()[0]
    print(f"文件：{output.resolve()}")
    print(f"行数：{parquet_file.metadata.num_rows}")
    print(f"Row groups：{parquet_file.metadata.num_row_groups}")
    print("第一条记录：")
    print(first_record)


def main() -> None:
    args = parse_args()
    validate_args(args)
    exported_rows = export_parquet(
        table=args.table,
        output=args.output,
        batch_size=args.batch_size,
        expected_rows=args.expected_rows,
    )
    print(f"导出完成，共 {exported_rows} 行。")
    print_verification(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
