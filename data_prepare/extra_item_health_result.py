import pymysql

# 数据库配置
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

# 需要处理的指标列表（排除 wt 和 ht）
METRICS = [
    "alt",
    "bmi",
    "fbg",
    "hb",
    "hba1c",
    "hct",
    "hdl_c",
    "rbc",
    "scr",
    "tg",
    "wbc",
]

# 新表名
NEW_TABLE = "result_merged_wide"

# 原表中需要保留的基础列（不含 result_*）
BASE_COLUMNS = [
    "uid",
    "id",
    "hospid",
    "regdate",
    "usex",
    "examage",
    "image_path_1",
    "image_path_2",
    "grade_1",
    "grade_2",
]

# 每批读取的行数（可根据内存大小调整）
READ_BATCH_SIZE = 20000
# 每批插入的行数（可根据事务日志大小调整）
INSERT_BATCH_SIZE = 5000


def get_db_connection():
    """创建数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def create_new_table(conn):
    """
    删除旧表（如果存在）并创建新表，主键为 (uid, id)
    """
    with conn.cursor() as cursor:
        # 删除旧表
        cursor.execute(f"DROP TABLE IF EXISTS {NEW_TABLE}")
        # 构建 CREATE TABLE 语句，字段类型与原表保持一致
        col_defs = [
            "uid char(32) NOT NULL",
            "id char(18) NOT NULL",
            "hospid bigint",
            "regdate date",
            "usex enum('MAN','WOMAN','UNK')",
            "examage double",
            "image_path_1 mediumtext",
            "image_path_2 mediumtext",
            "grade_1 enum('ERROR','GOOD','USABLE','REJECT')",
            "grade_2 enum('ERROR','GOOD','USABLE','REJECT')",
        ]
        # 添加 result_* 列（tinyint 可存储 0/1/NULL）
        for metric in METRICS:
            col_defs.append(f"result_{metric} tinyint")
        # 主键约束
        col_defs.append("PRIMARY KEY (uid, id)")

        create_sql = f"CREATE TABLE {NEW_TABLE} ({', '.join(col_defs)})"
        cursor.execute(create_sql)
        conn.commit()
        print(f"表 {NEW_TABLE} 创建成功（主键: uid, id）。")


def compute_result(value, low, high):
    """
    判断 value 是否在 [low, high] 区间内
    返回 0（正常）、1（异常）或 None（任一输入为 NULL）
    """
    if value is None or low is None or high is None:
        return None
    return 0 if low <= value <= high else 1


def main():
    conn = get_db_connection()
    try:
        # 1. 创建新表
        create_new_table(conn)

        # 2. 构建 SELECT 字段列表（仅查询需要的列）
        select_cols = (
            BASE_COLUMNS
            + [f"itemresult_{m}" for m in METRICS]
            + [f"normallowvalue_{m}" for m in METRICS]
            + [f"normalhighvalue_{m}" for m in METRICS]
        )
        select_sql = f"SELECT {', '.join(select_cols)} FROM item_merged_wide"

        # 3. 准备插入语句
        insert_cols = BASE_COLUMNS + [f"result_{m}" for m in METRICS]
        placeholders = ", ".join(["%s"] * len(insert_cols))
        insert_sql = f"INSERT INTO {NEW_TABLE} ({', '.join(insert_cols)}) VALUES ({placeholders})"

        # 4. 分批读取 + 批量插入
        with conn.cursor() as read_cursor:
            read_cursor.execute(select_sql)
            total_inserted = 0
            insert_buffer = []

            while True:
                rows = read_cursor.fetchmany(READ_BATCH_SIZE)
                if not rows:
                    break

                # 处理当前批次
                for row in rows:
                    # 基础字段值
                    base_values = list(row[: len(BASE_COLUMNS)])
                    offset = len(BASE_COLUMNS)

                    # 计算每个指标的 result
                    result_values = []
                    for i, metric in enumerate(METRICS):
                        item_val = row[offset + i]  # itemresult
                        low_val = row[offset + len(METRICS) + i]  # normallowvalue
                        high_val = row[offset + 2 * len(METRICS) + i]  # normalhighvalue
                        result_values.append(
                            compute_result(item_val, low_val, high_val)
                        )

                    # 组装完整行
                    insert_buffer.append(base_values + result_values)

                    # 当缓冲区达到批量插入阈值时执行插入
                    if len(insert_buffer) >= INSERT_BATCH_SIZE:
                        with conn.cursor() as insert_cursor:
                            insert_cursor.executemany(insert_sql, insert_buffer)
                        conn.commit()
                        total_inserted += len(insert_buffer)
                        print(f"已插入 {total_inserted} 行...")
                        insert_buffer = []

            # 处理剩余数据
            if insert_buffer:
                with conn.cursor() as insert_cursor:
                    insert_cursor.executemany(insert_sql, insert_buffer)
                conn.commit()
                total_inserted += len(insert_buffer)
                print(f"全部完成，共插入 {total_inserted} 行。")

    except Exception as e:
        conn.rollback()
        print(f"发生错误，事务已回滚: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
