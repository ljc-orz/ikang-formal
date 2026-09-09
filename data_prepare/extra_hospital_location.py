import json
import re
import time
import requests
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

# vLLM 服务地址和模型名称（请根据实际部署修改）
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-27B"  # 若部署时指定了 --served-model-name，请保持一致


def get_db_connection():
    """创建数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def infer_location(hospname):
    """
    调用 vLLM 从医院名称推测省份和城市
    返回 (province, city)，若失败则返回 (None, None)
    """
    system_prompt = "你是一个地理信息专家，能从医院名称中准确推断其所在省份和城市。"
    user_prompt = (
        f"请从以下医院名称推断省份和城市，只返回一个JSON对象，"
        f"键为 province 和 city，不要有其他文字。医院名称：{hospname}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }

    try:
        resp = requests.post(VLLM_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # 尝试直接解析 JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 若包含 markdown 代码块，提取其中的 JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        province = result.get("province", "")
        city = result.get("city", "")
        return province, city

    except Exception as e:
        print(f"推理失败 (hospname={hospname}): {e}")
        return None, None


def main():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 只处理 province 或 city 为 NULL 的记录（可酌情修改条件）
            select_sql = "SELECT hospid, hospname FROM hospital WHERE province IS NULL OR city IS NULL"
            cursor.execute(select_sql)
            rows = cursor.fetchall()
            print(f"共找到 {len(rows)} 条待处理记录。")

            for hospid, hospname in rows:
                print(f"正在处理 ID={hospid}: {hospname}")
                province, city = infer_location(hospname)

                if province is None or city is None:
                    print(f"跳过 ID={hospid}（推理失败或无结果）")
                    continue

                # 更新数据库
                update_sql = (
                    "UPDATE hospital SET province = %s, city = %s WHERE hospid = %s"
                )
                cursor.execute(update_sql, (province, city, hospid))
                print(f"已更新 ID={hospid}: province='{province}', city='{city}'")

                # 可选：每处理一条提交一次，避免长事务
                # conn.commit()

            # 全部处理完成后统一提交
            conn.commit()
            print("所有更新已提交。")

    except Exception as e:
        conn.rollback()
        print(f"发生错误，事务已回滚: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
