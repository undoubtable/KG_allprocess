import os
import csv
from neo4j import GraphDatabase
from pipeline_config import STEP4_NODES_TSV, STEP4_EDGES_TSV

NODE_TSV_PATH = str(STEP4_NODES_TSV)
EDGE_TSV_PATH = str(STEP4_EDGES_TSV)
# ============== 需要你修改的配置 ==============

# 1）Neo4j 连接信息
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "20011127"

# 2）节点 & 关系列表 TSV 路径
#   注意：这里用的是你“新 Step4”产出的文件
# NODE_TSV_PATH = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
# EDGE_TSV_PATH = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"

# 3）每次批量写入多少行
BATCH_SIZE = 500

# ============== 下面一般不用改 ==============


def clear_database(driver):
    """
    可选操作：清空当前数据库所有节点和关系
    使用前先确认数据库里没有别的数据！
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🔥 已清空 Neo4j 当前数据库（MATCH (n) DETACH DELETE n）")


def create_constraint(driver):
    """
    给 :Concept(id) 建唯一约束，避免同一 id 重复创建节点
    """
    cypher = """
    CREATE CONSTRAINT IF NOT EXISTS
    FOR (n:Concept)
    REQUIRE n.id IS UNIQUE
    """
    with driver.session() as session:
        session.run(cypher)
    print("✅ 已创建/存在唯一约束：(:Concept {id})")


def load_nodes(driver, tsv_path: str):
    """
    从 TSV 创建节点：
    预计列：node_id, name, label, page_no, sentence_id
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    def _create_nodes_tx(tx, rows):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (n:Concept {id: row.node_id})
            SET n.name        = row.name,
                n.label       = row.label,
                n.page_no     = row.page_no,
                n.sentence_id = row.sentence_id
            RETURN count(*) AS _
            """,
            rows=rows,
        )

    total = 0
    batch = []

    with open(tsv_path, "r", encoding="utf-8") as f, driver.session() as session:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # page_no 转 int，失败则记为 -1
            try:
                page_no = int(row.get("page_no", -1))
            except ValueError:
                page_no = -1

            batch.append(
                {
                    "node_id": row["node_id"],
                    "name": row["name"],
                    "label": row.get("label", "Concept"),
                    "page_no": page_no,
                    "sentence_id": row.get("sentence_id", ""),
                }
            )
            if len(batch) >= BATCH_SIZE:
                session.execute_write(_create_nodes_tx, batch)
                total += len(batch)
                print(f"🧱 已写入节点数：{total}")
                batch = []

        if batch:
            session.execute_write(_create_nodes_tx, batch)
            total += len(batch)
            print(f"🧱 已写入节点数：{total}")

    print(f"✅ 节点导入完成，总数：{total}")


def load_edges(driver, tsv_path: str):
    """
    从 TSV 创建关系：
    预计列：edge_id, src_id, dst_id, relation_type, confidence, page_no, sentence_id
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    def _create_edges_tx(tx, rows):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:Concept {id: row.src_id})
            MATCH (b:Concept {id: row.dst_id})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.type        = row.relation_type,
                r.confidence  = row.confidence,
                r.page_no     = row.page_no,
                r.sentence_id = row.sentence_id
            RETURN count(*) AS _
            """,
            rows=rows,
        )

    total = 0
    batch = []

    with open(tsv_path, "r", encoding="utf-8") as f, driver.session() as session:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                page_no = int(row.get("page_no", -1))
            except ValueError:
                page_no = -1
            try:
                confidence = float(row.get("confidence", 0.0))
            except ValueError:
                confidence = 0.0

            batch.append(
                {
                    "src_id": row["src_id"],
                    "dst_id": row["dst_id"],
                    "relation_type": row.get("relation_type", "related_to"),
                    "confidence": confidence,
                    "page_no": page_no,
                    "sentence_id": row.get("sentence_id", ""),
                }
            )
            if len(batch) >= BATCH_SIZE:
                session.execute_write(_create_edges_tx, batch)
                total += len(batch)
                print(f"🔗 已写入关系数：{total}")
                batch = []

        if batch:
            session.execute_write(_create_edges_tx, batch)
            total += len(batch)
            print(f"🔗 已写入关系数：{total}")

    print(f"✅ 关系导入完成，总数：{total}")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # 如需每次重建图，把下面这行的注释去掉：
        # clear_database(driver)

        create_constraint(driver)

        print("\n=== 开始导入节点 ===")
        load_nodes(driver, NODE_TSV_PATH)

        print("\n=== 开始导入关系 ===")
        load_edges(driver, EDGE_TSV_PATH)

        print("\n🎉 所有数据已导入 Neo4j！可以在 Browser 中执行：")
        print("   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50;")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
