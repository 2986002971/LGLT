#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import yaml


def read_graph_from_yaml(file_path):
    """从YAML文件读取图结构"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["edges"]


def create_line_graph(edges):
    """创建线图（对偶图）"""
    # 创建一个NetworkX图对象
    G = nx.Graph(edges)
    # 生成线图
    L = nx.line_graph(G)
    # 转换为边列表格式
    dual_edges = list(L.edges())
    # 为边重新编号
    edge_to_id = {e: i for i, e in enumerate(G.edges())}
    # 转换dual_edges中的表示方式，从边对转换为编号对
    numbered_edges = []
    for e1, e2 in dual_edges:
        numbered_edges.append((edge_to_id[e1], edge_to_id[e2]))
    return numbered_edges, edge_to_id


def save_dual_graph_to_yaml(dual_edges, output_file):
    """将对偶图保存为YAML格式"""
    # 创建简单的边列表格式
    output_data = {"edges": [list(edge) for edge in dual_edges]}

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)


def main():
    # 命令行参数处理
    import argparse

    parser = argparse.ArgumentParser(description="图的点线对偶转换工具")
    parser.add_argument("input", help="输入YAML文件路径")
    parser.add_argument("output", help="输出YAML文件路径")
    args = parser.parse_args()

    try:
        # 读取原图
        edges = read_graph_from_yaml(args.input)
        print(f"已读取原图，共有 {len(edges)} 条边")

        # 生成对偶图
        dual_edges, _ = create_line_graph(edges)  # 不再需要edge_mapping
        print(f"已生成对偶图，共有 {len(dual_edges)} 条边")

        # 保存结果
        save_dual_graph_to_yaml(dual_edges, args.output)
        print(f"已将结果保存至 {args.output}")

    except Exception as e:
        print(f"错误：{str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
