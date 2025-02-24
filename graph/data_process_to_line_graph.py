import json
import os

import numpy as np
from tqdm import tqdm


def compute_direction_cosines(point1, point2):
    """计算两点之间的方向余弦"""
    vector = np.array(point2) - np.array(point1)
    length = np.linalg.norm(vector)
    if length == 0:
        return [0, 0, 0]  # 处理两点重合的情况
    # 将numpy数组转换为普通Python列表
    return (vector / length).tolist()


def process_skeleton_data(input_path, output_path, edges):
    """处理骨架数据为边的方向余弦"""
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 遍历输入目录中的所有json文件
    for filename in tqdm(os.listdir(input_path)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(input_path, filename), "r") as f:
            data = json.load(f)

        # 获取原始骨架数据
        skeletons = data["skeletons"]

        # 转换为边的方向余弦
        edge_cosines = []
        for frame in skeletons:
            frame_edges = []
            for edge in edges:
                start_idx = edge[0] - 1  # 因为边的定义是从1开始的
                end_idx = edge[1] - 1
                direction_cosine = compute_direction_cosines(
                    frame[start_idx], frame[end_idx]
                )
                frame_edges.append(direction_cosine)
            edge_cosines.append(frame_edges)

        # 保存转换后的数据
        output_data = {"file_name": data["file_name"], "edge_cosines": edge_cosines}

        output_file = os.path.join(output_path, filename)
        with open(output_file, "w") as f:
            json.dump(output_data, f)


def main():
    # UCLA数据集的边定义
    edges = [
        (1, 2),
        (2, 3),
        (4, 3),
        (5, 3),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 3),
        (10, 9),
        (11, 10),
        (12, 11),
        (13, 1),
        (14, 13),
        (15, 14),
        (16, 15),
        (17, 1),
        (18, 17),
        (19, 18),
        (20, 19),
    ]

    input_path = "../data/NW-UCLA/all_sqe"  # 原始数据路径
    output_path = "../data/NW-UCLA/all_sqe_edge"  # 处理后数据保存路径

    process_skeleton_data(input_path, output_path, edges)
    print("处理完成！")


if __name__ == "__main__":
    main()
