import math
import random
import pandas as pd
import numpy as np
from graph_tool import topology
from graph_tool import generation
from graph_tool import centrality
from graph_tool import Graph
import graph_tool.all as gt
from scipy.spatial import cKDTree
import community as community_louvain  # 导入 community-louvain
import networkx as nx
import heapq

# import cugraph
# import cudf
# import nx_cugraph as nxcg

def compute_angle(node1, node2, g_pos):
    """
    This function is made for determine the first hop node which has the smallest angle to the initial node.

    :param node1: initial node
    :param node2: first hop node
    :param g_pos: the position object of a graph
    :return: angle in float format
    """
    x1, y1 = g_pos[node1]
    x2, y2 = g_pos[node2]

    v = np.array([x2 - x1, y2 - y1])
    angle = np.arctan2(v[1], v[0])

    return angle


def compute_angle_v4(node1, node2, node3, g_pos):
    """
    This function calculates the angle between line A (node1-node2) and line B (node2-node3).
    The angle is measured counterclockwise from line A to line B, and is always positive.

    :param node1: First node (reference node)
    :param node2: Second node
    :param node3: Third node
    :param g_pos: Position object of the graph (node positions)
    :return: Angle between line A and line B
    """
    pos_node1 = np.array(g_pos[node1])
    pos_node2 = np.array(g_pos[node2])
    pos_node3 = np.array(g_pos[node3])

    vector_a = pos_node1 - pos_node2
    vector_b = pos_node3 - pos_node2

    angle = np.degrees(np.arctan2(np.cross(vector_a, vector_b), np.dot(vector_a, vector_b)))
    angle = (angle + 360) % 360  # Ensure angle is positive and within [0, 360)

    return angle


def calculate_distance(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算纬度和经度的差值
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # 应用哈弗赛恩公式
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球平均半径：6371km
    R = 6371

    # 计算距离
    distance = R * c

    return distance


def un_convex_search(g, g_pos):
    """
    Used for finding all margin nodes of a graph, even if it is an un-convex graph.
    :param g: the graph object.
    :param g_pos: the position object
    :return: a list with all margin nodes index
    """
    # 找到y坐标最小的点作为起始点P0
    # 将节点位置信息转换为NumPy数组
    # TODO: 将99999和-1替换成正无穷大和负无穷大，并且需要注意将浮点类型转换为整型。
    x_min = 99999
    y_min = 99999
    x_max = -1
    y_max = -1

    x_min_index = -1
    x_max_index = -1
    y_min_index = -1
    y_max_index = -1

    margin_nodes_dict = {}

    # 遍历每个节点
    for v in g.vertices():
        # 获取当前节点的索引
        v_pos = g_pos[v]
        if v_pos[0] < x_min:
            x_min = v_pos[0]
            x_min_index = v
        if v_pos[0] > x_max:
            x_max = v_pos[0]
            x_max_index = v
        if v_pos[1] < y_min:
            y_min = v_pos[1]
            y_min_index = v
        if v_pos[1] > y_max:
            y_max = v_pos[1]
            y_max_index = v

    margin_nodes_dict['x_min'] = x_min_index
    margin_nodes_dict['x_max'] = x_max_index
    margin_nodes_dict['y_min'] = y_min_index
    margin_nodes_dict['y_max'] = y_max_index

    initial_vertex = y_min_index
    fake_initial_vertex = y_min_index
    first_hop_node = initial_vertex
    margin_candidates = []
    neighbors_list = []
    main_router = []
    main_router.append(int(initial_vertex))
    margin_candidates.append(int(initial_vertex))
    visited = set()
    visited.add(int(initial_vertex))
    step_count = 0
    fake_initial_flag = False

    for node in g.get_all_neighbors(initial_vertex):
        angle = compute_angle(initial_vertex, node, g_pos)
        degree = g.get_total_degrees([node])
        if degree == 1:
            visited.add(node)
            margin_candidates.append(node)
            continue
        neighbors_list.append((node, angle))

    neighbors_list.sort(key=lambda x: x[1], reverse=False)

    for item in neighbors_list:
        node, angle = item
        if angle >= 0:
            first_hop_node = node
            visited.add(first_hop_node)
            margin_candidates.append(int(first_hop_node))
            main_router.append(int(first_hop_node))
            step_count += 1
            if g.get_total_degrees([fake_initial_vertex]) == 1:
                fake_initial_vertex = first_hop_node
                step_count = 0
                fake_initial_flag = True
            break

    # start from the second hop
    stack = [(first_hop_node, initial_vertex)]
    while stack:
        current_vertex, previous_vertex = stack.pop()
        neighbors_list = []

        if fake_initial_flag:
            if g.get_total_degrees([current_vertex]) > 2:
                fake_initial_vertex = current_vertex
                fake_initial_flag = False
                step_count = 0

        # When finished the rounding search, the router would arrived at the node which is also be the first hop of
        # the original node
        if (not fake_initial_flag) and step_count > 1 and fake_initial_vertex in g.get_all_neighbors(current_vertex):
            break

        for neighbor in g.get_all_neighbors(current_vertex):
            # Compare the angle and choose the smallest node as the next hop, while if there are some node have
            # degree 1, then directly save them into the margin_candidates and continue the loop.
            if neighbor not in visited:
                if g.get_total_degrees([neighbor]) == 1:
                    margin_candidates.append(neighbor)
                    visited.add(neighbor)
                else:
                    angle = compute_angle_v4(previous_vertex, current_vertex, neighbor, g_pos)
                    neighbors_list.append((neighbor, angle))
        neighbors_list.sort(key=lambda x: x[1], reverse=False)

        if len(neighbors_list) == 0:
            while main_router:
                last_hop = main_router.pop()
                current_vertex = last_hop
                try:
                    previous_vertex = main_router[-1]
                except IndexError:
                    # It means that the router has back arrived to the initial node
                    temp_neighbors_list = []
                    for node in g.get_all_neighbors(current_vertex):
                        angle = compute_angle(current_vertex, node, g_pos)
                        degree = g.get_total_degrees([node])
                        if degree != 1 and node not in visited:
                            temp_neighbors_list.append((node, angle))
                    temp_neighbors_list.sort(key=lambda x: x[1], reverse=False)
                    for item in temp_neighbors_list:
                        node, angle = item
                        if angle >= 0 and node not in visited:
                            previous_vertex = current_vertex
                            current_vertex = node
                            main_router.append(previous_vertex)
                            main_router.append(current_vertex)
                            visited.add(int(current_vertex))
                            break

                for neighbor in g.get_all_neighbors(last_hop):
                    if neighbor not in visited:
                        angle = compute_angle_v4(previous_vertex, current_vertex, neighbor, g_pos)
                        neighbors_list.append((neighbor, angle))
                if len(neighbors_list) != 0:
                    neighbors_list.sort(key=lambda x: x[1], reverse=False)
                    break
            if len(neighbors_list) == 0:
                break

        if len(neighbors_list) != 0:
            item = neighbors_list[0]
            node = item[0]
            visited.add(node)
            margin_candidates.append(node)
            main_router.append(int(node))
            stack.append((node, current_vertex))
            step_count += 1

    # Finished the search
    return margin_candidates


def compute_second_largest_component_size(g, vertices, second_flag=True):
    # 获取所有连通子图的大小
    component_sizes, comp = topology.vertex_percolation(g, vertices, second=second_flag)
    return component_sizes, comp


def compute_second_largest_component_size_manual(g, second_flag=True):
    # 计算连通分量
    comp, hist = topology.label_components(g)

    # 将连通分量的大小排序
    comp_sizes = hist.tolist()  # hist 是每个连通分量的大小
    comp_sizes.sort(reverse=True)

    # 根据 second_flag 获取最大或第二大连通分量的大小
    if second_flag:
        # 如果存在第二大连通分量，则返回它的大小
        return comp_sizes[1] if len(comp_sizes) > 1 else 0
    else:
        # 否则返回最大连通分量的大小
        return comp_sizes[0] if comp_sizes else 0


def hub_attack(g, hub_attack_flag=True, second_flag=True):
    # 与删除顺序相反的列表。即，下列表为节点度从小到大的顺序。而在渗透函数中，该列表为后进先出法。
    vertices = sorted([v for v in g.vertices()], key=lambda v: g.get_total_degrees([v]))
    if not hub_attack_flag:
        # np.random.seed(42)
        np.random.shuffle(vertices)
    # 计算当前次最大连通子图的大小
    second_largest_size, components = compute_second_largest_component_size(g, vertices, second_flag=second_flag)
    return second_largest_size


def la_attack(g, mode="pure", second_flag=True, sampling_ratio=1.0):
    if mode == "pure":
        start_vertex = np.random.choice(g.get_vertices())
    elif mode == "hub+la":
        get_degrees = g.get_total_degrees  # 局部变量优化
        degrees = get_degrees(g.get_vertices())
        start_vertex = np.argmax(degrees)
    else:
        raise ValueError("Unsupported mode. Choose either 'pure' or 'hub+la'.")

    vertices_to_remove = [start_vertex]
    visited = {start_vertex}  # 用于记录已访问过的节点
    get_neighbors = g.get_all_neighbors  # 局部变量优化

    while len(vertices_to_remove) < g.num_vertices():
        current_neighbors = set()  # 存储当前层的所有邻居

        # 对每个待处理节点获取其邻居
        for vertex in vertices_to_remove:
            neighbors = set(get_neighbors(vertex))
            current_neighbors.update(neighbors - visited)  # 添加尚未访问的邻居

        current_neighbors = list(current_neighbors)
        np.random.shuffle(current_neighbors)  # 随机排序

        # 如果模式是 'pure' 或 'hub+la'，且 sampling_ratio 不为1， 则进行采样
        if mode in ["pure", "hub+la"] and 0 < sampling_ratio < 1:
            num_to_sample = int(len(current_neighbors) * sampling_ratio)
            current_neighbors = np.random.choice(current_neighbors, num_to_sample, replace=False).tolist()

        # 将这些邻居添加到待处理节点列表，并更新已访问节点集合
        vertices_to_remove.extend(current_neighbors)
        visited.update(current_neighbors)

    # 将vertices_to_remove逆序以适应后续计算
    vertices_to_remove = vertices_to_remove[::-1]

    second_largest_size, _ = compute_second_largest_component_size(g, vertices_to_remove, second_flag=second_flag)
    return second_largest_size


def betweenness_attack(g, second_flag=True, recalculate_betweenness=False, stop_fraction=1.0, graph_flag=False):
    # 为节点和边添加介数中心性属性
    v_betweenness_map = g.new_vertex_property("double")

    # 计算所有节点的介数中心性
    vertex_betweenness, _ = centrality.betweenness(g, vprop=v_betweenness_map, norm=False)
    vertices_sorted_by_betweenness = sorted([v for v in g.vertices()],
                                            key=lambda v: vertex_betweenness[v],
                                            reverse=True)

    # 存储每次移除节点后的连通分量大小
    sizes = []

    num_nodes = g.num_vertices()
    stop_count = int(num_nodes * stop_fraction)

    # 创建一个新的图对象来存储被攻击后的图
    # attacked_graph = g.copy()
    vertex_filter = g.new_vertex_property("bool", vals=True)

    # 如果不需要重新计算介数中心性，只需一次性移除所有节点
    if not recalculate_betweenness:
        sizes, _ = compute_second_largest_component_size(g, vertices_sorted_by_betweenness[:stop_count], second_flag=second_flag)
        if graph_flag:
            for v in vertices_sorted_by_betweenness[:stop_count]:
                vertex_filter[v] = False
            g.set_vertex_filter(vertex_filter)
            return sizes, g
        else:
            return sizes

    # 如果需要重新计算介数中心性，每次移除一个节点后都重新计算
    for i in range(0, stop_count):
        vertex_filter[vertices_sorted_by_betweenness[0]] = False
        g.set_vertex_filter(vertex_filter)
        current_node = vertices_sorted_by_betweenness[0]

        # 计算当前的连通分量大小(First or Second LCC)
        current_size = compute_second_largest_component_size_manual(g, second_flag=second_flag)
        sizes.append(current_size)

        # 重新计算介数中心性
        vertex_betweenness, _ = centrality.betweenness(g, vprop=v_betweenness_map, norm=False)
        vertices_sorted_by_betweenness = sorted([v for v in g.vertices()],
                                                key=lambda v: vertex_betweenness[v],
                                                reverse=True)
    sizes = np.array(sizes)
    sizes_reversed = np.flip(sizes)


    if graph_flag:
        g.set_vertex_filter(vertex_filter)
        return sizes_reversed, g
    else:
        return sizes_reversed


def module_based_attack(graph, second_flag=False):
    # 调用louvain划分方法生成社区网络
    partition = louvain_community_detection(graph)
    blocks= graph.new_vertex_property('int')
    graph.vp['community'] = blocks
    for node, community in partition.items():
        graph.vp.community[graph.vertex(node)] = community

    # 将划分了社区的网络对象传入mb攻击函数,得到攻击列表
    intercommunity_nodes = set()
    for edge in graph.edges():
        source_vertex = edge.source()
        target_vertex = edge.target()
        if graph.vp.community[source_vertex] != graph.vp.community[target_vertex]:
            intercommunity_nodes.add(source_vertex)
            intercommunity_nodes.add(target_vertex)

    node_betweenness = centrality.betweenness(graph)[0]
    sorted_nodes = sorted(intercommunity_nodes, key=lambda v: node_betweenness[v], reverse=True)

    removed_nodes = []
    while sorted_nodes:
        node = sorted_nodes.pop(0)
        if node not in removed_nodes:
            removed_nodes.append(node)
            for neighbor in node.all_neighbors():
                if graph.vp.community[neighbor] != graph.vp.community[node]:
                    if neighbor in sorted_nodes:
                        sorted_nodes.remove(neighbor)

            lcc_vertex_mask = topology.label_largest_component(graph)
            sorted_nodes = [v for v in sorted_nodes if lcc_vertex_mask[v]]

    if len(removed_nodes) < graph.num_vertices():
        nodes_not_removed = set(graph.iter_vertices()) - set(removed_nodes)
        padding_required = graph.num_vertices() - len(removed_nodes)

        for node in nodes_not_removed:
            if padding_required <= 0:
                break
            removed_nodes.append(node)
            padding_required -= 1

    removed_nodes.reverse()

    # 将攻击节点顺序传入 compute second largest函数获得LCC size并返回
    component_sizes, _ = compute_second_largest_component_size(graph, removed_nodes, second_flag)

    return component_sizes


def get_x_range(g_pos):
    # 计算X轴的区间范围
    x_values = [pos[0] for pos in g_pos]
    x_min = min(x_values)
    x_max = max(x_values)
    return x_min, x_max


def get_y_range(g_pos):
    # 计算Y轴的区间范围
    y_values = [pos[1] for pos in g_pos]
    y_min = min(y_values)
    y_max = max(y_values)
    return y_min, y_max


def find_center(g_pos, result_list, num_sectors):
    # 计算X轴和Y轴的区间范围
    x_min, x_max = get_x_range(g_pos)
    y_min, y_max = get_y_range(g_pos)

    # 计算中心点的坐标
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # 将极角划分为等间隔的角度
    sector_angle = 2 * math.pi / num_sectors

    # 存储扇形区域和其对应的节点
    sectors = {}

    for node in result_list:
        x, y = g_pos[node]
        angle = math.atan2(y - center_y, x - center_x)
        normalized_angle = math.fmod(angle, 2 * math.pi)
        # 将极角转换为非负值
        if normalized_angle < 0:
            normalized_angle += 2 * math.pi
        # 计算节点所属的扇形区域
        sector_index = int(normalized_angle / sector_angle)
        # 将节点添加到对应的扇形区域
        if sector_index in sectors:
            sectors[sector_index].append(node)
        else:
            sectors[sector_index] = [node]

    return sectors


def find_farthest_pairs(g, g_pos, num_sectors, margin_nodes_list, extend_search=0):
    """
    A function for getting all farthest_pairs in different sectors.
    :param g: graph object
    :param g_pos: position object
    :param num_sectors: even number only
    :param margin_nodes_list: all margin nodes of a network, list type only
    :param extend_search: the number of sectors to extend search if no pair is found in opposite sectors
    :return: a list of all farthest pairs located in all different sectors.
    """
    sectors = find_center(g_pos, margin_nodes_list, num_sectors)
    pairs_result = []
    edges_set = {(edge.source(), edge.target()) for edge in g.edges()}

    for sector_index in range(num_sectors):
        for extended_search_index in range(extend_search + 1):
            opposite_sector_index = (sector_index + num_sectors // 2 + extended_search_index) % num_sectors
            this_sector = sectors.get(sector_index)
            opposite_sector = sectors.get(opposite_sector_index)
            if this_sector is None or opposite_sector is None:
                continue

            this_sector_degrees = list(zip(this_sector, g.get_total_degrees(this_sector)))
            opposite_sector_degrees = list(zip(opposite_sector, g.get_total_degrees(opposite_sector)))

            this_sector_degrees.sort(key=lambda x: x[1])
            opposite_sector_degrees.sort(key=lambda x: x[1])

            for min_source_node_index, _ in this_sector_degrees:
                for min_target_node_index, _ in opposite_sector_degrees:
                    if min_source_node_index != min_target_node_index and \
                            (min_source_node_index, min_target_node_index) not in edges_set and \
                            (min_target_node_index, min_source_node_index) not in edges_set:
                        pairs_result.append((min_source_node_index, min_target_node_index))
                        break  # Found a valid pair, move to next source node
                else:
                    continue  # No valid pair found, try next source node
                break  # A valid pair found, move to next sector
            if pairs_result:
                break

    return pairs_result


def find_min_degree_product_pairs(g, g_pos, num_sectors, margin_nodes_list, extend_search=0, random_flag=False):
    """
    A function for getting all pairs with minimum degree product in different sectors.
    :param random_flag: if random_flag, the minimum_degree production process would be replaced by random selection.
    :param g: graph object
    :param g_pos: position object
    :param num_sectors: even number only
    :param margin_nodes_list: all margin nodes of a network, list type only
    :param extend_search: the number of sectors to extend search if no pair is found in opposite sectors
    :return: a list of all pairs with minimum degree product located in all different sectors.
    """
    if num_sectors == 0:
        if random_flag:
            temp_nodes_list = margin_nodes_list.copy()
            while len(temp_nodes_list) > 1:  # Ensure there are at least 2 nodes to form a pair
                node_i, node_j = random.sample(temp_nodes_list, 2)
                if not g.edge(node_i, node_j):
                    return [(node_i, node_j)]
                else:
                    # Remove the nodes from the temp list to ensure all nodes are checked
                    temp_nodes_list.remove(node_i)
                    temp_nodes_list.remove(node_j)

            # If the loop finishes and no pair is found, return an empty list
            return []
        else:
            # If num_sectors is 0, select the pair of nodes with the smallest degree product from the entire network
            min_degree_product = float('inf')
            min_product_pair = None

            # Sort the nodes by their degrees
            sorted_nodes = sorted(margin_nodes_list, key=lambda node: g.get_total_degrees([node]))
            for i, node_i in enumerate(sorted_nodes):
                for node_j in sorted_nodes[i + 1:]:
                    degree_product = g.get_total_degrees([node_i]) * g.get_total_degrees([node_j])
                    if degree_product < min_degree_product and not g.edge(node_i, node_j):
                        min_degree_product = degree_product
                        min_product_pair = (node_i, node_j)
                # If the degree product of the current node and the node with the smallest degree is already larger than the
                # current minimum degree product, we can stop the search because the degree product will only increase
                # for the following nodes
                if g.get_total_degrees([node_i]) * g.get_total_degrees([sorted_nodes[0]]) > min_degree_product:
                    break

            return [min_product_pair] if min_product_pair else []

    else:
        sectors = find_center(g_pos, margin_nodes_list, num_sectors)
        pairs_result = []
        edges_set = {(edge.source(), edge.target()) for edge in g.edges()}

        for sector_index in range(num_sectors):
            for extended_search_index in range(extend_search + 1):
                opposite_sector_index = (sector_index + num_sectors // 2 + extended_search_index) % num_sectors
                this_sector = sectors.get(sector_index)
                opposite_sector = sectors.get(opposite_sector_index)
                if this_sector is None or opposite_sector is None:
                    continue

                min_degree_product = float('inf')
                min_product_pair = None

                for this_node in this_sector:
                    for opposite_node in opposite_sector:
                        if this_node != opposite_node and \
                                (this_node, opposite_node) not in edges_set and \
                                (opposite_node, this_node) not in edges_set:
                            degree_product = g.get_total_degrees([this_node]) * g.get_total_degrees([opposite_node])
                            if degree_product < min_degree_product:
                                min_degree_product = degree_product
                                min_product_pair = (this_node, opposite_node)

                if min_product_pair:
                    pairs_result.append(min_product_pair)
                    break  # Found a valid pair, move to next sector
                else:
                    continue  # No valid pair found, try next source node

        return pairs_result


def get_sector_nodes(g_pos, num_sectors, margin_nodes_list):
    """
    A function for getting all nodes in different sectors.
    :param g_pos: position object
    :param num_sectors: even number only
    :param margin_nodes_list: all margin nodes of a network, list type only
    :return: a dict of sector to nodes mapping.
    """
    sectors = find_center(g_pos, margin_nodes_list, num_sectors)
    sector_nodes = {}  # Additional return value

    for sector_index in range(num_sectors):
        sector_nodes[sector_index] = sectors.get(sector_index, [])

    return sector_nodes


def multiply_and_sum_sector_nodes(sector_nodes):
    num_sectors = len(sector_nodes)
    sum_product = 0
    for sector_index in range(num_sectors // 2):
        opposite_sector_index = (sector_index + num_sectors // 2) % num_sectors
        sum_product += len(sector_nodes[sector_index]) * len(sector_nodes[opposite_sector_index])
    return sum_product


def find_min_degree_pair(graph, pairs, pos):
    # 设置一个最小值开始，用于之后的比较
    min_degree_sum = float('inf')
    min_pair = None

    # 遍历每一对节点
    for pair in pairs:
        # 计算每一对节点的度数之和
        degree_0 = graph.get_total_degrees([pair[0]])
        degree_1 = graph.get_total_degrees([pair[1]])
        degree_sum = degree_0 * degree_1

        # 如果这一对节点的度数之和小于当前的最小值
        if degree_sum < min_degree_sum:
            # 更新最小值和最小值对应的节点对
            min_degree_sum = degree_sum
            min_pair = pair

    # 返回度数之和最小的节点对
    return min_pair


def find_random_pair(graph, pairs, pos):
    # 随机选择一对节点
    random_pair = random.choice(pairs)

    return random_pair


def find_max_distance_pair(pairs, pos):
    max_distance = 0
    max_distance_pair = None

    for pair in pairs:
        distance = calculate_distance(pos[pair[0]], pos[pair[1]])
        if distance > max_distance:
            max_distance = distance
            max_distance_pair = pair

    return max_distance_pair


def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def extend_with_one_hop_neighbors(graph, margin_nodes_list):
    extended_nodes = set(margin_nodes_list)  # Start with the given nodes
    for node in margin_nodes_list:
        neighbors = graph.get_all_neighbors(node)  # Assuming `graph.neighbors()` gives the neighbors of a node
        extended_nodes.update(neighbors)  # Add neighbors to the set
    return list(extended_nodes)  # Convert set back to list


def sorted_nodes_by_degree(g):
    # 使用列表推导来创建一个包含 (节点, 度数) 的列表
    node_degree_pairs = [(v, v.out_degree() + v.in_degree()) for v in g.vertices()]

    # 根据度数对列表进行排序
    sorted_nodes = sorted(node_degree_pairs, key=lambda x: x[1])

    # 从 (节点, 度数) 对中提取并返回节点列表
    return [node for node, degree in sorted_nodes]


def calculate_mr(graph):
    """
    Calculate the mr value for a graph based on the formula:
    mr = <k> / (<k^2> - <k>)

    Parameters:
    - graph: A graph-tool Graph object.

    Returns:
    - mr value.
    """

    # Get the degrees of all vertices in the graph
    degrees = graph.degree_property_map("total").a

    # Calculate <k>
    avg_degree = sum(degrees) / len(degrees)

    # Calculate degree distribution P(k)
    degree_distribution = compute_degree_distribution(degrees)

    # Calculate <k^2> using the provided formula
    avg_degree_squared = sum(degree ** 2 * probability for degree, probability in degree_distribution.items())

    # Calculate and return mr
    mr = avg_degree / (avg_degree_squared - avg_degree)
    return mr


def compute_degree_distribution(degrees):
    # Step 1: 获取所有节点的度
    # degrees = graph.degree_property_map("total").a

    # Step 2: 计算每个度的频次
    degree_freq = {}
    for degree in degrees:
        if degree in degree_freq:
            degree_freq[degree] += 1
        else:
            degree_freq[degree] = 1

    # Step 3: 归一化得到 P(k)
    total_nodes = len(degrees)
    degree_distribution = {degree: freq / total_nodes for degree, freq in degree_freq.items()}

    return degree_distribution


def calculate_composite_costs(g, pos, delta=0):
    '''
    Calculate composite, apply the weighted distance to edges，
    delta between 0 and 1, From pure Euclidean distance to pure hop distance.
    '''
    cost = g.new_edge_property("double")
    for e in g.edges():
        source = e.source()
        target = e.target()
        source_pos = np.array(pos[source])
        target_pos = np.array(pos[target])
        distance = np.linalg.norm(source_pos - target_pos)
        cost[e] = (1 - delta) * distance + delta  # Simplifies to just distance if delta=0
    g.edge_properties["cost"] = cost
    return g


def calculate_average_degree(g):
    degree_map = g.degree_property_map("total")
    degree_list = degree_map.get_array()
    # 计算平均度
    average_degree = degree_list.mean()
    return average_degree


def dt_generator_by_csv(csv_file_path, inverse_y=False):
    """
    :param csv_file_path: input file path, till the .csv
    :return: DT graph object
    """
    data = pd.read_csv(csv_file_path)

    # 如果指示器为真，则翻转y轴
    if inverse_y:
        max_y = data['center_y'].max()
        min_y = data['center_y'].min()
        data['center_y'] = max_y + min_y - data['center_y']

    points = data[["center_x", "center_y"]].values
    populations = data["jinkou"].values  # Extract population data from the CSV file

    g, pos = generation.triangulation(points, type='delaunay')

    # Create vertex properties for 'number' and 'population'
    g.vertex_properties['number'] = g.new_vertex_property('int')
    g.vertex_properties['population'] = g.new_vertex_property('int', vals=populations)  # Assign population data

    for one_vertex in g.vertices():
        index = int(g.vertex_index[one_vertex])
        g.vertex_properties['number'][g.vertex(index)] = index

    return g, pos


def create_gg(graph_obj, graph_pos):
    """
    Attention: Till now, only available for Delaunay Triangle, might available for planar graphs, however,
    not suitable for real networks.
    :param graph_obj: the input must be a full connected graph, Delaunay Triangle is
    recommended.
    :param graph_pos: graph-tool pos object
    :return: gabriel graph object and gabriel pos object. At
    present, the output pos are the same as the input pos object.
    """
    gabriel_g = graph_obj.copy()
    gabriel_pos = graph_pos
    midpoint = [0, 0]
    gabriel_g.clear_edges()
    for e in graph_obj.edges():
        u, v = e.source(), e.target()
        u_pos, v_pos = graph_pos[u], graph_pos[v]
        midpoint[0] = (u_pos[0] + v_pos[0]) / 2
        midpoint[1] = (u_pos[1] + v_pos[1]) / 2
        radius = np.sqrt(np.square(midpoint[0] - u_pos[0]) + np.square(midpoint[1] - u_pos[1]))
        in_circle = False
        u_neighbors = graph_obj.get_all_neighbors(u)
        v_neighbors = graph_obj.get_all_neighbors(v)
        u_v_one_hop_neighbors = np.unique(np.concatenate((u_neighbors, v_neighbors)))
        for w in u_v_one_hop_neighbors:
            if w == u or w == v:
                continue
            w_pos = graph_pos[w]
            dist_w_mid = np.sqrt(np.square(midpoint[0] - w_pos[0]) + np.square(midpoint[1] - w_pos[1]))
            if dist_w_mid < radius:
                in_circle = True
                break
        if in_circle is False:
            gabriel_g.add_edge(u, v)
    return gabriel_g, gabriel_pos


def create_gg_v2(graph_obj, graph_pos):
    """
    Attention: Slow, but suitable for every network.
    :param graph_obj: the input must be a full connected graph
    :param graph_pos: graph-tool pos object
    :return: gabriel graph object and gabriel pos object. At present, the output pos are the same as the input pos object.
    """
    gabriel_g = graph_obj.copy()
    gabriel_pos = graph_pos
    midpoint = [0, 0]
    gabriel_g.clear_edges()
    for e in graph_obj.edges():
        u, v = e.source(), e.target()
        u_pos, v_pos = graph_pos[u], graph_pos[v]
        midpoint[0] = (u_pos[0] + v_pos[0]) / 2
        midpoint[1] = (u_pos[1] + v_pos[1]) / 2
        radius = np.sqrt(np.square(midpoint[0] - u_pos[0]) + np.square(midpoint[1] - u_pos[1]))
        in_circle = False
        u_neighbors = graph_obj.get_all_neighbors(u)
        v_neighbors = graph_obj.get_all_neighbors(v)
        for w in gabriel_g.vertices():
            if w == u or w == v:
                continue
            w_pos = graph_pos[w]
            dist_w_mid = np.sqrt(np.square(midpoint[0] - w_pos[0]) + np.square(midpoint[1] - w_pos[1]))
            if dist_w_mid < radius:
                in_circle = True
                break
        if in_circle is False:
            gabriel_g.add_edge(u, v)
    return gabriel_g, gabriel_pos


def create_gg_v3(graph_obj, graph_pos):
    """
        Creates a Gabriel graph using cKDTree for efficient range queries. Available for all networks.
        :param graph_obj: the input must be a full connected graph.
        :param graph_pos: graph-tool pos object.
        :return: gabriel graph object and gabriel pos object.
                 At present, the output pos are the same as the input pos object.
        """

    # 创建一个 gabriel_g 的拷贝
    gabriel_g = graph_obj.copy()
    gabriel_pos = graph_pos

    # 使用图中所有顶点位置构建cKDTree
    all_positions = [graph_pos[v][:2] for v in graph_obj.iter_vertices()]
    tree = cKDTree(all_positions)

    # 清除 gabriel_g 中的所有边
    gabriel_g.clear_edges()

    for e in graph_obj.edges():
        u, v = e.source(), e.target()
        u_pos, v_pos = graph_pos[u], graph_pos[v]

        # 计算中点和半径
        midpoint = [(u_pos[0] + v_pos[0]) / 2, (u_pos[1] + v_pos[1]) / 2]
        radius = np.sqrt(np.square(midpoint[0] - u_pos[0]) + np.square(midpoint[1] - u_pos[1]))

        # 使用cKDTree查询所有在半径内的点
        indices = tree.query_ball_point(midpoint, radius)

        # 检查查询结果中是否只有u和v，或者其他在圆内的点
        if set(indices).difference({graph_obj.vertex_index[u], graph_obj.vertex_index[v]}):
            continue

        gabriel_g.add_edge(u, v)

    return gabriel_g, gabriel_pos


def create_rng(graph_obj, graph_pos):
    """
    Create an RNG graph based on the new logic provided.

    :param graph_obj: The input graph, typically a Delaunay triangulation or Gabriel graph.
    :param graph_pos: Positional data for the graph vertices.
    :return: The RNG graph and its positional data.
    """
    rng_graph = graph_obj.copy()
    rng_graph.clear_edges()

    def dist(u_pos, v_pos):
        return np.sqrt((u_pos[0] - v_pos[0]) ** 2 + (u_pos[1] - v_pos[1]) ** 2)

    # Iterate over all the edges in the input graph
    for edge in graph_obj.edges():
        u, v = edge.source(), edge.target()
        u_pos, v_pos = graph_pos[u], graph_pos[v]

        # Determine the radius based on the edge length
        radius = dist(u_pos, v_pos)

        add_edge = True
        for w in graph_obj.vertices():
            if w != u and w != v:
                w_pos = graph_pos[w]
                if dist(u_pos, w_pos) < radius and dist(v_pos, w_pos) < radius:
                    add_edge = False
                    break

        if add_edge:
            rng_graph.add_edge(u, v)

    return rng_graph, graph_pos


def create_mst(graph_obj, graph_pos):
    """
    :param graph_obj: graph-tool's graph object
    :param graph_pos: graph-tool's pos object
    :return: mst_graph and mst_pos object
    Attention: If the graph object has the edge properties 'cost', that is the weighted distance.
    """
    try:
        # 尝试获取"cost"属性
        cost = graph_obj.edge_properties["cost"]
    except KeyError:
        # 如果"cost"属性不存在，设置cost为None
        cost = None

    # 根据cost是否存在来调用min_spanning_tree函数
    if cost is not None:
        mst_prop = topology.min_spanning_tree(graph_obj, weights=cost)
    else:
        mst_prop = topology.min_spanning_tree(graph_obj)

    mst_graph = Graph(directed=False)
    mst_graph.add_edge_list([(e.source(), e.target()) for e in graph_obj.edges() if mst_prop[e]])

    # 针对每个顶点和边属性，创建新的属性并复制数据
    for prop_name, prop_val in graph_obj.vertex_properties.items():
        new_prop_val = mst_graph.new_vertex_property(prop_val.value_type())
        for v in mst_graph.vertices():
            orig_v = graph_obj.vertex(v)
            new_prop_val[v] = prop_val[orig_v]
        mst_graph.vertex_properties[prop_name] = new_prop_val

    for prop_name, prop_val in graph_obj.edge_properties.items():
        new_prop_val = mst_graph.new_edge_property(prop_val.value_type())
        for e in mst_graph.edges():
            orig_e = graph_obj.edge(e.source(), e.target(), all_edges=False)
            new_prop_val[e] = prop_val[orig_e]
        mst_graph.edge_properties[prop_name] = new_prop_val

    mst_pos = graph_pos.copy()

    return mst_graph, mst_pos


def gt_to_nx(graph):
    nx_graph = nx.Graph()
    for v in graph.vertices():
        nx_graph.add_node(int(v))
    for e in graph.edges():
        nx_graph.add_edge(int(e.source()), int(e.target()))
    return nx_graph


def louvain_community_detection(graph):
    nx_graph = gt_to_nx(graph)
    partition = community_louvain.best_partition(nx_graph)
    return partition


# TODO: 验证下面两个louvain函数的正确性
def louvain_community_detection_with_modularity(graph):
    nx_graph = gt_to_nx(graph)
    partition = community_louvain.best_partition(nx_graph)
    modularity_score = community_louvain.modularity(partition, nx_graph)
    return partition, modularity_score


# def louvain_community_detection_cugraph(graph):
#     nx_graph = gt_to_nx(graph)
#     gdf = nxcg.from_networkx(nx_graph)
#     louvain_parts, modularity_score = cugraph.louvain(gdf)
#     return louvain_parts, modularity_score


def calculate_density(graph):
    N = graph.num_vertices()
    E = graph.num_edges()

    if graph.is_directed():
        density = E / (N * (N - 1))
    else:
        density = (2 * E) / (N * (N - 1))

    return density


def calculate_degree_4_with_degree_4_neighbors(g, no_neighbor_flag=False):
    if no_neighbor_flag:
        '''
        True for doesn't consider if neighbors are degree4 nodes
        '''
        degree_counts = np.array([v.out_degree() for v in g.vertices()])
        degree_4_ratio = np.sum(degree_counts == 4) / len(degree_counts)
        return degree_4_ratio
    else:
        degree_counts = np.array([v.out_degree() for v in g.vertices()])
        degree_4_nodes = [v for v in g.vertices() if v.out_degree() == 4]

        count = 0
        for v in degree_4_nodes:
            neighbors = v.out_neighbors()
            if all(g.vertex(u).out_degree() == 4 for u in neighbors):
                count += 1

        degree_4_with_4_neighbors_ratio = count / len(degree_counts)
        return degree_4_with_4_neighbors_ratio


def rewire_with_no_isolates(g):
    # 保存原始边数
    # original_num_edges = g.num_edges()
    #
    # # Step 1: Ensure no isolates by giving each node at least one edge
    # nodes = list(g.get_vertices())
    # for v in nodes:
    #     if g.vertex(v).out_degree() == 0:
    #         # 随机选择另一个节点来连接
    #         u = v
    #         while u == v or g.edge(u, v):
    #             u = random.choice(nodes)
    #         g.add_edge(v, u)
    #
    # # Step 2: Adjust the number of edges to match the original number
    # current_num_edges = g.num_edges()
    #
    # if current_num_edges > original_num_edges:
    #     # 如果当前边数超过原始边数，则随机删除多余的边
    #     edges_to_remove = current_num_edges - original_num_edges
    #     edges = list(g.edges())
    #     for _ in range(edges_to_remove):
    #         e = random.choice(edges)
    #         g.remove_edge(e)
    #         edges.remove(e)
    #
    # elif current_num_edges < original_num_edges:
    #     # 如果当前边数少于原始边数，则随机添加缺少的边
    #     edges_to_add = original_num_edges - current_num_edges
    #     while g.num_edges() < original_num_edges:
    #         u, v = random.sample(nodes, 2)
    #         if not g.edge(u, v):  # 确保边不存在
    #             g.add_edge(u, v)

    # Step 3: Perform random rewiring
    generation.random_rewire(g, model="configuration")


def relaxed_degree_preserving_redistribute_network(g, lattice_dim=2):
    new_g = g.copy()
    new_g.clear_edges()

    n = new_g.num_vertices()
    side = int(np.ceil(n ** (1 / lattice_dim)))
    pos = new_g.new_vertex_property("vector<double>")

    for i, v in enumerate(new_g.vertices()):
        coords = np.unravel_index(i, [side] * lattice_dim)
        pos[v] = coords

    tree = cKDTree([pos[v].a for v in new_g.vertices()])

    nodes_to_connect = [(v, g.vertex(v).out_degree()) for v in g.vertices()]
    max_iterations = len(nodes_to_connect) * 10
    iteration_count = 0
    difficult_nodes = set()

    while nodes_to_connect and iteration_count < max_iterations:
        v, degree_needed = nodes_to_connect.pop(0)
        if degree_needed == 0:
            continue

        search_k = min(n, max(100, 3 * degree_needed))

        distances, indices = tree.query(pos[v].a, k=search_k)

        probs = 1 / (1 + distances) ** 2
        probs[indices == v] = 0

        valid_mask = np.array([
            new_g.vertex(i).out_degree() < g.vertex(i).out_degree() and
            i != int(v) and
            not new_g.edge(v, new_g.vertex(i))
            for i in indices
        ])
        valid_indices = indices[valid_mask]
        valid_probs = probs[valid_mask]

        if len(valid_indices) == 0:
            # 放宽连接条件，考虑所有还需要连接的节点
            all_valid_nodes = [
                u for u in new_g.vertices()
                if u != v and
                   new_g.vertex(u).out_degree() < g.vertex(u).out_degree() and
                   not new_g.edge(v, u)
            ]

            if all_valid_nodes:
                u = np.random.choice(all_valid_nodes)
                new_g.add_edge(v, u)
                degree_needed -= 1

                # 更新被连接节点的度数
                for i, (node, deg) in enumerate(nodes_to_connect):
                    if node == u:
                        if deg > 1:
                            nodes_to_connect[i] = (node, deg - 1)
                        else:
                            nodes_to_connect.pop(i)
                        break
            else:
                difficult_nodes.add(int(v))
                nodes_to_connect.append((v, degree_needed))
            continue

        u = np.random.choice(valid_indices, size=1, p=valid_probs / np.sum(valid_probs))[0]

        new_g.add_edge(v, new_g.vertex(u))
        degree_needed -= 1

        if degree_needed > 0:
            nodes_to_connect.append((v, degree_needed))

        for i, (node, deg) in enumerate(nodes_to_connect):
            if node == new_g.vertex(u):
                if deg > 1:
                    nodes_to_connect[i] = (node, deg - 1)
                else:
                    nodes_to_connect.pop(i)
                break

        iteration_count += 1

    # 处理剩余的困难节点
    if difficult_nodes:
        print(f"Handling {len(difficult_nodes)} difficult nodes")
        for v in list(difficult_nodes):
            degree_needed = g.vertex(v).out_degree() - new_g.vertex(v).out_degree()
            if degree_needed <= 0:
                difficult_nodes.remove(v)
                continue

            all_nodes = list(new_g.vertices())
            random.shuffle(all_nodes)
            for u in all_nodes:
                if u != v and not new_g.edge(v, u):
                    new_g.add_edge(v, u)
                    degree_needed -= 1
                    if degree_needed == 0:
                        difficult_nodes.remove(v)
                        break

    if iteration_count == max_iterations:
        print(f"Warning: Maximum iterations ({max_iterations}) reached.")

    if difficult_nodes:
        print(f"Warning: {len(difficult_nodes)} nodes could not be fully connected.")

    return new_g, pos


# def improved_relaxed_degree_preserving_redistribute_network(g, lattice_dim=2, max_iterations=1000000):
#     new_g = g.copy()
#     new_g.clear_edges()
#
#     n = new_g.num_vertices()
#     side = int(np.ceil(n ** (1 / lattice_dim)))
#     pos = new_g.new_vertex_property("vector<double>")
#
#     for i, v in enumerate(new_g.vertices()):
#         coords = np.unravel_index(i, [side] * lattice_dim)
#         pos[v] = coords
#
#     tree = cKDTree([pos[v].a for v in new_g.vertices()])
#
#     nodes_to_connect = [(v, g.vertex(v).out_degree()) for v in g.vertices()]
#     iteration_count = 0
#     difficult_nodes = set()
#
#     while nodes_to_connect and iteration_count < max_iterations:
#         # print("iteration count ", iteration_count)
#         v, degree_needed = nodes_to_connect.pop(0)
#         if degree_needed == 0:
#             continue
#
#         search_k = min(n, max(100, 3 * degree_needed))
#
#         distances, indices = tree.query(pos[v].a, k=search_k)
#
#         valid_indices = [i for i in indices if
#                          new_g.vertex(i).out_degree() < g.vertex(i).out_degree() and
#                          i != int(v) and
#                          not new_g.edge(v, new_g.vertex(i))]
#
#         if not valid_indices:
#             all_valid_nodes = [u for u in new_g.vertices()
#                                if u != v and
#                                new_g.vertex(u).out_degree() < g.vertex(u).out_degree() and
#                                not new_g.edge(v, u)]
#
#             if all_valid_nodes:
#                 u = np.random.choice(all_valid_nodes)
#                 new_g.add_edge(v, u)
#                 degree_needed -= 1
#                 update_node_degrees(nodes_to_connect, u)
#             else:
#                 difficult_nodes.add(int(v))
#
#             if degree_needed > 0:
#                 nodes_to_connect.append((v, degree_needed))
#         else:
#             u = np.random.choice(valid_indices)
#             new_g.add_edge(v, new_g.vertex(u))
#             degree_needed -= 1
#             update_node_degrees(nodes_to_connect, u)
#
#             if degree_needed > 0:
#                 nodes_to_connect.append((v, degree_needed))
#
#         iteration_count += 1
#
#     # 处理剩余的困难节点
#     handle_difficult_nodes(difficult_nodes, new_g, g)
#
#     if iteration_count == max_iterations:
#         print(f"Warning: Maximum iterations ({max_iterations}) reached.")
#
#     if difficult_nodes:
#         print(f"Warning: {len(difficult_nodes)} nodes could not be fully connected.")
#
#     return new_g, pos


def improved_relaxed_degree_preserving_redistribute_network(g, lattice_dim=2, max_iterations=1000000):
    new_g = g.copy()
    new_g.clear_edges()

    n = new_g.num_vertices()
    side = int(np.ceil(n ** (1 / lattice_dim)))
    pos = new_g.new_vertex_property("vector<double>")

    for i, v in enumerate(new_g.vertices()):
        coords = np.unravel_index(i, [side] * lattice_dim)
        pos[v] = coords

    tree = cKDTree([pos[v].a for v in new_g.vertices()])

    nodes_to_connect = [(v, g.vertex(v).out_degree()) for v in g.vertices()]
    iteration_count = 0
    difficult_nodes = set()

    while nodes_to_connect and iteration_count < max_iterations:
        v, degree_needed = nodes_to_connect.pop(0)
        if degree_needed == 0:
            continue

        search_k = min(n, max(100, 3 * degree_needed))

        distances, indices = tree.query(pos[v].a, k=search_k)

        valid_indices = [i for i in indices if
                         new_g.vertex(i).out_degree() < g.vertex(i).out_degree() and
                         i != int(v) and
                         not new_g.edge(v, new_g.vertex(i))]

        if not valid_indices:
            all_valid_nodes = [u for u in new_g.vertices()
                               if u != v and
                               new_g.vertex(u).out_degree() < g.vertex(u).out_degree() and
                               not new_g.edge(v, u)]

            if all_valid_nodes:
                u = np.random.choice(all_valid_nodes)
                new_g.add_edge(v, u)
                degree_needed -= 1
                update_node_degrees(nodes_to_connect, u)
            else:
                difficult_nodes.add(int(v))

            if degree_needed > 0:
                nodes_to_connect.append((v, degree_needed))
        else:
            u = np.random.choice(valid_indices)
            new_g.add_edge(v, new_g.vertex(u))
            degree_needed -= 1
            update_node_degrees(nodes_to_connect, u)

            if degree_needed > 0:
                nodes_to_connect.append((v, degree_needed))
        iteration_count += 1

    # 检查是否达到最大迭代次数
    if iteration_count == max_iterations:
        unconnected_nodes = len(nodes_to_connect)
        total_missing_connections = sum(degree for _, degree in nodes_to_connect)
        print(f"Warning: Maximum iterations ({max_iterations}) reached.")
        print(f"Unconnected nodes: {unconnected_nodes}")
        print(f"Total missing connections: {total_missing_connections}")

        # 应急处理：尝试随机连接剩余的边
        emergency_connections = 0
        for v, degree_needed in nodes_to_connect:
            available_nodes = [u for u in new_g.vertices()
                               if u != v and not new_g.edge(v, u)]
            connections = min(degree_needed, len(available_nodes))
            for u in np.random.choice(available_nodes, connections, replace=False):
                new_g.add_edge(v, u)
                emergency_connections += 1

        print(f"Emergency connections made: {emergency_connections}")

    # 计算最终网络的统计信息
    final_unconnected = sum(1 for v in new_g.vertices() if new_g.vertex(v).out_degree() < g.vertex(v).out_degree())
    total_missing = sum(g.vertex(v).out_degree() - new_g.vertex(v).out_degree() for v in new_g.vertices())
    # print(f"Final unconnected nodes: {final_unconnected}")
    # print(f"Final missing connections: {total_missing}")

    return new_g, pos, {
        "max_iterations_reached": iteration_count == max_iterations,
        "unconnected_nodes": final_unconnected,
        "missing_connections": total_missing
    }


def update_node_degrees(nodes_to_connect, u):
    for i in range(len(nodes_to_connect)):
        if nodes_to_connect[i][0] == u:
            if nodes_to_connect[i][1] > 1:
                nodes_to_connect[i] = (u, nodes_to_connect[i][1] - 1)
            else:
                nodes_to_connect.pop(i)
            break


def handle_difficult_nodes(difficult_nodes, new_g, g):
    while difficult_nodes:
        v = difficult_nodes.pop()
        degree_needed = g.vertex(v).out_degree() - new_g.vertex(v).out_degree()
        if degree_needed <= 0:
            continue

        all_nodes = list(new_g.vertices())
        random.shuffle(all_nodes)
        for u in all_nodes:
            if u != v and not new_g.edge(v, u):
                new_g.add_edge(v, u)
                degree_needed -= 1
                if degree_needed == 0:
                    break

        if degree_needed > 0:
            difficult_nodes.add(v)  # 如果仍然无法完全连接，重新添加到困难节点集合


def lattice_based_network_reconstruction(original_graph):

    # 步骤1: 创建原始网络的度分布字典
    original_degrees = {int(v): v.out_degree() for v in original_graph.vertices()}

    # 步骤2: 创建2D方格网络
    n = original_graph.num_vertices()
    side = int(np.ceil(np.sqrt(n)))
    lattice = gt.lattice([side, side])

    # 复制原始图的顶点属性到新图
    for key, prop in original_graph.vertex_properties.items():
        new_prop = lattice.new_vertex_property(prop.value_type())
        lattice.vertex_properties[key] = new_prop
        for v in lattice.vertices():
            if int(v) < n:  # 确保我们不会超出原始图的节点范围
                new_prop[v] = prop[original_graph.vertex(int(v))]

    # 创建位置属性
    pos = lattice.new_vertex_property("vector<double>")
    for v in lattice.vertices():
        x, y = divmod(int(v), side)
        pos[v] = [x, y]

    # 移除多余的节点
    while lattice.num_vertices() > n:
        lattice.remove_vertex(lattice.vertex(lattice.num_vertices() - 1))

    # 步骤3: 初始化lattice度数字典
    lattice_degrees = {int(v): v.out_degree() for v in lattice.vertices()}

    # 步骤4: 创建优先队列
    pq = []
    for v in range(n):
        heapq.heappush(pq, (-abs(original_degrees[v] - lattice_degrees[v]), v))

    # 步骤5: 调整边
    while pq:
        _, v = heapq.heappop(pq)
        if lattice_degrees[v] < original_degrees[v]:
            # 需要添加边
            candidates = [u for u in range(n) if
                          u != v and not lattice.edge(v, u) and lattice_degrees[u] < original_degrees[u]]
            if candidates:
                u = max(candidates, key=lambda x: original_degrees[x] - lattice_degrees[x])
                lattice.add_edge(v, u)
                lattice_degrees[v] += 1
                lattice_degrees[u] += 1
                if lattice_degrees[v] != original_degrees[v]:
                    heapq.heappush(pq, (-abs(original_degrees[v] - lattice_degrees[v]), v))
                if lattice_degrees[u] != original_degrees[u]:
                    heapq.heappush(pq, (-abs(original_degrees[u] - lattice_degrees[u]), u))
        elif lattice_degrees[v] > original_degrees[v]:
            # 需要删除边
            neighbors = list(lattice.vertex(v).out_neighbors())
            if neighbors:
                u = max(neighbors, key=lambda x: lattice_degrees[int(x)] - original_degrees[int(x)])
                lattice.remove_edge(lattice.edge(v, int(u)))
                lattice_degrees[v] -= 1
                lattice_degrees[int(u)] -= 1
                if lattice_degrees[v] != original_degrees[v]:
                    heapq.heappush(pq, (-abs(original_degrees[v] - lattice_degrees[v]), v))
                if lattice_degrees[int(u)] != original_degrees[int(u)]:
                    heapq.heappush(pq, (-abs(original_degrees[int(u)] - lattice_degrees[int(u)]), int(u)))

    # 步骤6: 计算统计信息
    perfect_match = sum(1 for v in range(n) if lattice_degrees[v] == original_degrees[v])
    total_diff = sum(abs(lattice_degrees[v] - original_degrees[v]) for v in range(n))

    # print(f"Perfect matches: {perfect_match}")
    # print(f"Total degree difference: {total_diff}")
    # print(f"Original edges: {original_graph.num_edges()}, Reconstructed edges: {lattice.num_edges()}")

    return lattice, pos, {
        "perfect_matches": perfect_match,
        "total_degree_difference": total_diff,
        "nodes": n
    }