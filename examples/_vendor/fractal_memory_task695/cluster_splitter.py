import math
from typing import List, Dict, Any, Optional, Tuple
from .micro_list import MicroItem
from .macro_graph import cosine_similarity

class ClusterSplitter:
    """Unsupervised Mathematical Clustering Pass with Dynamic Floor Threshold Scaling.
    Smaller/younger memory graphs split easily to form initial structure; mature graphs
    scale the similarity floor upward to prevent node inundation.
    """
    def __init__(self, min_cluster_size: int = 2, base_threshold: float = 0.45, max_threshold: float = 0.68):
        self.min_cluster_size = min_cluster_size
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold

    def get_dynamic_threshold(self, total_graph_nodes: int) -> float:
        """Calculates dynamic similarity floor threshold based on total graph node count."""
        if total_graph_nodes <= 3:
            return self.base_threshold
        scaled = self.base_threshold + 0.05 * math.log2(1.0 + (total_graph_nodes / 4.0))
        return min(self.max_threshold, round(scaled, 4))

    def find_clusters(self, items: List[MicroItem], total_graph_nodes: int = 1) -> List[List[MicroItem]]:
        """Groups items into clusters using dynamic distance thresholding."""
        valid_items = [item for item in items if item.embedding]
        if len(valid_items) < self.min_cluster_size * 2:
            return [items]

        dynamic_sim_threshold = self.get_dynamic_threshold(total_graph_nodes)

        n = len(valid_items)
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(valid_items[i].embedding, valid_items[j].embedding)
                if sim >= dynamic_sim_threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        clusters_idx: List[List[int]] = []

        for i in range(n):
            if i not in visited:
                component = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters_idx.append(component)

        clusters: List[List[MicroItem]] = []
        unassigned: List[MicroItem] = []

        for group in clusters_idx:
            group_items = [valid_items[idx] for idx in group]
            if len(group_items) >= self.min_cluster_size:
                clusters.append(group_items)
            else:
                unassigned.extend(group_items)

        if len(clusters) > 1 and unassigned:
            for item in unassigned:
                best_cluster_idx = 0
                best_sim = -1.0
                for c_idx, cluster in enumerate(clusters):
                    avg_sim = sum(cosine_similarity(item.embedding, c_item.embedding) for c_item in cluster) / len(cluster)
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_cluster_idx = c_idx
                clusters[best_cluster_idx].append(item)

        items_without_embed = [item for item in items if not item.embedding]
        if items_without_embed:
            if clusters:
                clusters[0].extend(items_without_embed)
            else:
                clusters = [items_without_embed]

        return clusters if len(clusters) > 1 else [items]
