from enum import Enum

import numpy as np

from skimage.filters import butterworth
from skimage.morphology import skeletonize
from skimage.feature import corner_subpix

from rdp import rdp

import math


Connectivity = Enum('Connectivity', 'FOUR EIGHT')

DOOR_WIDTH = 0.85
POINT_THRESHOLD = 0.2
WALL_WIDTH_THRESHOLD = 0.05


class WallPostProcessing():
    def __init__(self, logger):
        self.logger = logger


    def __reset_var(self):
        self.original_mask = None
        self.size = None
        self.mask = None
        self.skel = None

        self.coords = None
        self.coord_edges = None
        self.coord_degree = None

        self.coord_partitions = None
        self.door_bbox = None

        self.pixel_per_meter = -1

        self.graphs = None


    def __call__(self, mask, symbol, **kwargs):
        self.__reset_var()

        self.door_bbox = [a['bounding_box_n'] for a in symbol if a['names'] == 'Door']

        self.original_mask = mask
        self.size = np.array(mask.shape)

        self.logger.log('C1', self.logger.format_image_data(
            name='segmentation-result',
            image=mask
        ))
        self._mask_skeleton(mask.copy(), **kwargs)

        self._find_skeleton_edges(**kwargs)
        self._graph_partitioning(**kwargs)

        self.logger.log('C2', self.logger.format_image_data(
            name='skeletonize',
            image=self.mask
        ))
        self._polygon_simplification(**kwargs)
        self._width_abduction(**kwargs)
        self._angle_refinement(**kwargs)

        coords = self.coords.copy().astype(float)
        
        coords[:,0] = coords[:,0] / self.size[0]
        coords[:,1] = coords[:,1] / self.size[1]
        self.logger.log('C3', {
            'name': 'polygon-simplification',
            'coords': coords.flatten(),
            'edges': self.coord_edges
        })

        self._graph_construction()

        return self.graphs, self.coords, self.coord_edges


    def _is_valid_point(self, point):
        return (point >= np.zeros(2)).all() and (point < self.size).all()


    def _mask_skeleton(
        self,
        mask,
        _mask_skeleton_cutoff_frequency_ratio=0.16,
        _mask_skeleton_order=8.0,
        _mask_skeleton_confidence_value=0.3,
        **kwargs
    ):
        mask[[0, -1],:] = 0
        mask[:,[0, -1]] = 0

        mask = (mask + mask.min()) / (mask.max() + mask.min())

        kwargs = {
            'cutoff_frequency_ratio': _mask_skeleton_cutoff_frequency_ratio,
            'order': _mask_skeleton_order,
            'high_pass': False
        }
        self.mask = butterworth(mask, **kwargs) >= _mask_skeleton_confidence_value

        door_widths = []
        for bbox in self.door_bbox:
            x1 = int(self.size[1] * bbox[0])
            y1 = int(self.size[0] * bbox[1])
            x2 = int(self.size[1] * bbox[2])
            y2 = int(self.size[0] * bbox[3])
            width = max(x2 - x1, y2 - y1)
            door_widths.append(width)
            self.mask[y1:y2, x1:x2] = 1
        median_width = np.median(door_widths)
        self.pixel_per_meter = median_width / DOOR_WIDTH
        print(door_widths, median_width, self.pixel_per_meter)
        
        self.skel = skeletonize(self.mask)


    def _find_skeleton_edges(
        self,
        _skeleton_edge_connectivity_type=Connectivity.EIGHT,
        **kwargs
    ):
        assert (_skeleton_edge_connectivity_type in [Connectivity.FOUR, Connectivity.EIGHT])

        coords = np.argwhere(self.skel)
        coords_idx = np.zeros_like(self.skel) - 1
        coords_LEN = len(coords)

        for idx,[x,y] in enumerate(coords):
            coords_idx[x,y] = idx

        if _skeleton_edge_connectivity_type == Connectivity.EIGHT:
            connectivity = np.array([
                [0, 0, 1, 1, 1, -1, -1, -1],
                [1, -1, 0, 1, -1, 0, 1, -1]
            ]).T
        else:
            connectivity = np.array([
                [0, 0, 1, -1],
                [1, -1, 0, 0]
            ]).T

        coord_edges = []
        coord_degree = np.zeros((coords_LEN), dtype=np.int8)

        for _ in range(coords_LEN):
            coord_edges.append([])

        for idx,point in enumerate(coords):
            for connection in connectivity:
                next_point = point + connection

                if not self._is_valid_point(next_point):
                    continue

                next_target = coords_idx[next_point[0], next_point[1]]

                if next_target > -1:
                    coord_edges[idx].append(next_target)
                    coord_degree[idx] += 1
        
        self.coords = coords
        self.coord_edges = coord_edges
        self.coord_degree = coord_degree


    def _graph_partitioning(
        self,
        **kwargs
    ):
        coords_LEN = len(self.coords)

        static_points = np.where(self.coord_degree > 2)[0]
        if static_points.shape[0] == 0:
            static_points = np.random.choice(np.where(self.coord_degree > 1)[0], 1)

        # belum dilanjut
        visited = np.zeros((coords_LEN), dtype=np.bool_)
        self.coord_partitions = []

        for point in static_points:
            if visited[point]:
                continue

            queue = []
            queue.append(point)
            while len(queue) > 0:
                current = queue.pop(0)
                visited[current] = True

                for next_coord in self.coord_edges[current]:
                    if visited[next_coord]:
                        continue

                    # create new partition
                    partition = [current]
                    part_current = current
                    part_next = next_coord

                    while True:
                        part_current = part_next
                        partition.append(part_current)
                        if self.coord_degree[part_current] != 2 or part_current == current:
                            break

                        visited[part_current] = True
                        for conn in self.coord_edges[part_current]:
                            if visited[conn]:
                                continue
                            part_next = conn


                    self.coord_partitions.append(partition)
                    if self.coord_degree[part_current] > 2:
                        queue.append(part_current)


    def _polygon_simplification(
        self,
        _rdp_epsilon=1.0,
        _corner_subpix_window_size=48,
        _nms_radius=None,
        _nms_threshold=0.5,
        **kwargs
    ):
        if _nms_radius is None:
            _nms_radius = int(self.pixel_per_meter * POINT_THRESHOLD)
        coords_LEN = len(self.coords)
        connections = []
        used_coords = []
        for partition in self.coord_partitions:
            partition = np.array(partition)
            partition_mask = rdp(self.coords[partition], epsilon=_rdp_epsilon, return_mask=True)
            new_partition = partition[partition_mask]
            used_coords_partition = []
            for coord in new_partition:
                used_coords_partition.append(coord)
            for idx in range(1, len(new_partition)):
                connections.append([new_partition[idx - 1], new_partition[idx]])
            used_coords += used_coords_partition

        inverse_used_coords = np.zeros((coords_LEN), dtype=np.int32) - 1
        for i,coord in enumerate(used_coords):
            inverse_used_coords[coord] = i
        for i,[u,v] in enumerate(connections):
            connections[i] = [
                inverse_used_coords[u],
                inverse_used_coords[v]
            ]

        before_subpix_coords = self.coords[used_coords]
        coords = corner_subpix(self.skel, before_subpix_coords, window_size=_corner_subpix_window_size).round()
        for idx,[x,y] in enumerate(coords):
            if math.isnan(x):
                coords[idx] = before_subpix_coords[idx]
        coords = coords.astype(np.int32)

        coords, mask_coords = self._non_max_suppression(coords, _nms_radius, _nms_threshold)

        connections = np.unique(
            np.array([
                [mask_coords[u], mask_coords[v]] for [u,v] in connections
            ])
        , axis=0)

        delete_connection = np.where(connections[:,0] == connections[:,1])
        connections = np.delete(connections, delete_connection, axis=0)

        self.coord_edges = connections
        self.coords = coords


    def _non_max_suppression(self, center, radius, overlapThresh):
        if len(center) == 0:
            return []
        double_rad_ = 2 * radius + 1
        center = center.astype(np.float32)
        pick = []
        
        area = double_rad_ * double_rad_
        idxs = np.argsort(center[:,1])
        mapper = np.empty(center.shape[0], dtype=np.int32)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            new_id = len(pick)
            pick.append(i)

            x1 = np.maximum(center[i,0], center[idxs[:last],0])
            y1 = np.maximum(center[i,1], center[idxs[:last],1])
            x2 = np.minimum(center[i,0], center[idxs[:last],0])
            y2 = np.minimum(center[i,1], center[idxs[:last],1])

            w = np.clip(x2 - x1 + double_rad_, a_min=0, a_max=None)
            h = np.clip(y2 - y1 + double_rad_, a_min=0, a_max=None)

            overlap = (w * h) / area

            must_delete = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
            mapper[idxs[must_delete]] = new_id
            idxs = np.delete(idxs, must_delete)
            
        return center[pick].round().astype(np.int32), mapper
    
    
    def _width_abduction(
        self,
        _width_abduction_sample_size=100,
        **kwargs
    ):
        coord_edges = []
        for [u,v] in self.coord_edges:
            point_vector = self.coords[u] - self.coords[v]
            dist = np.linalg.norm(point_vector)
            normalized_vector = point_vector / dist
            perpendicular_vector = normalized_vector[::-1] * np.array([1, -1])

            sample = min(_width_abduction_sample_size, round(dist) + 1)
            coords = [(self.coords[u] + point_vector * (i / sample)).round().astype(np.int32) for i in range(sample)]

            sample_width = np.array([self._width_search_naive(c, perpendicular_vector) for c in coords])
            median_width = np.median(sample_width).round()
            if median_width > 0:
                coord_edges.append([u, v, median_width])
        self.coord_edges = np.array(coord_edges, dtype=np.int32)
        print(self.coord_edges)


    def _width_search_naive(self, c, p):
        r = 0
        c1, c2 = c, c
        if not (self._is_valid_point(c1) and self._is_valid_point(c2)):
            return r
        while self.mask[c1[0], c1[1]] and self.mask[c2[0], c2[1]]:
            r += 1
            c1 = c - (r * p).astype(np.int32)
            c2 = c + (r * p).astype(np.int32)

            if not (self._is_valid_point(c1) and self._is_valid_point(c2)):
                return r
        return r

    
    def _angle_refinement(
        self,
        _angle_divider_count=32,
        _refined_threshold=0.7,
        **kwargs
    ):
        grad16 = np.around([[
            np.sin(i * np.pi * 2 / _angle_divider_count),
            np.cos(i * np.pi * 2 / _angle_divider_count),
        ] for i in range(_angle_divider_count)], decimals=5)

        change = self.coord_edges.shape[0]
        while change > (1 - _refined_threshold) * self.coord_edges.shape[0]:
            change = 0
            for [u,v,w] in self.coord_edges:
                cu = self.coords[u]
                cv = self.coords[v]

                dist = np.linalg.norm(cu - cv)
                norm_vector = (cu - cv) / dist
                preferred_grad_index = np.argmin(np.abs(grad16 - norm_vector).sum(axis=1))
                preferred_grad = grad16[preferred_grad_index]

                fixed = (cu + cv) / 2
                dv = fixed - dist * preferred_grad / 2
                du = fixed + dist * preferred_grad / 2

                dv = dv.round().astype(np.int32)
                du = du.round().astype(np.int32)

                if not ((du == cu).all() and (dv == cv).all()):
                    change += 1
                    self.coords[u] = du
                    self.coords[v] = dv


    def _graph_construction(self):
        pass