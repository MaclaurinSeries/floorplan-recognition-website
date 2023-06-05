from enum import Enum

import numpy as np

from skimage.filters import butterworth
from skimage.morphology import skeletonize, area_closing, binary_dilation
from skimage.measure import label
from skimage.feature import corner_subpix
from skimage.draw import line

from PIL import Image, ImageDraw

from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union, polygonize
from scipy import ndimage
from rdp import rdp
import jenkspy

from rasterio.features import shapes

import torch
from torch_geometric.data import Data

import math
from . import HouseConfig as Mapper
from .utils import save
import cv2


Connectivity = Enum('Connectivity', 'FOUR EIGHT')


class WallPostProcessing():
    def __init__(self, logger=None):
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
        self.window_bbox = None
        self.furniture_bbox = None

        self.pixel_per_meter = -1

        self.room_mask = None
        self.room_cnt = -1
        self.graph = None

        self.room_polygon = None


    def __call__(self, mask, symbol, floor_idx, **kwargs):
        self.__reset_var()

        self.door_bbox = []
        self.window_bbox = []
        self.furniture_bbox = []

        for b in symbol:
            if b['names'] == 'Door':
                self.door_bbox.append(b['bounding_box_n'])
            elif b['names'] == 'Window':
                self.window_bbox.append(b['bounding_box_n'])
            else:
                self.furniture_bbox.append({
                    "label": b['label'] - 2,
                    "prob": b['prob'],
                    "bbox": b['bounding_box_n']
                })

        self.original_mask = mask
        self.size = np.array(self.original_mask.shape)

        self.logger.log('C1', self.logger.format_image_data(
            name='segmentation-result',
            floor=floor_idx,
            image=self.original_mask
        ))
        self._mask_skeleton(self.original_mask.copy(), **kwargs)

        self._find_skeleton_edges(**kwargs)

        self._graph_partitioning(**kwargs)
        self._polygon_simplification(**kwargs)
        
        skel = self.skel.copy()
        thicc = int(round(self.size.max() / 400))
        for i in range(thicc):
            skel = binary_dilation(skel)
        skel = np.concatenate((skel[:,:,None], skel[:,:,None], skel[:,:,None]), axis=2).astype(np.float32)
        for coord in self.coords:
            cl = np.random.rand(1, 3)
            for i in range(thicc * 4):
                for j in range(thicc * 4):
                    if self._is_valid_point(np.array([coord[0]-i,coord[1]-j])):
                        skel[coord[0]-i,coord[1]-j,:] = cl
                    if self._is_valid_point(np.array([coord[0]-i,coord[1]+j])):
                        skel[coord[0]-i,coord[1]+j,:] = cl
                    if self._is_valid_point(np.array([coord[0]+i,coord[1]-j])):
                        skel[coord[0]+i,coord[1]-j,:] = cl
                    if self._is_valid_point(np.array([coord[0]+i,coord[1]+j])):
                        skel[coord[0]+i,coord[1]+j,:] = cl
        self.logger.log('C2', self.logger.format_image_data(
            name='skeletonize',
            floor=floor_idx,
            image=skel
        ))
        # save(skel)
        self._thickness_abduction(**kwargs)

        # im = Image.new("RGB", (skel.shape[1], skel.shape[0]))
        # draw = ImageDraw.Draw(im)
        # for u,v,w in self.coord_edges:
        #     u = self.coords[u]
        #     v = self.coords[v]
        #     draw.line(tuple(u[::-1]) + tuple(v[::-1]), fill=(255, 255, 255), width=w)
        # im.save('.\\saved\\3x.png')

        
        self._wall_selection(**kwargs)
        self._angle_refinement(**kwargs)
        self._wall_regularization(**kwargs)

        # im = Image.new("RGB", (skel.shape[1], skel.shape[0]))
        # draw = ImageDraw.Draw(im)
        # for u,v,w in self.coord_edges:
        #     u = self.coords[u]
        #     v = self.coords[v]
        #     draw.line(tuple(u[::-1]) + tuple(v[::-1]), fill=(255, 255, 255), width=w)
        # im.save('.\\saved\\4x.png')
        
        self._door_abduction(**kwargs)
        self._wall_polygonize(**kwargs)

        coords = self.coords.copy().astype(float)

        coords[:,0] = coords[:,0] / self.size[0]
        coords[:,1] = coords[:,1] / self.size[1]

        self.logger.log('C3', {
            'name': 'polygon-simplification',
            'floor': floor_idx,
            'res': {
                'shape': self.size,
                'ppm': self.pixel_per_meter
            },
            'coords': coords.flatten(),
            'edges': self.coord_edges
        })
        self._room_masking(**kwargs)
        self._graph_construction(**kwargs)

        self.room_polygon = []
        for room in range(self.room_cnt):
            binary = self.room_mask == (room + 1)
            binary = ndimage.binary_fill_holes(binary)
            poly = (s for i, (s, v) in enumerate(shapes(binary.astype(np.uint8), mask=binary)))

            pts = None
            max_area = 0

            for l in poly:
                if Polygon(l['coordinates'][0]).area > max_area:
                    max_area = Polygon(l['coordinates'][0]).area
                    pts = np.array(l['coordinates'][0], dtype=np.float32).T[::-1]
            pts[0] /= self.size[0]
            pts[1] /= self.size[1]
            self.room_polygon.append(pts)

        return self.graph, self.room_polygon, self.coords, self.coord_edges, self.pixel_per_meter


    def _is_valid_point(self, point):
        return (point >= np.zeros(2, dtype=np.int32)).all(-1) and (point < self.size).all(-1)


    def _mask_skeleton(
      self,
      mask,
      _mask_skeleton_cutoff_frequency_ratio=0.16,
      _mask_skeleton_order=8.0,
      _mask_skeleton_confidence_value=0.3,
      _door_width_regulation=0.85,
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
      # save(self.mask)

      door_widths = []
      for bbox in self.door_bbox + self.window_bbox:
        x1 = int(self.size[1] * bbox[0])
        y1 = int(self.size[0] * bbox[1])
        x2 = int(self.size[1] * bbox[2])
        y2 = int(self.size[0] * bbox[3])
        width = max(x2 - x1, y2 - y1)
        door_widths.append(width)
        self.mask[y1:y2, x1:x2] = 1
      median_width = np.median(door_widths)
      self.pixel_per_meter = median_width / _door_width_regulation
      
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
      
      # drawing
      # d = 3
      # ms = np.zeros((*self.size, 3))
      # for partition in self.coord_partitions:
      #     color = np.random.rand(1, 3)
      #     for coord in self.coords[partition]:
      #         ms[coord[0] - d: coord[0] + d, coord[1] - d: coord[1] + d, :] = color
      # save(ms)



    def _polygon_simplification(
        self,
        _rdp_epsilon=0.1,
        _corner_subpix_window_size=48,
        _nms_radius=0.2,
        _nms_threshold=0.5,
        **kwargs
    ):
      _rdp_epsilon = self.pixel_per_meter * _rdp_epsilon
      _nms_radius = int(self.pixel_per_meter * _nms_radius)
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

      refined = []
      for point in before_subpix_coords:
        if point[0] <=0:
          point[0] = 1
        if point[1] <= 0:
          point[1] = 1
        if point[0] >= self.size[0] - 1:
          point[0] = self.size[0] - 2
        if point[1] >= self.size[1] - 1:
          point[1] = self.size[1] - 2
        refined.append(point)
      before_subpix_coords = np.array(refined)

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
    
    
    def _thickness_abduction(
        self,
        _thickness_abduction_sample_size=100,
        **kwargs
    ):
      coord_edges = []
      for [u,v] in self.coord_edges:
        point_vector = self.coords[u] - self.coords[v]
        dist = np.linalg.norm(point_vector)
        normalized_vector = point_vector / dist
        perpendicular_vector = normalized_vector[::-1] * np.array([1, -1])

        sample = min(_thickness_abduction_sample_size, int(dist) + 1)
        coords = [(self.coords[v] + (point_vector * i / sample)).round().astype(np.int32) for i in range(sample)]

        sample_width = np.array([self._thickness_search_naive(c, perpendicular_vector) for c in coords])
        median_width = np.median(sample_width).round()
        wall_thickness = 2 * median_width + 1
        coord_edges.append([u, v, wall_thickness])
      self.coord_edges = np.array(coord_edges, dtype=np.int32)


    def _thickness_search_naive(self, c, p):
        c1, c2 = c.copy(), c.copy()
        white_point = lambda x: self._is_valid_point(x) and self.mask[x[0], x[1]]

        outer_r = 0
        inner_r = 0
        inc = 1
        while white_point(c1) or white_point(c2):
            outer_r += inc
            if white_point(c1) and white_point(c2):
                inner_r += inc
            c1 = (c - outer_r * p).astype(np.int32)
            c2 = (c + outer_r * p).astype(np.int32)

        return int((outer_r + inner_r) / 2)


    def _wall_selection(
        self,
        _wall_thickness_threshold=0.08,
        _disconnected_wall_length_threshold=0.3,
        _connected_wall_length_threshold=0.1,
        **kwargs
    ):
        coords = self.coords.copy()
        coord_edges = []
        mask = np.arange(len(coords))
        used_coords = np.zeros(len(coords), dtype=np.int32) - 1
        coord_degree = np.zeros(len(coords), dtype=np.int32)

        def find_mask(x):
            if mask[x] != x:
                mask[x] = find_mask(mask[x])
            return mask[x]

        for [u,v,w] in self.coord_edges:
            coord_degree[u] += 1
            coord_degree[v] += 1
        
        for [u,v,w] in self.coord_edges:
            dis = coord_degree[u] <= 1 or coord_degree[v] <= 1
            if w < _wall_thickness_threshold * self.pixel_per_meter:
                continue

            c1 = coords[u]
            c2 = coords[v]
            d = np.linalg.norm(c2 - c1)

            if (dis and d >= _disconnected_wall_length_threshold * self.pixel_per_meter) or (not dis and d >= _connected_wall_length_threshold * self.pixel_per_meter):
                coord_edges.append([u,v,w])
                continue

            if not dis:
                u = find_mask(u)
                v = find_mask(v)
                mask[u] = v

        used_idx = 0
        for idx,[u,v,w] in enumerate(coord_edges):
            u = find_mask(u)
            v = find_mask(v)
            if used_coords[u] < 0:
                used_coords[u] = used_idx
                used_idx += 1
            if used_coords[v] < 0:
                used_coords[v] = used_idx
                used_idx += 1
            coord_edges[idx][0] = used_coords[u]
            coord_edges[idx][1] = used_coords[v]

        inv_used_coord = np.zeros(used_idx, dtype=np.int32)
        for i,num in enumerate([u for u in used_coords]):
            if num > -1:
                inv_used_coord[num] = i

        coords = coords[inv_used_coord]
        self.coords = coords
        self.coord_edges = np.array(coord_edges)

    
    def _angle_refinement(
        self,
        _angle_divider_count=24,
        _refined_threshold=0.7,
        _max_iter=10,
        **kwargs
    ):
      grad16 = np.around([[
        np.sin(i * np.pi * 2 / _angle_divider_count),
        np.cos(i * np.pi * 2 / _angle_divider_count),
      ] for i in range(_angle_divider_count)], decimals=5)

      change = self.coord_edges.shape[0]
      iteration = 0
      while change > (1 - _refined_threshold) * self.coord_edges.shape[0]:
        iteration += 1
        if iteration > _max_iter:
          break
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


    def _wall_regularization(
        self,
        _jenks_gvf_threshold=0.5,
        **kwargs
    ):
      thickness = np.sort([w for [u,v,w] in self.coord_edges])
      
      sdam = thickness.var()
      # jenks natural break optimization (k means for 1D)
      gvf = 0
      n_class = 2
      while gvf < _jenks_gvf_threshold:
        split = jenkspy.jenks_breaks(thickness, n_classes=n_class)
        sdcm = 0

        sum_c = []
        for _ in range(len(split) - 1):
          sum_c.append([])

        for thicc in thickness:
          for i in range(1,len(split)):
            if thicc >= split[i - 1] and thicc < split[i]:
              sum_c[i - 1].append(thicc)
              break
        
        for i in range(len(sum_c)):
          sum_c[i] = np.array(sum_c[i])
          sdcm += sum_c[i].var()

        gvf = (sdam + sdcm) / sdam
        n_class += 1

      mean_c = []
      for s in sum_c:
        mean_c.append(s.mean())
      
      max_w = 1 * self.pixel_per_meter
      for idx,[u,v,w] in enumerate(self.coord_edges):
        for i in range(1,len(split)):
          if w >= split[i - 1] and w < split[i]:
            self.coord_edges[idx][2] = min(mean_c[i - 1], max_w)
            break


    def _point_distance_to_segment(
        self,
        point,
        segment_u,
        segment_v
    ):
        segment_vector = segment_v - segment_u
        point_vector = point - segment_u
        proj = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
        proj = min(1, max(0, proj))
        nearest = segment_u + proj * segment_vector
        return np.linalg.norm(point - nearest), nearest, proj


    def _object_length_abduction(
        self,
        center,
        vector,
        bbox,
        thickness
    ):
      vector = vector / np.linalg.norm(vector)
      bbox_poly = Polygon([(bbox[1], bbox[0]), (bbox[1], bbox[2]), (bbox[3], bbox[2]), (bbox[3], bbox[0])])
      half_w = 1
      current_iou = 0
      max_iou = -1
      while current_iou > max_iou:
        max_iou = current_iou
        half_w += 1
        
        u = center + half_w * vector
        v = center - half_w * vector

        line = LineString([(u[0], u[1]), (v[0], v[1])])
        obj_poly = line.buffer(thickness, cap_style=2)

        current_iou = bbox_poly.intersection(obj_poly).area / bbox_poly.union(obj_poly).area
      return 2 * half_w


    def _door_abduction(
        self,
        **kwargs
    ):
      coord_edges = []

      for coord_edge in self.coord_edges:
        [u,v,w] = coord_edge.tolist()
        coord_edges.append({
            'vertices': [u, v],
            'thickness_pixel': w,
            'thickness_meter': w / self.pixel_per_meter,
            'doors': [],
            'windows': []
        })

      for llist,nlist in [(self.door_bbox, 'doors'), (self.window_bbox, 'windows')]:
        for bbox in llist:
          bbox[1::2] = bbox[1::2] * self.size[0]
          bbox[0::2] = bbox[0::2] * self.size[1]
          center = np.array([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2])

          nearest_segment = 0
          min_dist = (self.size * self.size).sum()
          nearest_point = None
          projection = None

          for segment_idx,[u,v,w] in enumerate(self.coord_edges):
            d, near, proj = self._point_distance_to_segment(center, self.coords[u], self.coords[v])
            if d < min_dist:
              min_dist = d
              nearest_segment = segment_idx
              nearest_point = near
              projection = proj

          [u,v,w] = self.coord_edges[nearest_segment]
          vector = self.coords[u] - self.coords[v]
          
          length = self._object_length_abduction(nearest_point, vector, bbox, w)

          coord_edges[nearest_segment][nlist].append({
              'center': nearest_point,
              'projection': projection,
              'length': length / self.pixel_per_meter,
              'bbox': bbox,
          })
      self.coord_edges = coord_edges


    def _wall_polygonize(
        self,
        **kwargs
    ):
        inv_coord = []
        for iv in self.coords:
            inv_coord.append([])
        cos_pi_15 = np.cos(np.pi / 12)

        for idx,wall in enumerate(self.coord_edges):
            u,v = wall['vertices']
            inv_coord[u].append({
                'edge_index': idx,
                'loc': 0
            })
            inv_coord[v].append({
                'edge_index': idx,
                'loc': 1
            })
        
        def pseudoangle(dx, dy):
            p = dx / (abs(dx) + abs(dy))
            if dy < 0: return p - 1
            else: return 1 - p
        def get_angle(invc):
            edge_index = invc['edge_index']
            loc_index = invc['loc']
            edge = self.coord_edges[edge_index]
            luar = self.coords[edge['vertices'][1 - loc_index]]
            dalem = self.coords[edge['vertices'][loc_index]]

            return pseudoangle(luar[1] - dalem[1], luar[0] - dalem[0])

        for sgs in inv_coord:
            sorted_wall = sorted(sgs, key=get_angle)
            
            for idx,wall in enumerate(sorted_wall):
                nxt = (idx + 1) % len(sorted_wall)

                edge_index = wall['edge_index']
                loc_index = wall['loc']
                nxt_edge_index = sorted_wall[nxt]['edge_index']
                nxt_loc_index = sorted_wall[nxt]['loc']

                edge = self.coord_edges[edge_index]
                nxt_edge = self.coord_edges[nxt_edge_index]

                if 'polygon' not in edge:
                    edge['polygon'] = [None, None, None, None, None, None]
                    edge['polygon'][1] = self.coords[edge['vertices'][0]]
                    edge['polygon'][4] = self.coords[edge['vertices'][1]]
                if 'polygon' not in nxt_edge:
                    nxt_edge['polygon'] = [None, None, None, None, None, None]
                    nxt_edge['polygon'][1] = self.coords[nxt_edge['vertices'][0]]
                    nxt_edge['polygon'][4] = self.coords[nxt_edge['vertices'][1]]

                vector = self.coords[edge['vertices'][1 - loc_index]] - self.coords[edge['vertices'][loc_index]]
                nxt_vector = self.coords[nxt_edge['vertices'][1 - nxt_loc_index]] - self.coords[nxt_edge['vertices'][nxt_loc_index]]

                thickness = edge['thickness_pixel'] // 2
                nxt_thickness = nxt_edge['thickness_pixel'] // 2

                perpen_vec = vector[::-1] * np.array([1, -1]) # yg di min, x nya
                perpen_vec = perpen_vec * thickness / np.linalg.norm(perpen_vec)

                nxt_perpen_vec = nxt_vector[::-1] * np.array([-1, 1]) # yg di min, y nya
                nxt_perpen_vec = nxt_perpen_vec * nxt_thickness / np.linalg.norm(nxt_perpen_vec)

                av = vector / np.linalg.norm(vector)
                bv = nxt_vector / np.linalg.norm(nxt_vector)
                if abs(np.dot(av, bv)) > (1 + cos_pi_15)/2:
                    edge['polygon'][3 * loc_index + 2] = perpen_vec + self.coords[edge['vertices'][loc_index]]
                    nxt_edge['polygon'][3 * nxt_loc_index] = nxt_perpen_vec + self.coords[nxt_edge['vertices'][nxt_loc_index]]
                    for i in range(6):
                        if self.coord_edges[edge_index]['polygon'][i] is None:
                            self.coord_edges[edge_index]['polygon'][i] = edge['polygon'][i]
                    for i in range(6):
                        if self.coord_edges[nxt_edge_index]['polygon'][i] is None:
                            self.coord_edges[nxt_edge_index]['polygon'][i] = nxt_edge['polygon'][i]
                    continue

                n_vec = np.dot(nxt_perpen_vec, nxt_perpen_vec) / np.dot(vector, nxt_perpen_vec) * vector
                m_vec = np.dot(perpen_vec, perpen_vec) / np.dot(nxt_vector, perpen_vec) * nxt_vector

                disloc = self.coords[edge['vertices'][loc_index]] + n_vec + m_vec

                edge['polygon'][3 * loc_index + 2] = disloc
                nxt_edge['polygon'][3 * nxt_loc_index] = disloc 

                for i in range(6):
                    if self.coord_edges[edge_index]['polygon'][i] is None:
                        self.coord_edges[edge_index]['polygon'][i] = edge['polygon'][i]
                for i in range(6):
                    if self.coord_edges[nxt_edge_index]['polygon'][i] is None:
                        self.coord_edges[nxt_edge_index]['polygon'][i] = nxt_edge['polygon'][i]
        coord_edge = []
        for idx,edge in enumerate(self.coord_edges):
            ls = LineString(edge['polygon'])
            lr = LineString(ls.coords[:] + ls.coords[0:1])

            u = self.coords[edge['vertices'][0]]
            v = self.coords[edge['vertices'][1]]
            vector = v - u
            vector = vector / np.linalg.norm(vector)

            mls = unary_union(lr)
            poly = None
            for idx,polygon in enumerate(polygonize(mls)):
                if idx == 0:
                    poly = polygon
                elif polygon.distance(Point(u[0], u[1])) < 1e-8 and polygon.distance(Point(v[0], v[1])) < 1e-8:
                    poly = polygon
            for obnm in ['doors', 'windows']:
                for idx,obj in enumerate(edge[obnm]):
                    center = obj['center']
                    first_p = center + obj['length'] * vector * self.pixel_per_meter / 2
                    second_p = center - obj['length'] * vector * self.pixel_per_meter / 2

                    ls = LineString([(first_p[0], first_p[1]), (second_p[0], second_p[1])])
                    obj_poly = ls.buffer(edge['thickness_pixel'] * 2, cap_style=2)

                    opoly = obj_poly.intersection(poly)
                    poly = poly.difference(obj_poly)

                    if isinstance(opoly, MultiPolygon):
                        opoly = max(opoly, key=lambda a: a.area)
                    y,x = opoly.exterior.coords.xy
                    y = np.array(y.tolist()) / self.size[0]
                    x = np.array(x.tolist()) / self.size[1]
                    edge[obnm][idx]['polygon'] = np.array([x, y]).T
            polygons = []
            if isinstance(poly, Polygon):
                y,x = poly.exterior.coords.xy
                y = np.array(y.tolist()) / self.size[0]
                x = np.array(x.tolist()) / self.size[1]
                polygons.append(np.array([x, y]).T)
            elif isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    y,x = p.exterior.coords.xy
                    y = np.array(y.tolist()) / self.size[0]
                    x = np.array(x.tolist()) / self.size[1]
                    polygons.append(np.array([x, y]).T)

            edge['polygon'] = polygons
            coord_edge.append(edge)
        self.coord_edges = coord_edge


    def _room_masking(
        self,
        **kwargs
    ):
        pad = int(max([a['thickness_pixel'] for a in self.coord_edges]))
        mask = np.ones(self.size + 2 * pad, dtype=np.int32)
        for edge in self.coord_edges:
            vertices = edge['vertices']
            u = self.coords[vertices[0]]
            v = self.coords[vertices[1]]
            rr, cc = line(u[0]+pad, u[1]+pad, v[0]+pad, v[1]+pad)
            mask[rr,cc] = 0

        mask = label(mask, connectivity=1)[pad:self.size[0]+pad,pad:self.size[1]+pad]
        self.room_mask = mask - 1
        self.room_cnt = self.room_mask.max()


    def _find_room_code(
        self,
        vector,
        start
    ):
        vector = vector / np.linalg.norm(vector)
        current = start.round().astype(np.int32)
        r = 0
        if not self._is_valid_point(current):
            return -1
        while self.room_mask[current[0], current[1]] == -1:
            r += 1
            current = (start + r * vector).round().astype(np.int32)
            if not self._is_valid_point(current):
                return -1
        return self.room_mask[current[0], current[1]]


    def _graph_construction(
        self,
        **kwargs
    ):
      x = np.zeros((self.room_cnt, Mapper.lenIcon() - 2 + Mapper.lenRoom()))
      for furniture in self.furniture_bbox:
        label = furniture['label']
        prob = furniture['prob']
        bbox = furniture['bbox']
        
        center = np.array([
          (bbox[1] + bbox[3]) * self.size[0] / 2,
          (bbox[0] + bbox[2]) * self.size[1] / 2
        ]).round().astype(np.int32)
        
        room_id = self.room_mask[center[0], center[1]] - 1
        if room_id >= 0:
          x[room_id, label] += prob

      edge_index = []
      edge_attr = []
      for cidx,edge in enumerate(self.coord_edges):
        vertices = edge['vertices']
        thickness_pixel = edge['thickness_pixel']
        vector = self.coords[vertices[1]] - self.coords[vertices[0]]

        for idx,door in enumerate(edge['doors']):
          projection = door['projection']
          center = self.coords[vertices[0]] + projection * vector
          perpen_vec = vector[::-1] * np.array([1, -1])
          perpen_vec = perpen_vec / np.linalg.norm(perpen_vec)

          u = self._find_room_code(perpen_vec, center + thickness_pixel * perpen_vec)
          v = self._find_room_code(-perpen_vec, center - thickness_pixel * perpen_vec)
          if u == -1 or v == -1:
            continue

          if u > 0 and v > 0 and [u - 1, v - 1] not in edge_index:
            edge_index.append([u - 1, v - 1])
            edge_index.append([v - 1, u - 1])
            edge_attr.append([0, 1])
            edge_attr.append([0, 1])

          if u > 0 and v > 0:
            self.coord_edges[cidx]['doors'][idx]['connection'] = [int(u - 1), int(v - 1)]
      
      x = torch.tensor(np.array(x), dtype=torch.float).softmax(dim=-1)
      
      if edge_index == []:
        edge_index = None
      else:
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).transpose(0, 1)
      edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
      self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)