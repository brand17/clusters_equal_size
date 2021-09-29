import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean 
import scipy
from scipy.spatial import KDTree
from heapq import heappop, heapify, heappush
from numpy import linalg

def show_chart(coords, ind2cluster, centroids, nclusters, dynamic=True):
  if coords.shape[1] != 2:
    return
  plt.scatter(coords[:, 0], coords[:, 1], c=ind2cluster)
  plt.scatter(centroids[:, 0], centroids[:, 1], c=range(nclusters), s=72*4)
  plt.scatter(real_centroids[:, 0], real_centroids[:, 1], c='0.5', s=72*4)
  if dynamic:
    plt.pause(0.01)
    plt.draw()
  else:
    plt.show()
  plt.clf()

def init_centroids(coords, nclusters):
  left_coords = list(coords)
  import random
  random.seed(0)
  random.shuffle(left_coords)
  centroids = [left_coords.pop()]
  for _ in range(nclusters - 1):
    kdt = KDTree(centroids)
    neighbors = kdt.query(left_coords, k=1)
    D = neighbors[0]**2
    c = random.choices(range(len(D)), weights=D, k=1)[0]
    centroids += [left_coords[c]]
    del left_coords[c]
  centroids = np.array(centroids)
  return centroids
    
def cluster_equal_size(coords, nclusters, show_plt=True):
  from sklearn.cluster import KMeans
  old_int2cluster = None
  nverts = coords.shape[0]
  max_len = nverts / nclusters
  kmeans = KMeans(nclusters)
  kmeans.fit(coords)
  centroids = kmeans.cluster_centers_
  while True:
    dist2centroids = scipy.spatial.distance_matrix(coords, centroids)
    clusters = fill_clasters(nclusters, kmeans.labels_)
    cl_sorted = sorted(range(nclusters), key=lambda x: -len(clusters[x]))
    visited = set()
    for cl_from in cl_sorted[:-1]:
      visited.add(cl_from)
      to_transfer = len(clusters[cl_from]) - claster_len(max_len, cl_from)
      if to_transfer > 0:
        h = []
        for p in clusters[cl_from]:
          l = [(d, c) for d, c in zip(dist2centroids[p], range(nclusters)) if c not in visited]
          dist, cl_to = min(l, key=(lambda x: x[0]))
          heappush(h, (dist, p, cl_to))
        for _ in range(to_transfer):
          _, p, cl_to = heappop(h)
          clusters[cl_to] += [p]
          clusters[cl_from].remove(p)
    ind2cluster = [None for _ in range(nverts)]
    for cl, points in enumerate(clusters):
      for p in points:
        ind2cluster[p] = cl
    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])

    # dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    dists2curr_cl = dist2centroids[range(nverts), ind2cluster]
    # print(sum(dists2curr_cl))
    if True:#np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

def cluster_equal_size_pair_split(coords, nclusters, show_plt=True):
  centroids = init_centroids(coords, nclusters)
  # centroids = coords[np.random.choice(coords.shape[0], nclusters, replace=False), :]
  ind2cluster = [0] * coords.shape[0]
  ranks = [50] * nclusters
  old_int2cluster = None
  while True:
    centroid_distances = scipy.spatial.distance_matrix(centroids, centroids)
    kdt = KDTree(centroids)
    neighbors = kdt.query(coords, k=2)
    closest_centroids = {(i, j): [] for i in range(nclusters - 1) for j in range(i + 1, nclusters)}
    for vert_idx, (dist2centroids, centroid_idx) in enumerate(zip(neighbors[0], neighbors[1])):
      centroid_idx = tuple(centroid_idx)
      if centroid_idx[0] > centroid_idx[1]:
        dist2centroids = [dist2centroids[1], dist2centroids[0]]
        centroid_idx = (centroid_idx[1], centroid_idx[0])
      a, b = dist2centroids
      c = centroid_distances[centroid_idx]
      dist = (a**2 - b**2 + c**2) / (2 * c)
      closest_centroids[centroid_idx] += [(vert_idx, dist)]
    closest_centroids = {k: sorted(v, key=lambda tup: tup[1]) for k, v in closest_centroids.items()} 
    # for (c1, c2), verts  in closest_centroids.items():
    #   if len(verts) != 0:
    #     ranks[c1] += 1
    #     ranks[c2] += 1
    new_ranks = [0] * nclusters
    for (c1, c2), verts  in closest_centroids.items():
      l = len(verts) * ranks[c2] // (ranks[c1] + ranks[c2])
      # l = len(verts) // 2
      for vert in verts[:l]:
        ind2cluster[vert[0]] = c1
        new_ranks[c1] += 1
      for vert in verts[l:]:
        ind2cluster[vert[0]] = c2
        new_ranks[c2] += 1
      pass
    ranks = new_ranks
    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)
    groups = [[] for _ in range(nclusters)]
    for g, c in zip(ind2cluster, coords):
      groups[g] += [c]
    centroids = np.array([sum(l) / len(l) for l in groups])

    dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    # print(sum(dists2curr_cl))
    if np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

def cluster_equal_size_swap(coords, nclusters, show_plt=True):
  clusters, ind2cluster = init_clasters(coords, nclusters)
  old_int2cluster = None
  swap_proposals = [set() for _ in range(nclusters)]
  while True:
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])
    dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    # print(sum(dists2curr_cl))
    kdt = KDTree(centroids)
    _, cl_inds = kdt.query(coords, k=nclusters)
    for point, (curr_cl, nearest_clasters) in enumerate(zip(ind2cluster, cl_inds)):
      for nearest_cl in nearest_clasters:
        if curr_cl != nearest_cl:
          if len(swap_proposals[nearest_cl]) != 0:
            ind2cluster[point] = nearest_cl
            swapped_idx = swap_proposals[nearest_cl].pop()
            ind2cluster[swapped_idx] = curr_cl
            break
          else:
            swap_proposals[curr_cl].add(point)
        else:
          break
    
    clusters = fill_clasters(nclusters, ind2cluster)

    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)

    if np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

def init_clasters(coords, nclusters):
  centroids = init_centroids(coords, nclusters)
  kdt = KDTree(centroids)
  neighbors = kdt.query(coords, k=nclusters)
  dist_diff = [d[0] - d[-1] for d in neighbors[0]]
  h = [(dist, point_idx, cluster_idx) for point_idx, (dist, cluster_idx) in enumerate(zip(dist_diff, neighbors[1]))]
  heapify(h)
  clusters = [[] for _ in range(nclusters)]
  max_len = coords.shape[0] / nclusters
  while len(h):
    _, p_idx, cl_inds = heappop(h)
    for cl_idx in cl_inds:
      if is_claster_not_full(clusters, max_len, cl_idx):
        clusters[cl_idx] += [p_idx]
        break
  ind2cluster = [-1] * coords.shape[0]
  for cl_idx, cl in enumerate(clusters):
    for point in cl:
      ind2cluster[point] = cl_idx
  return clusters, ind2cluster

def is_claster_not_full(clusters, max_len, cl_idx):
  return len(clusters[cl_idx]) < claster_len(max_len, cl_idx)

def claster_len(max_len, cl_idx):
  return round(max_len * (cl_idx + 1)) - round(max_len * (cl_idx))

def fill_clasters(nclusters, ind2cluster):
  clusters = [[] for _ in range(nclusters)]
  for point, claster in enumerate(ind2cluster):
    clusters[claster] += [point]
  return clusters

def cluster_equal_size_elki(coords, nclusters, show_plt=True):
  clusters, ind2cluster = init_clasters(coords, nclusters)
  old_int2cluster = None
  swap_proposals = [set() for _ in range(nclusters)]
  while True:
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])
    dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    # print(sum(dists2curr_cl))
    kdt = KDTree(centroids)
    dist2clasters, cl_inds = kdt.query(coords, k=nclusters)
    dist_diff = [nearest[0] - dist2curr for dist2curr, nearest \
      in zip(dists2curr_cl, dist2clasters)]
    h = [(dd, point_idx, curr_cl, cluster_idx) \
      for point_idx, (dd, cluster_idx, curr_cl) \
      in enumerate(zip(dist_diff, cl_inds, ind2cluster))]
    heapify(h)
    while len(h):
      _, point, claster1, cl_inds_curr = heappop(h)
      swapped = False
      for claster2 in cl_inds_curr:
        if swapped:
          break
        if claster1 != claster2:
          for point2 in swap_proposals[claster2]:
            swap_benefit1 = dist2clasters[point][claster2] - dist2clasters[point][claster1]
            swap_benefit2 = dist2clasters[point2][claster1] - dist2clasters[point2][claster2]
            if swap_benefit1 + swap_benefit2 < 0:
              ind2cluster[point] = claster2
              ind2cluster[point2] = claster1
              swap_proposals[claster2].remove(point2)
              swapped = True
              break
        else:
          break
      if not swapped:
        swap_proposals[claster1].add(point)

    clusters = fill_clasters(nclusters, ind2cluster)

    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)
      
    if np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

def cluster_equal_size_detect_cycles(coords, nclusters, show_plt=False):
  clusters, ind2cluster = init_clasters(coords, nclusters)
  old_int2cluster = None
  while True:
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])
    # kdt = KDTree(centroids)
    # dist2clasters, cl_inds = kdt.query(coords, k=nclusters)
    # dist2clasters = np.linalg.norm(np.expand_dims(centroids, 0) - np.expand_dims(coords, 1), axis=2)
    dist2clasters = scipy.spatial.distance_matrix(coords, centroids)
    dists2curr_cl = dist2clasters[range(coords.shape[0]), ind2cluster]
    # dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    benefits = dist2clasters - dists2curr_cl[:, np.newaxis]
    # print(sum(dists2curr_cl))
    swap_proposals = [[[] for _ in range(nclusters)] for _ in range(nclusters)]
    for p, (ben, curr_cl) in enumerate(zip(benefits, ind2cluster)):
      for ci, b in enumerate(ben):
        if b < 0:
          heappush(swap_proposals[curr_cl][ci], (b, p))

    visited = set()
    queue = []
    cycles = []

    def dfs(node):
      if node not in visited:
        visited.add(node)
        queue.append(node)
        for cl, edge in enumerate(swap_proposals[node]):
          if len(edge) > 0:
            dfs(cl)
        queue.pop()
      else:
        try:
          pos = queue.index(node)
          cycles.append(queue[pos:] + [node])
        except:
          pass

    for node in range(nclusters):
      dfs(node)

    for path in cycles:
      n_min = coords.shape[0]
      for cl_from, cl_to in zip(path[:-1], path[1:]):
        e = len(swap_proposals[cl_from][cl_to])
        n_min = min(n_min, e)

      for cl_from, cl_to in zip(path[:-1], path[1:]):
        for _ in range(n_min):
          _, p = heappop(swap_proposals[cl_from][cl_to])
          ind2cluster[p] = cl_to
          for i, prop in enumerate(swap_proposals[cl_from]):
            swap_proposals[cl_from][i] = [pr for pr in prop if pr[1] != p]

    clusters = fill_clasters(nclusters, ind2cluster)
    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)

    if np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

def lloid_equal_size_linear_assignment(coords, nclusters, show_plt=True):
  from sklearn.cluster import KMeans
  from scipy.spatial.distance import cdist
  from scipy.optimize import linear_sum_assignment
  cluster_size = int(np.ceil(len(coords)/nclusters))
  # nclusters = int(np.ceil(len(coords)/cluster_size))
  kmeans = KMeans(nclusters)
  kmeans.fit(coords)
  centroids = kmeans.cluster_centers_
  old_int2cluster = None
  while True:
    centroids = centroids.reshape(-1, 1, coords.shape[-1]).repeat(cluster_size, 1).reshape(-1, coords.shape[-1])
    distance_matrix = cdist(coords, centroids)
    ind2cluster = linear_sum_assignment(distance_matrix)[1]//cluster_size
    if show_plt:
      show_chart(coords, ind2cluster, kmeans.cluster_centers_, nclusters)
    # dist2clasters = np.linalg.norm(np.expand_dims(kmeans.cluster_centers_, 0) - np.expand_dims(coords, 1), axis=2)
    dist2clasters = scipy.spatial.distance_matrix(coords, kmeans.cluster_centers_)
    dists2curr_cl = dist2clasters[range(coords.shape[0]), ind2cluster]
    # dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    # print(sum(dists2curr_cl))
    clusters = fill_clasters(nclusters, ind2cluster)
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])
    if True:#np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

  return ind2cluster

def cluster_equal_size_mincostmaxflow(coords, nclusters, show_plt=True):
  clusters, ind2cluster = init_clasters(coords, nclusters)
  import networkx as nx
  old_int2cluster = None
  maxint = 2**16 - 1
  while True:
    centroids = np.array([sum(coords[l]) / len(l) for l in clusters])
    # kdt = KDTree(centroids)
    # dist2clasters, cl_inds = kdt.query(coords, k=nclusters)
    # dist2clasters = np.linalg.norm(np.expand_dims(centroids, 0) - np.expand_dims(coords, 1), axis=2)
    dist2clasters = scipy.spatial.distance_matrix(coords, centroids)
    nverts = coords.shape[0]
    dists2curr_cl = dist2clasters[range(nverts), ind2cluster]
    # dists2curr_cl = [linalg.norm(centroids[cl] - coord) for cl, coord in zip(ind2cluster, coords)]
    # benefits = dist2clasters - dists2curr_cl[:, np.newaxis]
    # print(sum(dists2curr_cl))
    dist2clasters = (dist2clasters / dist2clasters.max() * maxint).astype(int)
    G = nx.DiGraph()
    source = nverts + nclusters
    sink = source + 1
    for p, dists in enumerate(dist2clasters):
      G.add_edge(source, p, capacity=1, weight=0)
      for c, d in enumerate(dists):
        G.add_edge(p, c + nverts, capacity=1, weight=d)
    for c in range(nclusters):
      G.add_edge(
        c + nverts, sink, 
        capacity=claster_len(nverts / nclusters, c), 
        weight=0
      )

    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    # mincost = nx.cost_of_flow(G, mincostFlow)
    for p in range(nverts):
      edges = mincostFlow[p]
      cl = next((p for p, w in edges.items() if w == 1), None) - nverts
      ind2cluster[p] = cl
        
    clusters = fill_clasters(nclusters, ind2cluster)

    if show_plt:
      show_chart(coords, ind2cluster, centroids, nclusters)

    if np.array_equal(ind2cluster, old_int2cluster):
      return ind2cluster, sum(dists2curr_cl)
    else:
      old_int2cluster = ind2cluster

if __name__ == '__main__':
  results = []
  np.random.seed(0)
  from time import time
  start = time()
  for _ in range(10):
    # print('starting...')
    nclusters = 10
    dim = 1700
    real_centroids = np.random.rand(nclusters, dim)
    offsets = np.random.normal(scale=0.1, size=(1, 50, dim))
    coords = (real_centroids.reshape((nclusters, 1, dim)) + offsets).reshape((-1, dim))
    # coords = np.random.rand(100, 2)
    # masses = np.random.rand(100)

    result = cluster_equal_size(coords, nclusters, False)
    # result = cluster_equal_size_pair_split(coords, nclusters, False)
    # result = cluster_equal_size_swap(coords, nclusters, False)
    # result = cluster_equal_size_elki(coords, nclusters, False)
    # result = cluster_equal_size_detect_cycles(coords, nclusters, False)
    # result = lloid_equal_size_linear_assignment(coords, nclusters, False)
    # result = cluster_equal_size_mincostmaxflow(coords, nclusters, False)
    results += [result[1]]

  print(mean(results))
  print(time() - start)
