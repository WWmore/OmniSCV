"""
https://programtalk.com/python-more-examples/open3d.geometry.TriangleMesh/#google_vignette
"""

import cv2
import numpy as np
import open3d as o3d

def load_mesh(model_fn, classification=True):
  # To load and clean up mesh - "remove vertices that share position"
  if classification:
    mesh_ = trimesh.load_mesh(model_fn, process=True)
    mesh_.remove_duplicate_faces()
  else:
    mesh_ = trimesh.load_mesh(model_fn, process=False)
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
  mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)

  return mesh

def create_mesh_with_uvmap(vertices, faces, texture_path=None, uvs=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if texture_path is not None and uvs is not None:
        if o3d_version==9:
            mesh.texture = o3d.io.read_image(texture_path)
            mesh.triangle_uvs = uvs
        elif o3d_version>=11:
            mesh.textures = [o3d.io.read_image(texture_path)]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(faces)), dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh

def export_mesh(name, v, f):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d.io.write_triangle_mesh(name, mesh)

def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
  # Create a mesh sphere
  spheres = o3d.geometry.TriangleMesh()
  s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
  s.compute_vertex_normals()

  for i, p in enumerate(pcd.points):
    si = copy.deepcopy(s)
    trans = np.identity(4)
    trans[:3, 3] = p
    si.transform(trans)
    si.paint_uniform_color(pcd.colors[i])
    spheres += si
  return spheres


def pointcloud_to_spheres(pcd, voxel_size, color, sphere_size=0.6):
  spheres = o3d.geometry.TriangleMesh()
  s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
  s.compute_vertex_normals()
  s.paint_uniform_color(color)
  if isinstance(pcd, o3d.geometry.PointCloud):
    pcd = np.array(pcd.points)
  for i, p in enumerate(pcd):
    si = copy.deepcopy(s)
    trans = np.identity(4)
    trans[:3, 3] = p
    si.transform(trans)
    # si.paint_uniform_color(pcd.colors[i])
    spheres += si
  return spheres


def to_o3d_mesh(self):
   """Convert to o3d mesh object for visualization
   """
   verts, faces, norms, colors = self.get_mesh()
   mesh = o3d.geometry.TriangleMesh()
   mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
   mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
   mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.)
   return mesh

def export_mesh(self, file_name, occupancy_function, iso_value):
   verts, faces = self.generate_mesh(occupancy_function, iso_value)
   mesh = o3d.geometry.TriangleMesh()
   mesh.vertices = o3d.utility.Vector3dVector(verts)
   mesh.triangles = o3d.utility.Vector3iVector(faces)
   o3d.io.write_triangle_mesh(file_name, mesh)

def batch_export_mesh(self, file_name_prefix, start_index, batch_size, occupancy_function, iso_value):
   batch_verts, batch_faces = self.batch_generate_mesh(batch_size, occupancy_function, iso_value)
   Path(file_name_prefix).mkdir(parents=True, exist_ok=True)
   for i in range(len(batch_verts)):
      mesh = o3d.geometry.TriangleMesh()
      mesh.vertices = o3d.utility.Vector3dVector(batch_verts[i])
      mesh.triangles = o3d.utility.Vector3iVector(batch_faces[i])
      o3d.io.write_triangle_mesh("%s/%d.ply" % (file_name_prefix, (start_index+i)), mesh)



def get_mesh(self, attribute='color'):
   """ Extract a mesh from the TSDF using marching cubes

   If TSDF also has atribute_vols, these are extracted as
   vertex_attributes. The mesh is also colored using the cmap

   Args:
      attribute: which tsdf attribute is used to color the mesh
      cmap: colormap for converting the attribute to a color

   Returns:
      open3d.geometry.TriangleMesh
   """

   tsdf_vol = self.tsdf_vol.detach().clone()

   tsdf_vol = tsdf_vol.clamp(-1, 1).cpu().numpy()

   if tsdf_vol.min() >= 0 or tsdf_vol.max()   <  = 0:
      return o3d.geometry.TriangleMesh()

   verts, faces, _, _ = measure.marching_cubes(tsdf_vol, level=0)

   verts_ind = np.round(verts).astype(int)
   # check if verts are at -1 to 1 crossing
   x, y, z = tsdf_vol.shape
   max_idx = [x-1, y-1, z-1]
   inds_000 = np.floor(verts).astype(np.int)
   inds_001 = np.clip(inds_000 + np.array([[0, 0, 1]]), [0, 0, 0], max_idx)
   inds_010 = np.clip(inds_000 + np.array([[0, 1, 0]]), [0, 0, 0], max_idx)
   inds_011 = np.clip(inds_000 + np.array([[0, 1, 1]]), [0, 0, 0], max_idx)
   inds_100 = np.clip(inds_000 + np.array([[1, 0, 0]]), [0, 0, 0], max_idx)
   inds_101 = np.clip(inds_000 + np.array([[1, 0, 1]]), [0, 0, 0], max_idx)
   inds_110 = np.clip(inds_000 + np.array([[1, 1, 0]]), [0, 0, 0], max_idx)
   inds_111 = np.clip(inds_000 + np.array([[1, 1, 1]]), [0, 0, 0], max_idx)
   val_000 = tsdf_vol[inds_000[:, 0], inds_000[:, 1], inds_000[:, 2]]
   val_001 = tsdf_vol[inds_001[:, 0], inds_001[:, 1], inds_001[:, 2]]
   val_010 = tsdf_vol[inds_010[:, 0], inds_010[:, 1], inds_010[:, 2]]
   val_011 = tsdf_vol[inds_011[:, 0], inds_011[:, 1], inds_011[:, 2]]
   val_100 = tsdf_vol[inds_100[:, 0], inds_100[:, 1], inds_100[:, 2]]
   val_101 = tsdf_vol[inds_101[:, 0], inds_101[:, 1], inds_101[:, 2]]
   val_110 = tsdf_vol[inds_110[:, 0], inds_110[:, 1], inds_110[:, 2]]
   val_111 = tsdf_vol[inds_111[:, 0], inds_111[:, 1], inds_111[:, 2]]

   bad_verts = ((val_000==1) |
               (val_001==1) |
               (val_010==1) |
               (val_011==1) |
               (val_100==1) |
               (val_101==1) |
               (val_110==1) |
               (val_111==1)) & ((val_000==-1) |
                                 (val_001==-1) |
                                 (val_010==-1) |
                                 (val_011==-1) |
                                 (val_100==-1) |
                                 (val_101==-1) |
                                 (val_110==-1) |
                                 (val_111==-1))

   bad_inds = np.arange(verts.shape[0])[bad_verts]
   verts = verts * self.voxel_size + self.origin.cpu().numpy()

   vertex_attributes = {}
   # get vertex attributes
   if 'instance' in self.attribute_vols:
      instance_vol = self.attribute_vols['instance']
      instance_vol = instance_vol.detach().cpu().numpy()
      instance = instance_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
      vertex_attributes['instance'] = instance

   # color mesh
   if attribute == 'color' and 'color' in self.attribute_vols:
      color_vol = self.attribute_vols['color']
      color_vol = color_vol.detach().clamp(0, 255).byte().cpu().numpy()
      colors = color_vol[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].T
   elif attribute == 'instance':
      label_viz = instance + 1
      n = label_viz.max()
      cmap = (colormap('jet')(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
      cmap = cmap[np.random.permutation(n), :]
      cmap = np.insert(cmap, 0, [0, 0, 0], 0)
      colors = cmap[label_viz, :]
   else:
      colors = None

   mesh = o3d.geometry.TriangleMesh()
   mesh.vertices = o3d.utility.Vector3dVector(verts)
   mesh.triangles = o3d.utility.Vector3iVector(faces)
   mesh.vertex_colors = o3d.utility.Vector3dVector(colors[..., [2, 1, 0]] / 255.)
   mesh.remove_vertices_by_index(bad_inds)

   return mesh

def save_mesh_pointclouds(self, inputs, epoch, center=None, scale=None):
   '''  Save meshes and point clouds.
   Args:
      inputs (torch.tensor)       : source point clouds
      epoch (int)                 : the number of iterations
      center (numpy.array)        : center of the shape
      scale (numpy.array)         : scale of the shape
   '''

   exp_pcl = self.cfg['train']['exp_pcl']
   exp_mesh = self.cfg['train']['exp_mesh']
   
   psr_grid, points, normals = self.pcl2psr(inputs)
   
   if exp_pcl:
      dir_pcl = self.cfg['train']['dir_pcl']
      p = points.squeeze(0).detach().cpu().numpy()
      p = p * 2 - 1
      n = normals.squeeze(0).detach().cpu().numpy()
      if scale is not None:
            p *= scale
      if center is not None:
            p += center
      export_pointcloud(os.path.join(dir_pcl, '{:04d}.ply'.format(epoch)), p, n)
   if exp_mesh:
      dir_mesh = self.cfg['train']['dir_mesh']
      with torch.no_grad():
            v, f, _ = mc_from_psr(psr_grid,
                  zero_level=self.cfg['data']['zero_level'], real_scale=True)
            v = v * 2 - 1
            if scale is not None:
               v *= scale
            if center is not None:
               v += center
      mesh = o3d.geometry.TriangleMesh()
      mesh.vertices = o3d.utility.Vector3dVector(v)
      mesh.triangles = o3d.utility.Vector3iVector(f)
      outdir_mesh = os.path.join(dir_mesh, '{:04d}.ply'.format(epoch))
      o3d.io.write_triangle_mesh(outdir_mesh, mesh)

   if self.cfg['train']['vis_psr']:
      dir_psr_vis = self.cfg['train']['out_dir']+'/psr_vis_all'
      visualize_psr_grid(psr_grid, out_dir=dir_psr_vis)



def visualize_points_mesh(vis, points, normals, verts, faces, cfg, it, epoch, color_v=None):
    ''' Visualization.

    Args:
        data (dict): data dictionary
        depth (int): PSR depth
        out_path (str): output path for the mesh 
    '''
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(np.array([0.7,0.7,0.7]))
    if color_v is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color_v)
    
    if vis is not None:
        dir_o3d = cfg['train']['dir_o3d']
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        p = points.squeeze(0).detach().cpu().numpy()
        n = normals.squeeze(0).detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.normals = o3d.utility.Vector3dVector(n)
        pcd.paint_uniform_color(np.array([0.7,0.7,1.0]))
        # pcd = pcd.uniform_down_sample(5)

        vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)
        
        #! Thingi wheel - an example for how to change cameras in Open3D viewers
        vis.get_view_control().set_front([ 0.0461, -0.7467, 0.6635 ])
        vis.get_view_control().set_lookat([ 0.0092, 0.0078, 0.0638 ])
        vis.get_view_control().set_up([ 0.0520, 0.6651, 0.7449 ])
        vis.get_view_control().set_zoom(0.7)
        vis.poll_events()
        
        out_path = os.path.join(dir_o3d, '{}.jpg'.format(it))
        vis.capture_screen_image(out_path)

        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.update_geometry(pcd)
        vis.get_render_option().point_show_normal=True # visualize point normals

        vis.get_view_control().set_front([ 0.0461, -0.7467, 0.6635 ])
        vis.get_view_control().set_lookat([ 0.0092, 0.0078, 0.0638 ])
        vis.get_view_control().set_up([ 0.0520, 0.6651, 0.7449 ])
        vis.get_view_control().set_zoom(0.7)
        vis.poll_events()

        out_path = os.path.join(dir_o3d, '{}_pcd.jpg'.format(it))
        vis.capture_screen_image(out_path)


def visualize_plane(annos, args, eps=0.9):
    """visualize plane
    """
    colormap = np.array(colormap_255) / 255
    junctions = [item['coordinate'] for item in annos['junctions']]

    if args.color == 'manhattan':
        manhattan = dict()
        for planes in annos['manhattan']:
            for planeID in planes['planeID']:
                manhattan[planeID] = planes['ID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    _, vertices_holes = np.where(np.array(annos['lineJunctionMatrix'])[lines_holes])
    vertices_holes = np.unique(vertices_holes)

    # load polygons
    polygons = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]
            lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
            junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
            polygon = convert_lines_to_vertices(junction_pairs)
            vertices, faces = clip_polygon(polygon, vertices_holes, junctions, plane_anno)
            polygons.append([vertices, faces, planeID, plane_anno['normal'], plane_anno['type'], semantic['type']])

    plane_set = []
    for i, (vertices, faces, planeID, normal, plane_type, semantic_type) in enumerate(polygons):
        # ignore the room ceiling
        if plane_type == 'ceiling' and semantic_type not in ['door', 'window']:
            continue

        plane_vis = open3d.geometry.TriangleMesh()

        plane_vis.vertices = open3d.utility.Vector3dVector(vertices)
        plane_vis.triangles = open3d.utility.Vector3iVector(faces)

        if args.color == 'normal':
            if np.dot(normal, [1, 0, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[0])
            elif np.dot(normal, [-1, 0, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[1])
            elif np.dot(normal, [0, 1, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[2])
            elif np.dot(normal, [0, -1, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[3])
            elif np.dot(normal, [0, 0, 1]) > eps:
                plane_vis.paint_uniform_color(colormap[4])
            elif np.dot(normal, [0, 0, -1]) > eps:
                plane_vis.paint_uniform_color(colormap[5])
            else:
                plane_vis.paint_uniform_color(colormap[6])
        elif args.color == 'manhattan':
            # paint each plane with manhattan world
            if planeID not in manhattan.keys():
                plane_vis.paint_uniform_color(colormap[6])
            else:
                plane_vis.paint_uniform_color(colormap[manhattan[planeID]])

        plane_set.append(plane_vis)

    draw_geometries_with_back_face(plane_set)

def extract_mash(self, occupancy_grid: torch.Tensor, cluster_offset: Optional[int] = 12 ** 3,
            smooth: Optional[bool] = True, latent_vector: Optional[torch.Tensor] = None,
            refine: Optional[bool] = False, simplify: Optional[bool] = True,
            refinement_steps: Optional[int] = 200,
            patches: Optional[torch.Tensor] = None,
            large_patches: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
   """
   Method extracts a mesh from a given occupancy grid.
   :param occupancy_grid: (torch.Tensor) Occupancy grid
   :param cluster_offset: (Optional[int]) Offset of cluster size to be plotted smaller clusters will be ignored
   :param smooth: (Optional[bool]) If true binary occupancy grid gets smoothed before marching cubes algorithm
   :param latent_vector: (Optional[torch.Tensor]) Latent vector of model needed when refinement should be performed
   :param refine: (Optional[bool]) If true produced mesh will be refined by the gradients of the network
   :param simplify: (Optional[bool]) If true mesh gets simplified
   :param refinement_steps: (Optional[int]) Number of optimization steps to be utilized in refinement
   :param patches: (Optional[torch.Tensor]) Patches of the input volume
   """
   # To numpy
   occupancy_grid = occupancy_grid.cpu().numpy()
   # Cluster occupancy grid
   occupancy_grid = label(occupancy_grid)[0]
   # Get sizes of clusters
   cluster_indexes, cluster_sizes = np.unique(occupancy_grid, return_counts=True)
   # Iterate over clusters to eliminate clusters smaller than the offset
   for cluster_index, cluster_size in zip(cluster_indexes[1:], cluster_sizes[1:]):
      if (cluster_size   <   cluster_offset):
            occupancy_grid = np.where(occupancy_grid == cluster_index, 0, occupancy_grid)
   # Produce binary grid
   occupancy_grid = (occupancy_grid > 0.0).astype(float)
   # Remove batch dim if needed
   if occupancy_grid.ndim != 3:
      occupancy_grid = occupancy_grid.reshape(occupancy_grid.shape[-3:])
   # Apply distance transformation
   if smooth:
      occupancy_grid = mcubes.smooth(occupancy_grid)
   # Perform marching cubes
   vertices, triangles = mcubes.marching_cubes(occupancy_grid, 0. if smooth else 0.5)
   # Perform simplification if utilized
   if simplify:
      # Make open 3d mesh
      mesh = o3d.geometry.TriangleMesh()
      mesh.vertices = o3d.utility.Vector3dVector(vertices)
      mesh.triangles = o3d.utility.Vector3iVector(triangles)
      # Simplify mesh
      mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
      # Get vertices and triangles
      vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
   # Perform refinement if utilized
   if refine:
      vertices = torch.from_numpy(vertices)
      triangles = torch.from_numpy(triangles.astype(np.int64))
      # Init parameters
      vertices_parameter = nn.Parameter(vertices.clone().to(self.device), requires_grad=True)
      # Triangles to device
      triangles = triangles.long().to(self.device)
      # Init optimizer
      optimizer = torch.optim.RMSprop([vertices_parameter], lr=1e-03)
      # Optimization loop
      for _ in range(refinement_steps):
            # Rest gradients
            optimizer.zero_grad()
            # Get triangles vertices
            triangles_vertices = vertices_parameter[triangles]
            # Generate samples from dirichlet distribution
            samples = np.random.dirichlet((0.5, 0.5, 0.5), size=triangles_vertices.shape[0])
            samples = torch.from_numpy(samples).float().to(self.device)
            triangles_samples = (triangles_vertices.float() * samples.unsqueeze(dim=-1)).sum(dim=1)
            # Get different triangles vertices
            triangles_vertices_1 = triangles_vertices[:, 1] - triangles_vertices[:, 0]
            triangles_vertices_2 = triangles_vertices[:, 2] - triangles_vertices[:, 1]
            # Normalize triangles vertices
            triangles_vertices_normalized = torch.cross(triangles_vertices_1, triangles_vertices_2)
            triangles_vertices_normalized = triangles_vertices_normalized / \
                                          (triangles_vertices_normalized.norm(dim=1, keepdim=True) + 1e-10)
            if patches is not None:
               # Get input patches
               input_patches = patches[:, :,
                              triangles_samples[:, 0].long(),
                              triangles_samples[:, 1].long(),
                              triangles_samples[:, 2].long()].transpose(1, 2).float()
               # Get large patches if utilized
               if large_patches is not None:
                  large_input_patches = large_patches[:, :,
                                          triangles_samples[:, 0].long(),
                                          triangles_samples[:, 1].long(),
                                          triangles_samples[:, 2].long()].transpose(1, 2).float()
                  # Downscale patches
                  large_input_patches = F.max_pool3d(large_input_patches[0], kernel_size=(2, 2, 2),
                                                      stride=(2, 2, 2))[None]
                  # Concat small and large patches
                  input_patches = torch.cat([input_patches, large_input_patches], dim=2)
               # Data to device
               input_patches = input_patches.float().to(self.device)
            else:
               input_patches = None
            # Predict occupancy values
            triangles_vertices_predictions, _ = self.model(None, triangles_samples.unsqueeze(dim=0).float(),
                                                         input_patches, inference=True,
                                                         latent_vectors=latent_vector)
            # Calc targets
            targets = \
               - autograd.grad([triangles_vertices_predictions.sum()], [triangles_samples], create_graph=True)[0]
            # Normalize targets
            targets_normalized = targets / (targets.norm(dim=1, keepdim=True) + 1e-10)
            # Calc target loss
            targets_loss = ((triangles_vertices_predictions - 0.5) ** 2).mean()
            # Calc normalization loss
            normalization_loss = ((triangles_vertices_normalized - targets_normalized) ** 2).sum(dim=1).mean()
            # Calc final loss
            loss = targets_loss + 0.01 * normalization_loss
            # Calc gradients
            loss.backward()
            # Perform optimization
            optimizer.step()
      return vertices_parameter.data.detach().cpu(), triangles.detach().cpu()
   return torch.from_numpy(vertices), torch.from_numpy(triangles.astype(np.int64))

def compute_metrics(
    scene_path: str,
    voxel_size: float,
    metrics: List[str] = VALID_METRICS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Computes the 3D scene metrics for a given file.

    Args:
        scene_path: path to the scene file (glb / ply)
        voxel_size: specifies the voxel size for scene simplification
        metrics: list of metrics to compute

    Outputs:
        metric_values: a dict mapping from the required metric names to values
    """
    # sanity check
    for metric in metrics:
        assert metric in VALID_METRICS
    # load scene in habitat_simulator and trimesh
    hsim = robust_load_sim(scene_path)
    trimesh_scene = trimesh.load(scene_path)
    # Simplify scene-mesh for faster metric computation
    # Does not impact the final metrics much
    o3d_scene = o3d.geometry.TriangleMesh()
    vertices = np.array(trimesh_scene.triangles).reshape(-1, 3)
    faces = np.arange(0, len(vertices)).reshape(-1, 3)
    o3d_scene.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_scene.triangles = o3d.utility.Vector3iVector(faces)
    o3d_scene = o3d_scene.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    if verbose:
        print(
            f"=====> Downsampled mesh from {len(trimesh_scene.triangles)} "
            f"to {len(o3d_scene.triangles)}"
        )
    trimesh_scene = trimesh.Trimesh()
    trimesh_scene.vertices = np.array(o3d_scene.vertices)
    trimesh_scene.faces = np.array(o3d_scene.triangles)

    metric_values = {}
    for metric in metrics:
        metric_values[metric] = METRIC_TO_FN_MAP[metric](hsim, trimesh_scene)
    metric_values["scene"] = scene_path
    hsim.close()
    return metric_values


def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))

def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

    model = smplx.build_layer(
        model_folder, model_type=model_type,
        gender=gender, use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext)
    print(model)

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))


def main(model_folder, corr_fname, ext='npz',
         head_color=(0.3, 0.3, 0.6),
         gender='neutral'):

    head_idxs = np.load(corr_fname)

    model = smplx.create(model_folder, model_type='smplx',
                         gender=gender,
                         ext=ext)
    betas = torch.zeros([1, 10], dtype=torch.float32)
    expression = torch.zeros([1, 10], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()

    colors = np.ones_like(vertices) * [0.3, 0.3, 0.3]
    colors[head_idxs] = head_color

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])

def optimize_visulize():
    # read scene mesh, scene sdf
    scene, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(args.dataset_path,
                                                                                            args.dataset,
                                                                                            args.scene_name)
    smplx_model = smplx.create(args.smplx_model_path, model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=1
                               ).to(device)
    print('[INFO] smplx model loaded.')

    vposer_model, _ = load_vposer(args.vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')


    ##################### load optimization results ##################
    shift_list = np.load('{}/{}/shift_list.npy'.format(args.optimize_result_dir, args.scene_name))
    rot_angle_list_1 = np.load('{}/{}/rot_angle_list_1.npy'.format(args.optimize_result_dir, args.scene_name))

    if args.optimize:
        body_params_opt_list_s1 = np.load('{}/{}/body_params_opt_list_s1.npy'.format(args.optimize_result_dir, args.scene_name))
        body_params_opt_list_s2 = np.load('{}/{}/body_params_opt_list_s2.npy'.format(args.optimize_result_dir, args.scene_name))
    body_verts_sample_list = np.load('{}/{}/body_verts_sample_list.npy'.format(args.optimize_result_dir, args.scene_name))
    n_sample = len(body_verts_sample_list)


    ########################## evaluation (contact/collision score) #########################
    loss_non_collision_sample, loss_contact_sample = 0, 0
    loss_non_collision_opt_s1, loss_contact_opt_s1 = 0, 0
    loss_non_collision_opt_s2, loss_contact_opt_s2 = 0, 0
    body_params_prox_list_s1, body_params_prox_list_s2 = [], []
    body_verts_opt_prox_s2_list = []

    for cnt in tqdm(range(0, n_sample)):
        body_verts_sample = body_verts_sample_list[cnt]  # [10475, 3]

        # smplx params --> body mesh
        body_params_opt_s1 = torch.from_numpy(body_params_opt_list_s1[cnt]).float().unsqueeze(0).to(device)  # [1,75]
        body_params_opt_s1 = convert_to_3D_rot(body_params_opt_s1)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1,-1)  # [1, 63]
        body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]
        body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()

        body_params_opt_s2 = torch.from_numpy(body_params_opt_list_s2[cnt]).float().unsqueeze(0).to(device)
        body_params_opt_s2 = convert_to_3D_rot(body_params_opt_s2)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_opt_s2[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s2 = gen_body_mesh(body_params_opt_s2, body_pose_joint, smplx_model)[0]
        body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()

        ####################### transfrom local body verts to prox coodinate system ####################
        # generated body verts from cvae, before optimization
        body_verts_sample_prox = np.zeros(body_verts_sample.shape)   # [10475, 3]
        temp = body_verts_sample - shift_list[cnt]
        body_verts_sample_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_sample_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_sample_prox[:, 2] = temp[:, 2]

        ######### optimized body verts
        trans_matrix_1 = np.array([[math.cos(-rot_angle_list_1[cnt]), -math.sin(-rot_angle_list_1[cnt]), 0, 0],
                                   [math.sin(-rot_angle_list_1[cnt]), math.cos(-rot_angle_list_1[cnt]), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        trans_matrix_2 = np.array([[1, 0, 0, -shift_list[cnt][0]],
                                   [0, 1, 0, -shift_list[cnt][1]],
                                   [0, 0, 1, -shift_list[cnt][2]],
                                   [0, 0, 0, 1]])
        ### stage 1: simple optimization results
        body_verts_opt_prox_s1 = np.zeros(body_verts_opt_s1.shape)  # [10475, 3]
        temp = body_verts_opt_s1 - shift_list[cnt]
        body_verts_opt_prox_s1[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s1[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s1[:, 2] = temp[:, 2]
        # transfrom local params to prox coordinate system
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_opt_s1[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s1 = update_globalRT_for_smplx(body_params_prox_s1, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s1.append(body_params_prox_s1)

        ### stage 2: advanced optimiation results
        body_verts_opt_prox_s2 = np.zeros(body_verts_opt_s2.shape)  # [10475, 3]
        temp = body_verts_opt_s2 - shift_list[cnt]
        body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angle_list_1[cnt]) - \
                                       temp[:, 1] * math.sin(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angle_list_1[cnt]) + \
                                       temp[:, 1] * math.cos(-rot_angle_list_1[cnt])
        body_verts_opt_prox_s2[:, 2] = temp[:, 2]
        # transfrom local params to prox coordinate system
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_opt_s2[0].cpu().numpy(), smplx_model, trans_matrix_2)  # [72]
        body_params_prox_s2 = update_globalRT_for_smplx(body_params_prox_s2, smplx_model, trans_matrix_1)  # [72]
        body_params_prox_list_s2.append(body_params_prox_s2)
        body_verts_opt_prox_s2_list.append(body_verts_opt_prox_s2)

        ########################### visualization ##########################
        if args.visualize:
            body_mesh_sample = o3d.geometry.TriangleMesh()
            body_mesh_sample.vertices = o3d.utility.Vector3dVector(body_verts_sample_prox)
            body_mesh_sample.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_sample.compute_vertex_normals()

            body_mesh_opt_s1 = o3d.geometry.TriangleMesh()
            body_mesh_opt_s1.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s1)
            body_mesh_opt_s1.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_opt_s1.compute_vertex_normals()

            body_mesh_opt_s2 = o3d.geometry.TriangleMesh()
            body_mesh_opt_s2.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s2)
            body_mesh_opt_s2.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_opt_s2.compute_vertex_normals()

            o3d.visualization.draw_geometries([scene, body_mesh_sample])  # generated body mesh by cvae
            o3d.visualization.draw_geometries([scene, body_mesh_opt_s1])  # simple-optimized body mesh
            o3d.visualization.draw_geometries([scene, body_mesh_opt_s2])  # adv-optimizaed body mesh


        #####################  compute non-collision/contact score ##############
        # body verts before optimization
        body_verts_sample_prox_tensor = torch.from_numpy(body_verts_sample_prox).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_sample_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item()   <   1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_sample += 1.0
            loss_contact_sample += 0.0
        else:
            loss_non_collision_sample += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_sample += 1.0

        # stage 1: simple optimization results
        body_verts_opt_prox_tensor = torch.from_numpy(body_verts_opt_prox_s1).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_opt_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item()  <  1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_opt_s1 += 1.0
            loss_contact_opt_s1 += 0.0
        else:
            loss_non_collision_opt_s1 += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_opt_s1 += 1.0

        # stage 2: advanced optimization results
        body_verts_opt_prox_tensor = torch.from_numpy(body_verts_opt_prox_s2).float().unsqueeze(0).to(device)  # [1, 10475, 3]
        norm_verts_batch = (body_verts_opt_prox_tensor - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                       padding_mode='border')
        if body_sdf_batch.lt(0).sum().item()  <  1:  # if no interpenetration: negative sdf entries is less than one
            loss_non_collision_opt_s2 += 1.0
            loss_contact_opt_s2 += 0.0
        else:
            loss_non_collision_opt_s2 += (body_sdf_batch > 0).sum().float().item() / 10475.0
            loss_contact_opt_s2 += 1.0


    print('scene:', args.scene_name)

    loss_non_collision_sample = loss_non_collision_sample / n_sample
    loss_contact_sample = loss_contact_sample / n_sample
    print('w/o optimization body: non_collision score:', loss_non_collision_sample)
    print('w/o optimization body: contact score:', loss_contact_sample)

    loss_non_collision_opt_s1 = loss_non_collision_opt_s1 / n_sample
    loss_contact_opt_s1 = loss_contact_opt_s1 / n_sample
    print('optimized body s1: non_collision score:', loss_non_collision_opt_s1)
    print('optimized body s1: contact score:', loss_contact_opt_s1)

    loss_non_collision_opt_s2 = loss_non_collision_opt_s2 / n_sample
    loss_contact_opt_s2 = loss_contact_opt_s2 / n_sample
    print('optimized body s2: non_collision score:', loss_non_collision_opt_s2)
    print('optimized body s2: contact score:', loss_contact_opt_s2)


