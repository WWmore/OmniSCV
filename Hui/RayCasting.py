"""
http://www.open3d.org/blog/
http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
"""
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# Create scene and add a cube
cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(cube)

# Use a helper function to create rays for a pinhole camera.
rays = scene.create_rays_pinhole(fov_deg=60, center=[0.5,0.5,0.5], eye=[-1,-1,-1], up=[0,0,1],
                               width_px=320, height_px=240)

# Compute the ray intersections and visualize the hit distance (depth)
ans = scene.cast_rays(rays)
plt.imshow(ans['t_hit'].numpy())

#plt.imshow(np.abs(ans['primitive_normals'].numpy()))

#plt.imshow(ans['geometry_ids'].numpy(), vmax=3)

## Creating a virtual point cloud
hit = ans['t_hit'].isfinite()
points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
pcd = o3d.t.geometry.PointCloud(points)
# Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
o3d.visualization.draw_geometries([pcd.to_legacy()],
                                  front=[0.5, 0.86, 0.125],
                                  lookat=[0.23, 0.5, 2],
                                  up=[-0.63, 0.45, -0.63],
                                  zoom=0.7)
# o3d.visualization.draw([pcd]) # new API






###  Below has bug
# import open3d.core as o3c

# mesh = o3d.t.geometry.TriangleMesh()
# mesh.vertex["positions"] = o3c.Tensor([[0.0, 0.0, 1.0],
#                                      [0.0, 1.0, 0.0],
#                                      [1.0, 0.0, 0.0],
#                                      [1.0, 1.0, 1.0]], dtype=o3c.float32)
# mesh.vertex["my_custom_labels"] = o3c.Tensor([0, 1, 2, 4], dtype=o3c.int32)
# mesh.triangle["indices"] = o3c.Tensor([[0, 1, 2], 
#                                      [1, 2, 3]], dtype=o3c.int32)


# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])