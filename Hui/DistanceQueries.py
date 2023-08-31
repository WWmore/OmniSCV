"""
http://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html
"""
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# Load mesh and convert to open3d.t.geometry.TriangleMesh
armadillo_data = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_data.path) ##open3d.t.geometry.TriangleMesh

# print(mesh.vertices, mesh.triangles)
o3d.visualization.draw_geometries([mesh])

mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

# Use a helper function to create rays for a pinhole camera.
rays = scene.create_rays_pinhole(fov_deg=60, center=[0.5,0.5,0.5], eye=[-1,-1,-1], up=[0,0,1],
                               width_px=320, height_px=240)

# Compute the ray intersections and visualize the hit distance (depth)
ans = scene.cast_rays(rays)
plt.imshow(ans['t_hit'].numpy())


query_point = o3d.core.Tensor([[10, 10, 10]], dtype=o3d.core.Dtype.Float32)

# Compute distance of the query point from the surface
unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)


print("unsigned distance", unsigned_distance.numpy())
print("signed_distance", signed_distance.numpy())
print("occupancy", occupancy.numpy())

min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

N = 256
query_points = np.random.uniform(low=min_bound, high=max_bound,
                                 size=[N, 3]).astype(np.float32)

# Compute the signed distance for N random points
signed_distance = scene.compute_signed_distance(query_points)

xyz_range = np.linspace(min_bound, max_bound, num=32)
#print(xyz_range.shape) ##shape=[32,3]

# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

# signed distance is a [32,32,32] array
signed_distance = scene.compute_signed_distance(query_points)
#print(signed_distance) ##shape=(32,32,32)

# We can visualize a slice of the distance field directly with matplotlib
plt.imshow(signed_distance.numpy()[:, :, 15])



from matplotlib import cm
# create the figure
fig = plt.figure()

# show the reference image
ax1 = fig.add_subplot(121)
data = signed_distance.numpy()[:, :, 15]
ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])

# show the 3D rotated projection
ax2 = fig.add_subplot(122, projection='3d')
cset = ax2.contourf(xyz_range[:,0], xyz_range[:,1], data, 100, zdir='z', offset=0.5, cmap=cm.BrBG)

ax2.set_zlim((0.,1.))

plt.colorbar(cset)
plt.show()