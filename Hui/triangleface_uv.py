"""
https://programtalk.com/python-more-examples/open3d.geometry.TriangleMesh/#google_vignette
"""

import cv2
import numpy as np
import open3d as o3d

vert=[[0,0,0],[0,1,0],[1,1,0],[1,0,0],
   [0,0,1],[0,1,1],[1,1,1],[1,0,1]]

faces=[[0, 1, 2], [0, 2, 3], [6, 5, 4],
 [7, 6, 4], [5, 1, 0], [0, 4, 5], [3, 2, 6],
 [6, 7, 3], [0, 3, 7], [0, 7, 4], [1, 5, 6],
 [1, 6, 2]]

m = o3d.geometry.TriangleMesh()
m.vertices = o3d.utility.Vector3dVector(vert)
m.triangles = o3d.utility.Vector3iVector(faces)

m.compute_vertex_normals()
o3d.visualization.draw_geometries([m])



if 0:
    DX,DY=0.5/2,0.66/2
    v_uv=[[DX,DY],[DX,2*DY],[2*DX,2*DY],[2*DX,DY],
        [0,DX],[DX,1],[3*DX,2*DY],[3*DX,DY]]

    v_uv=np.asarray(v_uv)
    v_uv=np.concatenate((v_uv,v_uv,v_uv),axis=0)
    m.triangle_uvs = o3d.utility.Vector2dVector(v_uv)

    text=cv2.imread('cupe_uv.png')
    # cv2.imshow('texture',text)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    m.textures=[o3d.geometry.Image(text)]

    o3d.visualization.draw_geometries([m])

else:
    uvs = np.random.rand(len(faces) * 3, 2) 
    m.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    m.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(faces)), dtype=np.int32))
    o3d.visualization.draw_geometries([m])


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
