import kaolin as kal
import numpy as np
import torch
import torch.nn.functional as F
import math
import os

from .utils import grid_sample_bilinear, circpad

from packaging import version

class MyMeshTemplate():
    
    def __init__(self, mesh_path, is_symmetric=True):
        self.mesh = kal.rep.TriangleMesh.from_obj(mesh_path, enable_adjacency=True)
        self.mesh.cuda()
        
    
    def forward_renderer(self, vertex_positions, num_gpus=1, **kwargs):
        mesh_faces = self.mesh.faces
        mesh_face_textures = self.mesh.face_textures
        if num_gpus > 1:
            mesh_faces = mesh_faces.repeat(num_gpus, 1)
            mesh_face_textures = mesh_face_textures.repeat(num_gpus, 1)

        # input_uvs, input_texture = self.adjust_uv_and_texture(texture)

        # image, alpha, _ = renderer(points=[vertex_positions, mesh_faces],
        #                            uv_bxpx2=input_uvs,
        #                            texture_bx3xthxtw=input_texture,
        #                            ft_fx3=mesh_face_textures,
        #                            **kwargs)
        return vertex_positions,mesh_faces,mesh_face_textures
        
        
class MeshTemplate:
    
    def __init__(self, mesh_path, is_symmetric=True):
        
        MeshTemplate._monkey_patch_dependencies()
        
        mesh = kal.rep.TriangleMesh.from_obj(mesh_path, enable_adjacency=True)
        mesh.cuda()
        
        print('---- Mesh definition ----')
        print(f'Vertices: {mesh.vertices.shape}')
        print(f'Indices: {mesh.faces.shape}')
        print(f'UV coords: {mesh.uvs.shape}')
        print(f'UV indices: {mesh.face_textures.shape}')
        # 取出y方向(第二个方向)上的最大值和最小值位置,作为两极
        poles = [mesh.vertices[:, 1].argmax().item(), mesh.vertices[:, 1].argmin().item()] # North pole, south pole

        # Compute reflection information (for mesh symmetry)
        # 对于x轴向
        axis = 0
        if version.parse(torch.__version__) < version.parse('1.2'):
            neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4)[:, 0].cpu().numpy()
            zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4)[:, 0].cpu().numpy()
        else:
            # 取出x轴小于0的所有的索引
            neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4, as_tuple=False)[:, 0].cpu().numpy()
            # 取出x轴=0的所有的索引
            zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4, as_tuple=False)[:, 0].cpu().numpy()
            
        pos_indices = []
        # 对于所有的x轴小于0的索引
        for idx in neg_indices:
            # 取出其坐标
            opposite_vtx = mesh.vertices[idx].clone()
            # x轴取反对称到正半轴
            opposite_vtx[axis] *= -1
            # 计算所有的点和x轴小于0关于yoz平面对称点的距离
            dists = (mesh.vertices - opposite_vtx).norm(dim=-1)
            # 最小距离和最小索引
            minval, minidx = torch.min(dists, dim=0)
            # 断言最小距离=0 (任何点关于yoz平面对称不会与另一个点重合)
            assert minval < 1e-4, minval
            # 记录每个小于0的点的对侧点索引
            pos_indices.append(minidx.item())
        # 断言对侧点数目=x轴负点数目
        assert len(pos_indices) == len(neg_indices)
        # 断言对侧点数目不重复
        assert len(pos_indices) == len(set(pos_indices)) # No duplicates
        pos_indices = np.array(pos_indices)

        pos_indices = torch.LongTensor(pos_indices).cuda()
        neg_indices = torch.LongTensor(neg_indices).cuda()
        zero_indices = torch.LongTensor(zero_indices).cuda()
        nonneg_indices = torch.LongTensor(list(pos_indices) + list(zero_indices)).cuda()

        # 所有数目=对侧点数目+负点数目+零点数目
        total_count = len(pos_indices) + len(neg_indices) + len(zero_indices)
        assert total_count == len(mesh.vertices), (total_count, len(mesh.vertices))

        index_list = {}
        segments = 32
        rings = 31 if '31rings' in mesh_path else 16
        print(f'The mesh has {rings} rings')
        print('-------------------------')
        # mesh.face_textures : 三角面片uv值
        # mesh.face : 三角面片顶点序号
        # 对于每一个三角面片
        for faces, vertices in zip(mesh.face_textures, mesh.faces):
            # 对于三角面片中的每一个顶点的uv值和顶点
            for face, vertex in zip(faces, vertices):
                # 如果当前顶点没有被访问过，那么对该顶点序号创建空列表
                if vertex.item() not in index_list:
                    index_list[vertex.item()] = []
                # 该三角面片的u值乘段数，v值乘环数（一共32个段，31个环）
                res = mesh.uvs[face].cpu().numpy() * [segments, rings]
                # 判断浮点数是否相等，u值乘段数是否等于段数 -》 u值为1？
                if math.isclose(res[0], segments, abs_tol=1e-4):
                    # u值如果为1，那么直接变为0，1的话超过边界了
                    res[0] = 0 # Wrap around
                # 该顶点的u值乘段数，v值乘环数，被append进该顶点的index_list
                index_list[vertex.item()].append(res)

        topo_map = torch.zeros(mesh.vertices.shape[0], 2)
        # index_list 长度 962的哈希表，每个位置存储的是一个长度为6的列表(每个顶点有且仅有6个面共有)，每个列表储存的都是该三角形的u值乘段数，v值乘环数
        for idx, data in index_list.items():
            # 对六个面的u值乘段数，v值乘环数求平均后分别再除段数和环数
            avg = np.mean(np.array(data, dtype=np.float32), axis=0) / [segments, rings]
            # 记录第idx个顶点的avg
            topo_map[idx] = torch.Tensor(avg)
        # 看论文图中可知，圆的拓扑构成为经纬线（即代码中的段和环），段和环的索引就是uv值（空间坐标和uv值一一对应），段和环的索引也可以直接被当成法向
        # 如果手动求法向，是不是就不需要了段和环的约束了？？？当然还要考虑固定uv情况下的分布尽可能均匀
        # Flip topo map
        topo_map = topo_map * 2 - 1
        topo_map = topo_map * torch.FloatTensor([1, -1]).to(topo_map.device)
        topo_map = topo_map.cuda()
        nonneg_topo_map = topo_map[nonneg_indices]

        # Force x = 0 for zero-indices if symmetry is enabled
        symmetry_mask = torch.ones_like(mesh.vertices).unsqueeze(0)
        symmetry_mask[:, zero_indices, 0] = 0

        # Compute mesh tangent map (per-vertex normals, tangents, and bitangents)
        mesh_normals = F.normalize(mesh.vertices, dim=1)
        # y方向单位向量
        up_vector = torch.Tensor([[0, 1, 0]]).to(mesh_normals.device).expand_as(mesh_normals)
        # 垂直y轴的（纬线切向）
        mesh_tangents = F.normalize(torch.cross(mesh_normals, up_vector, dim=1), dim=1)
        # 经线切向
        mesh_bitangents = torch.cross(mesh_normals, mesh_tangents, dim=1)
        # North pole and south pole have no (bi)tangent
        mesh_tangents[poles[0]] = 0
        mesh_bitangents[poles[0]] = 0
        mesh_tangents[poles[1]] = 0
        mesh_bitangents[poles[1]] = 0
        # 切向图为 法向，切向，垂直切向三者堆叠（恰好构成一个笛卡尔系）
        tangent_map = torch.stack((mesh_normals, mesh_tangents, mesh_bitangents), dim=1).cuda()
        nonneg_tangent_map = tangent_map[nonneg_indices] # For symmetric meshes
        
        self.mesh = mesh
        self.topo_map = topo_map
        self.nonneg_topo_map = nonneg_topo_map
        self.nonneg_indices = nonneg_indices
        self.neg_indices = neg_indices
        self.pos_indices = pos_indices
        self.symmetry_mask = symmetry_mask
        self.tangent_map = tangent_map
        self.nonneg_tangent_map = nonneg_tangent_map
        self.is_symmetric = is_symmetric
        
    def deform(self, deltas):
        """
        Deform this mesh template along its tangent map, using the provided vertex displacements.
        """
        # tangent_map : precomputed rotation matrix
        tgm = self.nonneg_tangent_map if self.is_symmetric else self.tangent_map
        # R@delta
        # tgm中每个点的三列分别表示xyz坐标轴(表示旋转),即生成的位移量是在每个顶点的切平面坐标系中运动,而不是在全局坐标系中
        # 右乘,是相对于自身运动,将运动量变换到自己的切平面坐标系中(长度不变,只变方向)
        return (deltas.unsqueeze(-2) @ tgm.expand(deltas.shape[0], -1, -1, -1)).squeeze(-2)

    def compute_normals(self, vertex_positions):
        """
        Compute face normals from the *final* vertex positions (not deltas).
        """
        a = vertex_positions[:, self.mesh.faces[:, 0]]
        b = vertex_positions[:, self.mesh.faces[:, 1]]
        c = vertex_positions[:, self.mesh.faces[:, 2]]
        v1 = b - a
        v2 = c - a
        normal = torch.cross(v1, v2, dim=2)
        return F.normalize(normal, dim=2)

    def get_vertex_positions(self, displacement_map):
        """
        Deform this mesh template using the provided UV displacement map.
        Output: 3D vertex positions in object space.
        """
        topo = self.nonneg_topo_map if self.is_symmetric else self.topo_map
        # displacement_map : bs*3*W*H (W=H)
        _, displacement_map_padded = self.adjust_uv_and_texture(displacement_map)
        if self.is_symmetric:
            # Compensate for even symmetry in UV map
            delta = 1/(2*displacement_map.shape[3])
            expansion = (displacement_map.shape[3]+1)/displacement_map.shape[3]
            topo = topo.clone()
            topo[:, 0] = (topo[:, 0] + 1 + 2*delta - expansion)/expansion # Only for x axis
        topo_expanded = topo.unsqueeze(0).unsqueeze(-2).expand(displacement_map.shape[0], -1, -1, -1)
        # 在displacement map上采样得到离散坐标（顶点沿其切向的运动量）
        # topo_expanded : bs*962*1*2(962个顶点，每个顶点两个参数uv值)
        vertex_deltas_local = grid_sample_bilinear(displacement_map_padded, topo_expanded).squeeze(-1).permute(0, 2, 1)
        # vertex_deltas_local：4*962*3 bs*点数*3(在局部的坐标系内的位移量)
        vertex_deltas = self.deform(vertex_deltas_local)
        if self.is_symmetric:
            # Symmetrize
            vtx_n = torch.Tensor(vertex_deltas.shape[0], self.topo_map.shape[0], 3).to(vertex_deltas.device)
            vtx_n[:, self.nonneg_indices] = vertex_deltas
            vtx_n2 = vtx_n.clone()
            vtx_n2[:, self.neg_indices] = vtx_n[:, self.pos_indices] * torch.Tensor([-1, 1, 1]).to(vtx_n.device)
            vertex_deltas = vtx_n2 * self.symmetry_mask
        # v' = v+R@delta
        vertex_positions = self.mesh.vertices.unsqueeze(0) + vertex_deltas
        return vertex_positions

    def adjust_uv_and_texture(self, texture, return_texture=True):
        """
        Returns the UV coordinates of this mesh template,
        and preprocesses the provided texture to account for boundary conditions.
        If the mesh is symmetric, the texture and UVs are adjusted accordingly.
        """
        
        if self.is_symmetric:
            delta = 1/(2*texture.shape[3])
            expansion = (texture.shape[3]+1)/texture.shape[3]
            uvs = self.mesh.uvs.clone()
            uvs[:, 0] = (uvs[:, 0] + delta)/expansion
            
            uvs = uvs.expand(texture.shape[0], -1, -1)
            texture = circpad(texture, 1) # Circular padding
        else:
            uvs = self.mesh.uvs.expand(texture.shape[0], -1, -1)
            texture = torch.cat((texture, texture[:, :, :, :1]), dim=3)
            
        return uvs, texture
    def forward_renderer(self, vertex_positions, texture, num_gpus=1, **kwargs):
        mesh_faces = self.mesh.faces
        mesh_face_textures = self.mesh.face_textures
        if num_gpus > 1:
            mesh_faces = mesh_faces.repeat(num_gpus, 1)
            mesh_face_textures = mesh_face_textures.repeat(num_gpus, 1)

        input_uvs, input_texture = self.adjust_uv_and_texture(texture)

        # image, alpha, _ = renderer(points=[vertex_positions, mesh_faces],
        #                            uv_bxpx2=input_uvs,
        #                            texture_bx3xthxtw=input_texture,
        #                            ft_fx3=mesh_face_textures,
        #                            **kwargs)
        return vertex_positions,mesh_faces,input_uvs,input_texture,mesh_face_textures
    
    def export_obj(self, path_prefix, vertex_positions, texture):
        assert len(vertex_positions.shape) == 2
        mesh_path = path_prefix + '.obj'
        material_path = path_prefix + '.mtl'
        material_name = os.path.basename(path_prefix)
        
        # Export mesh .obj
        with open(mesh_path, 'w') as file:
            print('mtllib ' + os.path.basename(material_path), file=file)
            for v in vertex_positions:
                print('v {:.5f} {:.5f} {:.5f}'.format(*v), file=file)
            for uv in self.mesh.uvs:
                print('vt {:.5f} {:.5f}'.format(*uv), file=file)
            print('usemtl ' + material_name, file=file)
            for f, ft in zip(self.mesh.faces, self.mesh.face_textures):
                print('f {}/{} {}/{} {}/{}'.format(f[0]+1, ft[0]+1, f[1]+1, ft[1]+1, f[2]+1, ft[2]+1), file=file)
                
        # Export material .mtl
        with open(material_path, 'w') as file:
            print('newmtl ' + material_name, file=file)
            print('Ka 1.000 1.000 1.000', file=file)
            print('Kd 1.000 1.000 1.000', file=file)
            print('Ks 0.000 0.000 0.000', file=file)
            print('d 1.0', file=file)
            print('illum 1', file=file)
            print('map_Ka ' + material_name + '.png', file=file)
            print('map_Kd ' + material_name + '.png', file=file)
            
        # Export texture
        import imageio
        texture = (texture.permute(1, 2, 0)*255).clamp(0, 255).cpu().byte().numpy()
        imageio.imwrite(path_prefix + '.png', texture)
                
    @staticmethod
    def _monkey_patch_dependencies():
        if version.parse(torch.__version__) < version.parse('1.2'):
            def torch_where_patched(*args, **kwargs):
                if len(args) == 1:
                    return (torch.nonzero(args[0]), )
                else:
                    return torch._where_original(*args)

            torch._where_original = torch.where
            torch.where = torch_where_patched
            
        if version.parse(torch.__version__) >= version.parse('1.5'):
            from .monkey_patches import compute_adjacency_info_patched
            # Monkey patch
            kal.rep.Mesh.compute_adjacency_info = staticmethod(compute_adjacency_info_patched)
                
                