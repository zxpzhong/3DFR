import torch.nn.functional as F
import torch
# external loss function from loss dir
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def CE(output, target):
    return F.binary_cross_entropy(output, target)

def L2(recon, data):
    return F.mse_loss(recon,data)

def L1(recon,data):
    return F.l1_loss(recon,data)

def Lap_Loss(adj,vertices):
    # 邻接矩阵N*N @ 顶点N*3 = N*3 -> 拉普拉斯坐标
    new_lap = torch.matmul(adj, vertices)
    # loss = mean(新的lap N*3 - 原始顶点N*3)
    # 新的拉普拉斯坐标和原坐标的差距最小 -> 让顶点的分布尽可能均匀,每个点处于周围的中间!
    loss = torch.mean((new_lap - vertices) ** 2) * vertices.shape[0] * 3
    return loss 


def compute_normals(vertex_positions,mesh):
    """
    Compute face normals from the *final* vertex positions (not deltas).
    """
    # 取出所有三角面片的第一个顶点、第二个顶点、第三个顶点
    a = vertex_positions[:, mesh.faces[:, 0]]
    b = vertex_positions[:, mesh.faces[:, 1]]
    c = vertex_positions[:, mesh.faces[:, 2]]
    # 三角形第一条边
    v1 = b - a
    # 第二条变
    v2 = c - a
    # 叉积求法向
    normal = torch.cross(v1, v2, dim=2)
    # 长度归一化
    return F.normalize(normal, dim=2)

def Loss_flat(raw_vtx,mesh): 
    # mesh.ff.shape : 2040*3（面数*3）
    # ff的意思是face2face
    # 求出每个三角形的法向
    norms = compute_normals(raw_vtx,mesh)
    loss  = 0.
    for i in range(3):
        norm1 = norms # 这个就是上面计算得到的每个面的法向
        norm2 = norms[:, mesh.ff[:, i]] # 要按照面的相邻关系对应上
        cos = torch.sum(norm1 * norm2, dim=2)
        loss += torch.mean((cos - 1) ** 2) 
    loss *= (mesh.faces.shape[0]/2.)
    return loss

# TODO: 其他损失增加, DIB原生的几个loss

# 1. 轮廓mask IOU 
# 2. colored image L1 loss
# 3. 形状正则化的smooth loss
# 4. 形状正则化的lap loss


def Edge_regularization(pred, edges):
    """
    :param pred: batch_size * num_points * 3
    :param edges: num_edges * 2
    :return:
    """
    l2_loss = nn.MSELoss(reduction='mean')
    temp = pred[:, edges[:, 0]] - pred[:, edges[:, 1]]
    return l2_loss(temp,torch.zeros_like(temp).cuda()) * pred.size(-1)
