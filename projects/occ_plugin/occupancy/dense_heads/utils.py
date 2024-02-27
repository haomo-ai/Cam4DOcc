# borrowed from https://github.com/GuoPingPan/RPVNet/blob/main/core/models/utils/utils.py

import time
import numpy as np

import torch
from torch.nn.functional import grid_sample

import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point',
           'range_to_point','point_to_range']


def initial_voxelize(z: PointTensor, after_res) -> SparseTensor:

    new_float_coord = torch.cat(
        [z.C[:, :3]  / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.round(new_float_coord).int())

    sparse_hash = torch.unique(pc_hash)

    idx_query = F.sphashquery(pc_hash, sparse_hash)

    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.round(new_float_coord), idx_query,counts)

    inserted_coords = torch.round(inserted_coords).int()

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)

    new_tensor.cmaps.setdefault((1,1,1), new_tensor.coords)

    z.additional_features['idx_query'][(1,1,1)] = idx_query
    z.additional_features['counts'][(1,1,1)] = counts

    return new_tensor.to(z.F.device)


def point_to_voxel(x: SparseTensor, z: PointTensor) -> SparseTensor:
    if z.additional_features is None or z.additional_features['idx_query'] is None \
            or z.additional_features['idx_query'].get(x.s) is None:

        pc_hash = F.sphash(
            torch.cat([
                torch.round(z.C[:, :3] / x.s[0]).int(),
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)

        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


def voxel_to_point(x: SparseTensor, z: PointTensor, nearest=False) -> torch.Tensor:
    if z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None \
            or z.weights.get(x.s) is None:

        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)

        old_hash = F.sphash(
            torch.cat([
                torch.round(z.C[:, :3] / x.s[0]).int(),
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)


        pc_hash = F.sphash(x.C.to(z.F.device))

        idx_query = F.sphashquery(old_hash, pc_hash)

        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()

        idx_query = idx_query.transpose(0, 1).contiguous()

        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1

        new_feat = F.spdevoxelize(x.F, idx_query, weights)

        if x.s == (1,1,1):
            z.idx_query[x.s] = idx_query
            z.weights[x.s] = weights
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))

    return new_feat

def range_to_point(x,px,py):

    r2p = []

    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        pypx = torch.stack([p_x,p_y],dim=2).to(px[0].device)
        resampled = grid_sample(x[batch].unsqueeze(0),pypx.unsqueeze(0))
        r2p.append(resampled.squeeze().permute(1,0))
    return torch.concat(r2p,dim=0)


def point_to_range(range_shape,pF,px,py):
    H, W = range_shape
    cnt = 0
    r = []
    # t1 = time.time()
    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        image = torch.zeros(size=(H,W,pF.shape[1])).to(px[0].device)
        image_cumsum = torch.zeros(size=(H,W,pF.shape[1])) + 1e-5

        p_x = torch.floor((p_x/2. + 0.5) * W).long()
        p_y = torch.floor((p_y/2. + 0.5) * H).long()

        ''' v1: directly assign '''
        # image[p_y,p_x] = pF[cnt:cnt+p_x.shape[1]]

        ''' v2: use average '''
        image[p_y,p_x] += pF[cnt:cnt+p_x.shape[1]]
        image_cumsum[p_y,p_x] += torch.ones(pF.shape[1])
        image = image/image_cumsum.to(px[0].device)

        r.append(image.permute(2,0,1))
        cnt += p_x.shape[1]
    return torch.stack(r,dim=0).to(px[0].device)