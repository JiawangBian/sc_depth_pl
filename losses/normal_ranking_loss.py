import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########


def randomSamplingNormal(inputs, targets, masks, sample_num):

    # find A-B point pairs from predictions
    num_effect_pixels = torch.sum(masks)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).to(device)
    valid_inputs = inputs[:, masks]
    valid_targes = targets[:, masks]
    inputs_A = valid_inputs[:, shuffle_effect_pixels[0: sample_num * 2: 2]]
    inputs_B = valid_inputs[:, shuffle_effect_pixels[1: sample_num * 2: 2]]
    # find corresponding pairs from GT
    targets_A = valid_targes[:, shuffle_effect_pixels[0: sample_num * 2: 2]]
    targets_B = valid_targes[:, shuffle_effect_pixels[1: sample_num * 2: 2]]
    if inputs_A.shape[1] != inputs_B.shape[1]:
        num_min = min(targets_A.shape[1], targets_B.shape[1])
        inputs_A = inputs_A[:, :num_min]
        inputs_B = inputs_B[:, :num_min]
        targets_A = targets_A[:, :num_min]
        targets_B = targets_B[:, :num_min]
    return inputs_A, inputs_B, targets_A, targets_B


###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = torch.div(idx, cols, rounding_mode='floor')  # idx // cols
    c = idx - r * cols
    return r, c


def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx


def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):
    # find edges
    edges_max = edges_img.max()
    edges_min = edges_img.min()
    edges_mask = edges_img.ge(edges_max * 0.1)
    edges_loc = edges_mask.nonzero(as_tuple=False)

    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = thetas_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(
        0, minlen, (sample_num,), dtype=torch.long).to(device)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    # compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(3, 20, (4, sample_num)).to(device)
    pos_or_neg = torch.ones(4, sample_num).to(device)
    pos_or_neg[:2, :] = -pos_or_neg[:2, :]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = (
        col_anchors.unsqueeze(0).expand(4, sample_num).long()
        + torch.round(
            distance_matrix.double() * torch.abs(torch.cos(theta_anchors)).unsqueeze(0)
        ).long()
    )
    row = (
        row_anchors.unsqueeze(0).expand(4, sample_num).long()
        + torch.round(
            distance_matrix.double() * torch.abs(torch.sin(theta_anchors)).unsqueeze(0)
        ).long()
    )

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col < 0] = 0
    col[col > w - 1] = w - 1
    row[row < 0] = 0
    row[row > h - 1] = h - 1

    # a-b, b-c, c-d
    a = sub2ind(row[0, :], col[0, :], w)
    b = sub2ind(row[1, :], col[1, :], w)
    c = sub2ind(row[2, :], col[2, :], w)
    d = sub2ind(row[3, :], col[3, :], w)
    A = torch.cat((a, b, c), 0)
    B = torch.cat((b, c, d), 0)

    # mask map for visualization
    # mask_map_A = np.zeros((h, w), np.bool)
    # mask_map_A[row[0,:].cpu().numpy(), col[0,:].cpu().numpy()] = True
    # mask_map_A[row[1,:].cpu().numpy(), col[1,:].cpu().numpy()] = True
    # mask_map_B = np.zeros((h, w), np.bool)
    # mask_map_B[row[2, :].cpu().numpy(), col[2, :].cpu().numpy()] = True
    # mask_map_B[row[3, :].cpu().numpy(), col[3, :].cpu().numpy()] = True

    inputs_A = inputs[:, A]
    inputs_B = inputs[:, B]
    targets_A = targets[:, A]
    targets_B = targets[:, B]
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())
    return (
        inputs_A,
        inputs_B,
        targets_A,
        targets_B,
        masks_A,
        masks_B,
        sample_num,
        row,
        col,
    )


######################################################
# EdgeguidedNormalRankingLoss
#####################################################
class EdgeguidedNormalRankingLoss(nn.Module):
    def __init__(
        self,
        point_pairs=10000,
        cos_theta1=0.25,
        cos_theta2=0.98,
        cos_theta3=0.5,
        cos_theta4=0.86,
        mask_value=-1e-8,
    ):
        super(EdgeguidedNormalRankingLoss, self).__init__()
        self.point_pairs = point_pairs  # number of point pairs
        self.mask_value = mask_value
        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree
        self.kernel = torch.tensor(
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
            requires_grad=False,
        )[None, None, :, :].to(device)

    def getEdge(self, images):
        n, c, h, w = images.size()
        a = (
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        b = (
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def getNormalEdge(self, normals):
        n, c, h, w = normals.size()
        a = (
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        b = (
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .to(device)
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        gradient_x = torch.abs(F.conv2d(normals, a, groups=c))
        gradient_y = torch.abs(F.conv2d(normals, b, groups=c))
        gradient_x = gradient_x.mean(dim=1, keepdim=True)
        gradient_y = gradient_y.mean(dim=1, keepdim=True)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def forward(self, gt_depths, images, inputs_normal, targets_normal):
        """
        inputs and targets: surface normal image
        images: rgb images
        """
        masks = gt_depths > self.mask_value

        inputs = inputs_normal
        targets = targets_normal
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)
        # find edges from normals
        edges_normal, thetas_normal = self.getNormalEdge(targets)
        mask_img_border = torch.ones_like(
            edges_normal)  # normals on the borders
        mask_img_border[:, :, 5:-5, 5:-5] = 0
        edges_normal[mask_img_border.bool()] = 0
        # find edges from depth
        edges_depth, _ = self.getEdge(gt_depths)
        edges_depth_mask = edges_depth.ge(edges_depth.max() * 0.1)
        edges_mask_dilate = torch.clamp(
            torch.nn.functional.conv2d(
                edges_depth_mask.float(), self.kernel, padding=(1, 1)
            ),
            0,
            1,
        ).bool()
        edges_normal[edges_mask_dilate] = 0
        edges_img[edges_mask_dilate] = 0
        # =============================
        n, c, h, w = targets.size()

        inputs = inputs.contiguous().view(n, c, -1).double()
        targets = targets.contiguous().view(n, c, -1).double()
        masks = masks.contiguous().view(n, -1)
        edges_img = edges_img.contiguous().view(n, -1).double()
        thetas_img = thetas_img.contiguous().view(n, -1).double()
        edges_normal = edges_normal.view(n, -1).double()
        thetas_normal = thetas_normal.view(n, -1).double()

        # # initialization
        # loss = torch.DoubleTensor([0.0]).to(device)
        losses = []
        for i in range(n):
            # Edge-Guided sampling
            (
                inputs_A,
                inputs_B,
                targets_A,
                targets_B,
                masks_A,
                masks_B,
                sample_num,
                row_img,
                col_img,
            ) = edgeGuidedSampling(
                inputs[i, :],
                targets[i, :],
                edges_img[i],
                thetas_img[i],
                masks[i, :],
                h,
                w,
            )
            (
                normal_inputs_A,
                normal_inputs_B,
                normal_targets_A,
                normal_targets_B,
                normal_masks_A,
                normal_masks_B,
                normal_sample_num,
                row_normal,
                col_normal,
            ) = edgeGuidedSampling(
                inputs[i, :],
                targets[i, :],
                edges_normal[i],
                thetas_normal[i],
                masks[i, :],
                h,
                w,
            )

            # Combine EGS + EGNS
            inputs_A = torch.cat((inputs_A, normal_inputs_A), 1)
            inputs_B = torch.cat((inputs_B, normal_inputs_B), 1)
            targets_A = torch.cat((targets_A, normal_targets_A), 1)
            targets_B = torch.cat((targets_B, normal_targets_B), 1)
            masks_A = torch.cat((masks_A, normal_masks_A), 0)
            masks_B = torch.cat((masks_B, normal_masks_B), 0)

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A & masks_B

            # GT ordinal relationship
            target_cos = torch.abs(torch.sum(targets_A * targets_B, dim=0))
            input_cos = torch.abs(torch.sum(inputs_A * inputs_B, dim=0))

            losses += [torch.abs(target_cos-input_cos)]

            # # Random Sampling regression
            # random_sample_num = torch.sum(mask_cos10 & consistency_mask)
            # (
            #     random_inputs_A,
            #     random_inputs_B,
            #     random_targets_A,
            #     random_targets_B,
            # ) = randomSamplingNormal(
            #     inputs[i, :], targets[i, :], masks[i, :], random_sample_num
            # )
            # # GT ordinal relationship
            # random_target_cos = torch.abs(
            #     torch.sum(random_targets_A * random_targets_B, dim=0)
            # )
            # random_input_cos = torch.abs(
            #     torch.sum(random_inputs_A * random_inputs_B, dim=0)
            # )
            # losses += [torch.abs(random_target_cos - random_input_cos)]

        loss = torch.cat(losses, dim=0).mean()
        return loss
