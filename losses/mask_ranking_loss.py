import torch
from torch import nn

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class Mask_Ranking_Loss(nn.Module):
    def __init__(self, sample_ratio=0.1, filter_depth=1e-8):
        super(Mask_Ranking_Loss, self).__init__()
        self.sample_ratio = sample_ratio
        self.filter_depth = filter_depth

    def generate_global_target(self, depth, pred, theta=0.15):
        B, C, H, W = depth.shape
        mask_A = torch.rand(C, H, W).to(device)
        mask_A[mask_A >= (1 - self.sample_ratio)] = 1
        mask_A[mask_A < (1 - self.sample_ratio)] = 0
        idx = torch.randperm(mask_A.nelement())
        mask_B = mask_A.view(-1)[idx].view(mask_A.size())
        mask_A = mask_A.repeat(B, 1, 1).view(depth.shape) == 1
        mask_B = mask_B.repeat(B, 1, 1).view(depth.shape) == 1
        za_gt = depth[mask_A]
        zb_gt = depth[mask_B]
        mask_ignoreb = zb_gt > self.filter_depth
        mask_ignorea = za_gt > self.filter_depth
        mask_ignore = mask_ignorea | mask_ignoreb
        za_gt = za_gt[mask_ignore]
        zb_gt = zb_gt[mask_ignore]

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 > 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target

    def generate_percentMask_target(self, depth, pred, invalid_mask, theta=0.15):
        B, C, H, W = depth.shape
        valid_mask = ~invalid_mask
        gt_inval, gt_val, pred_inval, pred_val = None, None, None, None
        for bs in range(B):
            gt_invalid = depth[bs, :, :, :]
            pred_invalid = pred[bs, :, :, :]
            # select the area which belongs to invalid/occlusion
            mask_invalid = invalid_mask[bs, :, :, :]
            gt_invalid = gt_invalid[mask_invalid]
            pred_invalid = pred_invalid[mask_invalid]

            gt_valid = depth[bs, :, :, :]
            pre_valid = pred[bs, :, :, :]
            # select the area which belongs to valid/reliable
            mask_valid = valid_mask[bs, :, :, :]
            gt_valid = gt_valid[mask_valid]
            pre_valid = pre_valid[mask_valid]
            # generate the sample index. index range -> (0, len(gt_valid)). The amount -> gt_invalid.size()
            idx = torch.randint(0, len(gt_valid), gt_invalid.size())
            gt_valid = gt_valid[idx]
            pre_valid = pre_valid[idx]

            if bs == 0:
                gt_inval, gt_val, pred_inval, pred_val = gt_invalid, gt_valid, pred_invalid, pre_valid
                continue
            gt_inval = torch.cat((gt_inval, gt_invalid), dim=0)
            gt_val = torch.cat((gt_val, gt_valid), dim=0)
            pred_inval = torch.cat((pred_inval, pred_invalid), dim=0)
            pred_val = torch.cat((pred_val, pre_valid), dim=0)

        za_gt = gt_inval
        zb_gt = gt_val

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).to(device)
        target[mask1] = 1
        target[mask2] = -1

        return pred_inval, pred_val, target

    def cal_ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        pred_depth = z_A - z_B
        #log_loss = torch.mean(torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        # squared_loss = torch.mean(pred_depth[target == 0] ** 2)  # if pred depth is not zero adds to loss
        log_loss = torch.sum(
            torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        pointNum = len(target[target != 0])
        return log_loss, pointNum

    def get_unreliable(self, tgt_valid_weight):
        # invalidMask = tgt_valid_weight < 0.75
        B, C, H, W = tgt_valid_weight.shape
        unreliable_percent = 0.2
        invalidMask = torch.ones_like(tgt_valid_weight)
        for bs in range(B):
            weight = tgt_valid_weight[bs]
            maskIv = invalidMask[bs]
            weight = weight.view(-1)
            maskIv = maskIv.view(-1)

            weight_sorted, indices = torch.sort(weight)
            # each item in indices represent an index(valid)
            indices[:int(unreliable_percent*H*W)] = indices[H*W-1]
            # use indices for the selection. mask=0 -> valid
            maskIv[indices] = 0

        return invalidMask > 0

    def get_textureWeight(self, tgt_img):
        grad_img_x = torch.mean(
            torch.abs(tgt_img[:, :, :, :-1] - tgt_img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(tgt_img[:, :, :-1, :] - tgt_img[:, :, 1:, :]), 1, keepdim=True)
        textureWeight = torch.zeros_like(tgt_img[:, :1, :, :])
        textureWeight[:, :, :, :-1] = textureWeight[:, :, :, :-1] + grad_img_x
        textureWeight[:, :, :-1, :] = textureWeight[:, :, :-1, :] + grad_img_y
        textureWeight = textureWeight / 2.0
        return textureWeight

    def forward(self, pred_depth, gt_depth, tgt_valid_weight):
        # dynamic mask
        unreliableMask = self.get_unreliable(tgt_valid_weight)
        za_1, zb_1, target_1 = self.generate_percentMask_target(
            gt_depth, pred_depth, unreliableMask)
        loss_percentMask, pointNum_1 = self.cal_ranking_loss(
            za_1, zb_1, target_1)

        # global
        za_2, zb_2, target_2 = self.generate_global_target(
            gt_depth, pred_depth)
        loss_global, pointNum_2 = self.cal_ranking_loss(za_2, zb_2, target_2)

        total_loss = (loss_global + loss_percentMask)/(pointNum_2 + pointNum_1)
        return total_loss
