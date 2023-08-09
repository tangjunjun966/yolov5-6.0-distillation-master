

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.loss import smooth_BCE,FocalLoss









class FocalLoss_distill(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss_distill, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss








class ComputeLossDistill:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss_distill(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        # p为模型预测输出结果
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors ,tconf = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification

                lcls += self.BCEcls(ps[:, 5:], tcls[i])  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image_id,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  #  每个点anchor数量(3), targets(每个batch中的标签个数)
        tcls, tbox, indices, anch ,tconf = [], [], [], [], [] # tcls表示类别，tbox表示box的坐标（x,y,w,h），indices表示图像索引，anch表示选取的anchor的索引
        gain = torch.ones(targets.shape[-1]+1, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # [na,nt] same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # targets[image_id,x,y,w,h，conf,...cls,anchor_id]
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets



        for i in range(self.nl):  # 循环3个特征层
            anchors, shape = self.anchors[i], p[i].shape
            gain[1:5] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7),在特征图中恢复gt尺寸，[img_id,x,y,w,h,conf,...cls,anchor_id]
            if nt:
                # Matches,选择正负样本方法，通过gt与anchor的wh比列筛选
                r = t[..., 3:5] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter，通过筛除后获得正样本

                # Offsets　获取选择完成的box的*中心点*坐标-gxy（以图像左上角为坐标原点），并转换为以特征图右下角为坐标原点的坐标-gxi
                gxy = t[:, 1:3]  # grid xy
                gxi = gain[[1, 2]] - gxy  # inverse 特征图右下角为坐标原点
                # 分别判断box的（x，y）坐标是否大于1，并距离网格左上角的距离（准确的说是y距离网格上边或x距离网格左边的距离）小于0.5，
                # 如果（x，y）中满足上述两个条件，则选中.gxy.shape=[182,2]，包含x,y,所以判别后转置得到j,k,2个结果
                # 对转换之后的box的（x，y）坐标分别进行判断是否大于1，并距离网格右下角的距离（准确的说是y距离网格下边或x距离网格右边的距离）距离小于0.5，
                # 如果（x，y）中满足上述两个条件，为Ture，
                j, k = ((gxy % 1 < g) & (gxy > 1)).T    # gxy>1，以左上角为坐标原点，表示排除上边与左边边缘格子
                l, m = ((gxi % 1 < g) & (gxi > 1)).T    # gxi>1同理，以右下角为坐标原点，排除右边与下边边缘格子
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 第一行为自己本身正样本值
                t = t.repeat((5, 1, 1))[j]  # 根据j挑选正样本，但未移动相邻网格
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 根据j处理对应正样本偏置(确定移动相邻网格)
            else:
                t = targets[0]
                offsets = 0

            # Define  b=img_id,c=[...cls],conf=conf-->预测置信度 gxy=grid xy, gwh=grid wh, a=anchors_id
            b=t[:,0].long()
            c=t[:,6:-1]
            conf=t[:,5]
            gxy= t[:,1:3]
            gwh=t[:,3:5]
            a=t[:,-1].long()


            gij = (gxy - offsets).long()  # xy与offsets对应
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image_id, anchor_id,与网格坐标grid_x，grid_y
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 获取（x,y)相对于网格点的偏置，以及box的宽高
            anch.append(anchors[a])  # anchors  获得对应的anchor
            tcls.append(c)  # class 获得对应类别
            tconf.append(conf)




        return tcls, tbox, indices, anch,tconf







