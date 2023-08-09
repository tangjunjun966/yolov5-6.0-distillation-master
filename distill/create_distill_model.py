import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.general import xyxy2xywh,xyxy2xywhn

# 构建teacher模型
def create_teacher_model(weights,device):
    # device = torch.device('cuda:0')

    model=attempt_load(weights, map_location=device).eval()

    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names


    teacher_model={'model':model,
                   'stride':stride,
                   'names':names
                   }

    return teacher_model



from distill.distill_general import non_max_suppression


def collate_distill(batch):
    img, label, path, shapes = zip(*batch)  # transposed


    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes
def infer_teacher(model_info,imgs,student_cls_names):
    teacher_cls_names = model_info['names']
    assert sorted(student_cls_names)==sorted(teacher_cls_names),ValueError("class names no match between teach and student")


    model=model_info['model']
    pred=model(imgs)[0]
    pre_h,pre_w=imgs.shape[-2:]
    output_distill, output = non_max_suppression(pred, 0.25, 0.45,  False, max_det=1000)
    # [x1,y1,x2,y2,conf,...cls_name]

    # 添加图片索引维度
    output_distill_tmp=[]
    for i in range(len(output_distill)):
        if len(output_distill[i])>0:
            img_idx=torch.ones(output_distill[i].shape[0],1)*i
            output_distill_tmp.append( torch.cat((img_idx.to(output_distill[i].device),output_distill[i]),1) )

    if len(output_distill_tmp)>1:
        teacher_target=torch.cat(output_distill_tmp)
    elif(len(output_distill_tmp)==1):
        teacher_target=output_distill_tmp[0]
    else:
        teacher_target=[]

    if len(teacher_target)>0:
        teacher_target[:,1:5]=xyxy2xywhn(teacher_target[:, 1:5], w=pre_w, h=pre_h, clip=True, eps=1E-3)

    # 调整类别顺序
    teacher_target_tmp = teacher_target.detach().clone()

    for i,t_name in enumerate(teacher_cls_names):
        s_idx=list(student_cls_names).index(t_name)
        teacher_target[:,4+s_idx]=teacher_target_tmp[:,4+i]
    # [image_id,x1,y1,x2,y2,conf,...cls_name]
    return teacher_target






if __name__ == '__main__':

    root='../yolov5s.pt'
    device = torch.device('cuda:0')
    model=create_teacher_model(root,device)
    imgs=torch.ones((4,3,640,640)).to(device)
    res=infer_teacher(model,imgs)
    print(res)












