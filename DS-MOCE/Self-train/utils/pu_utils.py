from bdb import set_trace
import imp
import torch
import torch.nn.functional as F
import torch.nn as nn


def loss_func(yTrue, yPred):
        y = torch.eye(2)[yTrue].float().cuda()
        if len(y.shape) == 1:
            y = y[None, :]
        # y = torch.from_numpy(yTrue).float().cuda()
        loss = torch.mean((y * (1 - yPred)).sum(dim=1))
        return loss


def gce_loss(logits,targets,q=0.7):
    pred = F.softmax(logits, dim=-1)
    pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
    loss = (1-pred**q) / q
    loss = loss.view(-1).sum()
    #loss = (loss.view(-1).sum()/targets.size(-1))
    # import pdb;pdb.set_trace()
    return loss


def my_pu_loss(labels,outputs,args):
    # for pseudo labels shape is like (8,512,3)

    if labels.shape == outputs[1].shape:
        flag = torch.argmax(labels,axis=2)
    else:
        flag = labels # shape (8,512)
    positive_mask = (flag > 0) # shape (8,512)
    positive_mask_all = torch.stack((positive_mask,positive_mask,positive_mask),dim=2)# shape (8,512,3) 
    unlabeled_mask = (flag == 0) # shape (8,512)
    unlabeled_mask_all = torch.stack((unlabeled_mask,unlabeled_mask,unlabeled_mask),dim=2) # shape(8,512,3)
    
    # results = torch.argmax(outputs[1], axis=2) # no argmax
    # 
    
    assert outputs[1].shape == positive_mask_all.shape

    hP = (torch.masked_select(outputs[1],positive_mask_all.cuda())).contiguous().view(-1,3)
    hU = (torch.masked_select(outputs[1],unlabeled_mask_all.cuda())).contiguous().view(-1,3)

    #import pdb;pdb.set_trace()
    matrix = torch.tensor([[1,0],[0,1],[0,1]],dtype=torch.float32).cuda()

    mae = nn.L1Loss()
    positive = torch.eye(2)[1].int().cuda()
    negative =torch.eye(2)[0].int().cuda()

    # import pdb;pdb.set_trace()
    bi_logits =torch.mm(hP,matrix)
    # bi_labels = torch.full([hP.shape[0]],1,dtype=torch.int64).cuda()
    # pRisk = gce_loss(bi_logits,bi_labels)

    bi_logits_u =torch.mm(hU,matrix)
    # bi_labels_u = torch.full([hU.shape[0]],0,dtype=torch.int64).cuda()
    # uRisk = gce_loss(bi_logits_u,bi_labels_u)

    # import pdb;pdb.set_trace()

    pRisk = mae( F.softmax(bi_logits, dim=-1),positive.repeat(hP.shape[0]).view(-1,2))
    uRisk = mae( F.softmax(bi_logits_u, dim=-1),negative.repeat(hU.shape[0]).view(-1,2))

    # pRisk = loss_func(1,torch.mm(hP,matrix))
    # uRisk = loss_func(0,torch.mm(hU,matrix))

    nRisk = uRisk - args.prior * (1 - pRisk)
    m=5
    risk = m * pRisk + nRisk
    if nRisk < -args.beta:
        # print(nRisk.data)
        risk = -args.gamma * nRisk
        # print(risk.data)
    # risk = self.model.loss_func(label, result)
    
    # debug 
    # risk.backward() # no grad_fn
    # import pdb; pdb.set_trace()
    return risk

