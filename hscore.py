import torch


def neg_hscore(f,g):
    f0 = f - torch.mean(f,0)
    g0 =g-torch.mean(g,0)
    corr=torch.mean(torch.sum(f0*g0,1))
    cov_f=torch.mm(torch.t(f0),f0)/(f0.size()[0]-1.)
    cov_g=torch.mm(torch.t(g0),g0)/(g0.size()[0]-1.)
    return -corr+torch.trace(torch.mm(cov_f,cov_g))/2.