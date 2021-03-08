import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class GP(nn.Module):
    def __init__(self, n_rand_effs=1, vsum2one=True):

        super(GP, self).__init__()

        # store stuff
        self.n_rand_effs = n_rand_effs
        self.vsum2one = vsum2one

        # define variables
        self.n_vcs = n_rand_effs + 1
        self.lvs = nn.Parameter(torch.zeros([self.n_vcs]))

    def U_UBi_Shb(self, Vs, vs):

        # compute U and V
        V = torch.cat([torch.sqrt(vs[i].cpu()) * V.cpu() for i, V in enumerate(Vs)], 1)
        U = V / torch.sqrt(vs[-1].cpu())
        eye = torch.eye(U.shape[1])
        B = torch.mm(torch.transpose(U, 0, 1), U) + eye
        B = B.cpu()
        Ub, Shb, Vb = torch.svd(B)
        Bi = torch.inverse(B)
        UBi = torch.mm(U.cpu(), Bi.cpu())

        return U.cpu(), UBi, Shb.cpu()

    def solve(self, X, U, UBi, vs):

        UX = U.transpose(0, 1).mm(X.cpu())
        UBiUX = UBi.mm(UX)
        RV = (X.cpu() - UBiUX) / vs[-1].cpu()

        return RV

    def get_vs(self):
        if self.vsum2one:
            rv = F.softmax(self.lvs, 0)
        else:
            rv = torch.exp(self.lvs) / float(self.n_vcs)
        return rv

    def taylor_coeff(self, X, Vs):

        # solve
        vs = self.get_vs()
        U, UBi, Shb = self.U_UBi_Shb(Vs, vs)
        Xb = self.solve(X, U, UBi, vs)

        # variables to fill
        Vbs = []
        vbs = Variable(torch.zeros([self.n_rand_effs + 1]), requires_grad=False)#.cuda()

        # compute Vbs and vbs
        for iv, V in enumerate(Vs):
            XbV = Xb.transpose(0, 1).mm(V.cpu())
            XbXbV = Xb.mm(XbV)
            KiV = self.solve(V, U, UBi, vs)
            Vb = vs[iv] * (X.shape[1] * KiV - XbXbV)
            Vbs.append(Vb)

            # compute vgbar
            vbs[iv] = -0.5 * torch.einsum("ij,ij->", [XbV, XbV])
            vbs[iv] += 0.5 * X.shape[1] * torch.einsum("ij,ij->", [V.cpu(), KiV])

        # compute vnbar
        trKi = (X.shape[0] - torch.einsum("ni,ni->", [UBi, U])) / vs[-1]
        vbs[-1] = -0.5 * torch.einsum("ij,ij->", [Xb, Xb])
        vbs[-1] += 0.5 * X.shape[1] * trKi

        # compute negative log likelihood (nll)
        quad_term = torch.einsum("ij,ij->i", [X.cpu(), Xb])[:, None]
        logdetK = Xb.shape[0] * Xb.shape[1] * torch.log(vs[-1])
        logdetK += Xb.shape[1] * torch.sum(torch.log(Shb))
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        # detach all
        Xb = Xb.detach()
        Vbs = [Vb.detach() for Vb in Vbs]
        vbs = vbs.detach()
        nll = nll.detach()

        return Xb, Vbs, vbs, nll

    def nll(self, X, Vs):

        # solve
        vs = self.get_vs()
        U, UBi, Shb = self.U_UBi_Shb(Vs, vs)
        Xb = self.solve(X, U, UBi, vs)

        # compute negative log likelihood (nll)
        quad_term = torch.einsum("ij,ij->i", [X, Xb])[:, None]
        logdetK = Xb.shape[0] * Xb.shape[1] * torch.log(vs[-1])
        logdetK += Xb.shape[1] * torch.sum(torch.log(Shb))
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        return nll

    def nll_ineff(self, X, Vs):
        vs = self.get_vs()
        V = torch.cat([torch.sqrt(vs[i]) * V for i, V in enumerate(Vs)], 1)
        K = V.mm(V.transpose(0, 1)) + vs[-1] * torch.eye(X.shape[0]).cuda()
        Uk, Shk, Vk = torch.svd(K)
        Ki = torch.inverse(K)
        Xb = Ki.mm(X)

        quad_term = torch.einsum("ij,ij->i", [X, Xb])[:, None]
        logdetK = X.shape[1] * torch.log(Shk).sum()
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        return nll

    def taylor_expansion(self, X, Vs, Xb, Vbs, vbs):
        rv = torch.einsum("ij,ij->i", [Xb, X.cpu()])[:, None]
        for V, Vb in zip(Vs, Vbs):
            rv += torch.einsum("ij,ij->i", [Vb, V.cpu()])[:, None]
        vs = self.get_vs()
        rv += torch.einsum("i,i->", [vbs, vs.cpu()]) / float(X.shape[0])
        return rv.cuda()

