import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
import os
import torch.nn.functional as F
import numpy as np
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
from gp import GP
from vmod import normalize_rows


def encode_Y(vae, train_queue):

    vae.eval()

    with torch.no_grad():

        n = train_queue.dataset.Y.shape[0]
        Zm = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()
        Zs = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()

        for batch_i, data in enumerate(train_queue):
            y = data[0].cuda()
            idxs = data[-1].cuda()
            zm, zs = vae.encode(y)
            Zm[idxs], Zs[idxs] = zm.detach(), zs.detach()

    return Zm, Zs


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="data_chairs.h5",
    help="dataset path",
)

parser.add_option(
    "--pose",
    dest="pose",
    type=str,
    default="pose.npy",
    help="pose file",
)

parser.add_option(
    "--outdir", dest="outdir", type=str, default="./../out/vae", help="output dir"
)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default=None)
parser.add_option("--vae_weights", dest="vae_weights", type=str, default=None)
parser.add_option("--gp_weights", dest="gp_weights", type=str, default=None)
parser.add_option("--vm_weights", dest="vm_weights", type=str, default=None)

parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")



(opt, args) = parser.parse_args()
opt_dict = vars(opt)


if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt.outdir, "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.outdir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

vm_weights = torch.load(opt.vm_weights)
gp_weights = torch.load(opt.gp_weights)
vae_cfg = pickle.load(open(opt.vae_cfg, "rb"))

def main():

    torch.manual_seed(opt.seed)

    # load pretrained VAE weights
    vae = FaceVAE(**vae_cfg).to(device)
    RV = torch.load(opt.vae_weights)
    vae.load_state_dict(RV)
    vae.to(device)

    gp = GP(n_rand_effs=1)
    gp = torch.nn.DataParallel(gp)
    gp.load_state_dict(gp_weights)
    gp.to(device)

    img, obj, view = read_face_data(opt.data)  # image, object, and view

    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    Dt = torch.tensor(obj["train"][:, 0].long()).to(device)
    wt = torch.tensor(view["train"][:, 0].long()).to(device)
    train_queue = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = FaceDataset(img["test"], obj["test"], view["test"])
    Dtest = Variable(obj["test"][:, 0].long(), requires_grad=False).to(device)
    wtest = Variable(view["test"][:, 0].long(), requires_grad=False).to(device)
    test_queue = DataLoader(test_data, batch_size=64, shuffle=False)

    ell = vm_weights['ell']
    poses = np.load(opt.pose)
    p = poses.shape[0]

    Xt = F.embedding(Dt, normalize_rows(vm_weights['x0']))

    v1 = torch.zeros(p, p).cuda()
    for i in range(p):
        for j in range(p):
            R1 = torch.tensor((poses[i].reshape(3, 4))[:3, :3]).float().to(device)
            R2 = torch.tensor((poses[j].reshape(3, 4))[:3, :3]).float().to(device)
            v1[i, j] = torch.exp(-0.5 * (
                torch.trace(torch.exp(torch.diag(ell)) - R1.mm(torch.exp(torch.diag(ell)).mm(R2.transpose(0, 1))))))

    Wt = F.embedding(wt, v1)

    Vt = torch.einsum("ij,ik->ijk", [Xt, Wt])
    Vt = Vt.reshape([Vt.shape[0], -1])


    Xtest = F.embedding(Dtest, normalize_rows(vm_weights['x0']))
    Wtest = F.embedding(wtest, v1)


    Vtest = torch.einsum("ij,ik->ijk", [Xtest, Wtest])
    Vtest = Vtest.reshape([Vtest.shape[0], -1])


    Zm, Zs = encode_Y(vae, train_queue)

    vs = gp.module.get_vs()
    U, UBi, _ = gp.module.U_UBi_Shb([Vt], vs)
    Kiz = gp.module.solve(Zm, U, UBi, vs).to(device)

    Zo = vs[0] * Vtest.mm(Vt.transpose(0, 1).mm(Kiz))

    mse = []
    with torch.no_grad():
        for batch_i, data in enumerate(test_queue):
            idxs = data[-1].cuda()
            Yo = vae.decode(Zo[idxs])

            for idx, _ in enumerate(idxs):
                y = np.expand_dims(data[0][idx].numpy(), 0).transpose(0, 2, 3, 1)[0]
                res = np.expand_dims(Yo[idx].cpu().numpy(), 0).transpose(0, 2, 3, 1)[0]
                mse.append(np.mean((y - res) ** 2))

    print('MSE: %f std: %f ' %(np.mean(mse), np.std(mse)))

if __name__ == "__main__":
    main()
