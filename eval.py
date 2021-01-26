import csv
import numpy as np
import scipy
from scipy.linalg import *
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

import argparse

parser = argparse.ArgumentParser(description='PIONEER')
parser.add_argument('--face_id', type=int, help='Face ID to use from the camera run dataset')
parser.add_argument('--kernel_mode', type=str, default='viewaware', choices=['viewaware', 'quat', 'euler'])
parser.add_argument('--data_path', type=str, help='Directory under which the /face0x directories reside')
parser.add_argument('--full_smoothing', action='store_true', help='Uses the full movie for training and testing')

args = parser.parse_args()

## CONFIGURATION

Kmode = args.kernel_mode #'viewaware' #'quat' #'euler' #'viewaware' 
face_id = args.face_id #2
root = Path(args.data_path) / 'face0{}'.format(face_id) #'C:/Users/arihe/git/PINE_SNG/PINE/src/PINE/face0{}'.format(face_id))
frames_csv_path = root / 'frames.csv'

includeTranslationInPoseDistance = False

def frameRangeForFace(face_id, full_smoothing):
    # The various adjustments are made to:
    # 1) Remove corrupted original frames
    # 2) Remove cases where StyleGAN and/or alignment fail
    # 3) Allowing the start and end frames symmetric enough so that the linear interpolation baseline does not trivially fail
    
    if face_id == 4:
        N = 506
        frames_available_N = 1080
    elif face_id == 3:
        frames_available_N = 1076
        N = 1076 - 726 + 1
    elif face_id == 2:
        frames_available_N = 976
        #Issues in StyleGAN / alignment require us to use the short version for (first, last) interpolation

        if not full_smoothing:
            N = 976 - 606 + 1
        else:
            N = 976
    elif face_id == 5:    
        frames_available_N = 1076
        N = 806 - 401  + 1
            
    return N, frames_available_N

def build_stylegan_z(path):
    path = Path(path)
    train_indices = []
    z = []
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith("z0.npy"):
            spath = path / filename
            z_i = np.load(spath).flatten()
            z += [z_i]
            print(spath)
            train_indices += [int(filename[:4])-1] #index correction
    return torch.from_numpy(np.array(z)), train_indices

def read_frames(frames_csv_path):
    timestamp = np.zeros(frames_available_N)
    frame_number = np.zeros(frames_available_N)
    pos = np.zeros((frames_available_N, 4))
    R_col1 = np.zeros((frames_available_N, 4))
    R_col2 = np.zeros((frames_available_N, 4))
    R_col3 = np.zeros((frames_available_N, 4))
    R = np.zeros((frames_available_N, 4, 4))

    with open(frames_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count >= frames_available_N:
                print("Stopping at line count {}. Skipping the rest.".format(line_count))
                break

            timestamp[line_count] = row[0]
            frame_number[line_count] = row[1]
            pos[line_count] = np.array(row[2:5] + [1])
            R_col1[line_count] = np.array(row[5:8] + [0])
            R_col2[line_count] = np.array(row[8:11] + [0])
            R_col3[line_count] = np.array(row[11:14] + [0])
            R[line_count] = np.stack((R_col1[line_count], R_col2[line_count], R_col3[line_count], pos[line_count]), axis=1)
            line_count += 1
        print(f'Processed {line_count} lines.')
        
        # Normalize all rotations to 1 (there have been some small discrepancies)

        for i in range(len(R)):
            Di = R[i,:3,:3].T.dot(R[i,:3,:3])
            R[i,:3,:3] /= np.sqrt(np.matrix.trace(Di) / 3)
            if i%100==0:
                print(i)

        return R

def pose_distance2(p1, p2):
    R1 = p1[:3,:3]
    R2 = p2[:3,:3]
    return np.sqrt(max(1e-12, np.matrix.trace(np.eye(3)-R1.T.dot(R2)))) #WAS: 0.000001

def genDistM(poses, indices1, indices2, distance_metric):
    n = len(poses)
    print("Poses: {}".format(n))
    D = np.zeros((len(indices1),len(indices2)))
    print(np.shape(D))
    for i,x in enumerate(indices1):
        for j,y in enumerate(indices2):
#            print((x,y))
            D[i,j] = distance_metric(poses[x],poses[y])
    return D

def Kformula(_gamma, _D, _ell):
    return _gamma**2 *  torch.exp(-_D / 2 / _ell**2)
    #Matern:
    #return (_gamma**2) * (1 + np.sqrt(3) * _D / _ell) * torch.exp(-np.sqrt(3) * _D / _ell)

def getHPs():  
    gamma = torch.tensor(np.sqrt(13.8156))
    ell = torch.tensor(1.0975)
    sigma = np.sqrt(torch.tensor(1.4))
    
    return gamma,ell,sigma

def gen_samples(mean, K, M):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K

    arguments:
    K   -- NxN kernel matrix
    M   -- number of samples (scalar)

    returns NxM matrix
    """
    _D = np.shape(K)[0]
    _z = np.random.normal(0,1,(_D,M)).T

    return (mean + (_z@np.linalg.cholesky(K).T)).T

# Separable view kernel

def separableKformula(p1, p2, _gamma, _ell):
    ret = _gamma**2
    for i in range(3):
        ret *= torch.exp(-2 * np.sin((p1[i]-p2[i]) / 2)**2 / _ell**2)
    return  ret

def genK_separable(poses, indices1, indices2, _gamma, _ell):
    n = len(poses)
    K = torch.zeros((len(indices1),len(indices2)))

    for i,x in enumerate(indices1):
        for j,y in enumerate(indices2):
            K[i,j] = separableKformula(poses[x], poses[y], _gamma, _ell)
    return K

def getSeparableK(Y, R, train_indices, test_indices, _gamma,_ell,_sigma):
    R_euler = Rot.from_matrix(R[:,:3,:3]).as_euler('xyz')

    K = genK_separable(R_euler, train_indices, train_indices,_gamma, _ell).unsqueeze(0)
    Kt = genK_separable(R_euler, test_indices, train_indices,_gamma, _ell).unsqueeze(0)
    Ktt = genK_separable(R_euler, test_indices, test_indices,_gamma, _ell).unsqueeze(0)

    K = K.cpu()
    y = torch.from_numpy(Y)

    N2 = len(train_indices)
    I = torch.eye(N2).expand(1, N2, N2).float().cpu()

    X,_ = torch.solve(y.type(torch.float), K+(_sigma**2)*I) #give the X in [K+(sigma**2)*I]*X = y

    Z2 = Kt.bmm(X)[0,:,:]
    
    return Z2, K, Kt, Ktt

# Quaternion view kernel

def quatKformula(q1, q2, _gamma, _ell):
    return _gamma**2 * torch.exp(-np.linalg.norm(q1 - q2)**2 / 2 / _ell**2)

def genK_quat(poses, indices1, indices2, _gamma, _ell):
    n = len(poses)
    K = torch.zeros((len(indices1),len(indices2)))
    for i,x in enumerate(indices1):
        for j,y in enumerate(indices2):
            K[i,j] = separableKformula(poses[x], poses[y], _gamma, _ell)
    return K

def getQuatK(Y, R, train_indices, test_indices, _gamma,_ell,_sigma):
    R_quat =  Rot.from_matrix(R[:,:3,:3]).as_quat()

    K = genK_quat(R_quat, train_indices, train_indices,_gamma, _ell).unsqueeze(0)
    Kt = genK_quat(R_quat, test_indices, train_indices,_gamma, _ell).unsqueeze(0)
    Ktt = genK_quat(R_quat, test_indices, test_indices,_gamma, _ell).unsqueeze(0)

    K = K.cpu()
    y = torch.from_numpy(Y)

    N2 = len(train_indices)
    I = torch.eye(N2).expand(1, N2, N2).float().cpu()

    X,_ = torch.solve(y.type(torch.float), K+(_sigma**2)*I) #give the X in [K+(sigma**2)*I]*X = y

    Z2 = Kt.bmm(X)[0,:,:]
    
    return Z2, K, Kt, Ktt

def getMUK(Y, R, train_indices, test_indices, gamma,ell,sigma):
    D  = torch.from_numpy(np.expand_dims(genDistM(R, train_indices, train_indices, pose_distance2), 0)).float()
    Dt = torch.from_numpy(np.expand_dims(genDistM(R, test_indices, train_indices, pose_distance2), 0)).float() #2nd WAS: 1080,2
    Dtt= torch.from_numpy(np.expand_dims(genDistM(R, test_indices, test_indices, pose_distance2), 0)).float() #2nd WAS: 1080,2

    K = Kformula(gamma, D, ell)
    Kt = Kformula(gamma, Dt, ell)
    Ktt = Kformula(gamma, Dtt, ell)

    K = K.cpu()
    y = torch.from_numpy(Y)

    N2 = len(train_indices)
    I = torch.eye(N2).expand(1, N2, N2).float().cpu()

    X,_ = torch.solve(y.type(torch.float), K+(sigma**2)*I) #give the X in [K+(sigma**2)*I]*X = y

    Z2 = Kt.bmm(X)[0,:,:]

    return Z2, K, Kt, Ktt

def getCovar(K, Kt, Ktt, sigma, N2):
    ### Create the covariance matrix for sampling

    I = torch.eye(N2).expand(1, N2, N2).float().cpu()
    K0 = K[0,:,:]
    X,_ = torch.solve(torch.transpose(Kt, 1,2), K0+(sigma**2)*I) #give the X in [K0+(sigma**2)*I]*X = Kt.T
    # I.e.  [K0+(sigma**2)*I]^-1 [K0+(sigma**2)*I]*X = [K0+(sigma**2)*I]^-1 * Kt.T
    print(X.size())
    cov = Ktt - Kt.bmm(X)
    # RIGHT:    Kt * (K + sigma2*I)^-1 * Kt'
    
    cov2use = cov[0,:,:]

    # Eigenvalue cleanup to make cov2use positive definite

    from numpy import linalg as LA
    w,v = LA.eigh(cov2use)
    w[w<1e-5] = 1e-5
    w2 = np.diag(w)

    cov2use = v.dot(w2).dot(np.linalg.inv(v))

    return cov2use

def getSamples(cov2use, mu, spf, step_offset, useFrames, useDims):
    print((N,useDims,spf))
    yc = torch.zeros((useFrames,m,spf), dtype=torch.float64)
    for i in range(useDims): #For each dimension, separately. This will take some time, since m=9216
        yc[:,i,:] = torch.from_numpy(gen_samples(mu[step_offset:(step_offset+useFrames),i].numpy(), cov2use, spf))
    return yc

gamma,ell,sigma = getHPs()
N, frames_available_N = frameRangeForFace(face_id = face_id, full_smoothing = args.full_smoothing)

if not args.full_smoothing:
    zpath     = root / 'first_last_interpolation'
    zout_path = root / 'stylegan.z2'
    ZstdPath  = root / Path('sigma_stylegan.z2.npy')
else:
    zpath     = root
    zout_path = root / 'stylegan_full_smoothing.z2'  

if Kmode == 'euler':
    zout_path = str(zout_path)[:-3] + '_euler.z2'
elif Kmode == 'quat':
    zout_path = str(zout_path)[:-3] + '_quat.z2'

print('Z from {} with N = {}'.format(zpath, N))

### Load the corner Z frames

z, train_indices = build_stylegan_z(zpath)
#print(train_indices)

first_train_index = np.min(train_indices)
test_indices = np.arange(first_train_index, first_train_index+N)
#print(test_indices)

R = read_frames(frames_csv_path)

#print(R[0])
#print(np.shape(R))

# We assume that the input z matrix contains all the training data that we have, and nothing else
### Remove the mean-per-dimension now and add it back at the end

Y_dimensionwise_mean = np.mean(z.cpu().numpy(),axis=0, dtype=np.float64)
Y = z.cpu().numpy() - Y_dimensionwise_mean
_,m = Y.shape

"""
z: Original (observed) training points
y: Dimensionwise-zero-mean training points
test_indices: Indices of the testing points 
mu: Predicted y mean values at the testing points
cov: Predicted covariances at the testing points
yc: Sampled y values at the testing points
"""

if True:
    print("SIGMA {}".format(sigma))
    print(gamma,ell,sigma)

    if Kmode == 'viewaware':
        Z2, K, Kt, Ktt = getMUK(Y, R, train_indices, test_indices, gamma,ell,sigma)
    elif Kmode == 'euler':
        Z2, K, Kt, Ktt = getSeparableK(Y, R, train_indices, test_indices, gamma,ell,sigma)
    elif Kmode == 'quat':
        Z2, K, Kt, Ktt = getQuatK(Y, R, train_indices, test_indices, gamma,ell,sigma)
    else:
        print('UNKNOWN MODE. FAIL.')

    # Just looking at the mu's, not the covariance

    Z2 += torch.from_numpy(Y_dimensionwise_mean).float()
    Z2r = Z2.numpy().reshape(Z2.size()[0], 18, 512)
    np.save(zout_path, Z2r)

    print("Saved fixed z at {} with shape: {}".format(zout_path, np.shape(Z2r)))

    if Kmode == 'viewaware':
        cov2use = getCovar(K=K, Kt=Kt, Ktt=Ktt, sigma=sigma, N2=len(train_indices))
        yc = getSamples(cov2use, mu=Z2, spf=1, step_offset=0, useFrames=N, useDims=z.size()[1]) #spf = samples per frame (=> 100)

        #Let's now get the final latents by adding back the per-dimension mean:
        zc = yc + torch.from_numpy(Y_dimensionwise_mean).unsqueeze(1)

        if not args.full_smoothing:
            print(ZstdPath)
            np.save(ZstdPath, zc)
        else:
            print("Full smoothing: Ignore std frames")

        # Mean + per-dimension mean
        Z3 = Z2+torch.from_numpy(Y_dimensionwise_mean).float()

        #D, K, Kt, Ktt only contain the frames that are listed in test/train_indices

        for d in [0,-1]: #range(1):
            for test_frame in [0,300,-1]:
                print("Test frame index [{}] (real frame {}) dim[{}]:".format(test_frame, test_indices[test_frame], d))
                for train_frame in [0,1]:
                    print("   Train frame index [{}] (real frame {}):".format(train_frame, train_indices[train_frame]))        
                    print("      Training point: {}".format(z[train_frame][d]))
                    print("      Training point, zero-mean: {}".format(Y[train_frame][d]))
                print("   Predicted mean at test point: {}".format(Z2[test_frame][d]))
                print("   Predicted variance at test point: {}".format(cov2use[test_frame][test_frame]))
                print("   Sampled values at test point: {}".format(yc[test_frame][d]))
                print("   Sampled values (+mean/dim): {}".format(zc[test_frame][d]))
                print("   Predicted mean (+mean/dim) at test point: {}\n".format(Z3[test_frame][d]))
            
        print("\n\n\n")

### Create the covariance matrices

covar_visualizations = False

if covar_visualizations:
    Kfull = torch.from_numpy(cov2use).unsqueeze(0)
    plt.imshow((Kfull / Kfull.max())[0].numpy(),
            cmap = parula_map)

    print(Y_dimensionwise_mean)

    N = 1080
    Xp = np.linspace(0, 1, N)[:, None]

    Kfull = torch.from_numpy(cov2use).unsqueeze(0)
    Kfull.size()

    DN=706
    Dfull = torch.from_numpy(np.expand_dims(genDistM(R, test_indices[:DN], test_indices[:DN], pose_distance2), 0)).float() 
    Kfull = Kformula(gamma, Dfull, ell)
    Kfull.size()

    from matplotlib.colors import LinearSegmentedColormap
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
    0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
    0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
    0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
    0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
    0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
    0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
    0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
    0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
    0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
    0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
    0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
    0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
    0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
    0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
    0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]]
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    plt.imshow((Ktt / Ktt.max()).numpy()[0],
            cmap = parula_map)

    ## You need to actually select each face ID first and then re-run the below code for each face.

    #face2
    plt.imshow((Kfull / Kfull.max()).numpy()[0],
            cmap = parula_map)
    labels= np.array([0, 5, 10, 15, 19])
    plt.xticks(labels, labels)
    plt.yticks(labels, labels)
    import matplotlib2tikz
    matplotlib2tikz.save(Path("kernel_face{}".format(face_id)))

    ## The below entry fails, the code should be as with the previous faces.
    ## You should ensure that in the input folders the start and end frame are correct.
    ## Face ID 4 should also be checked.
    ## It should not be multiplying the 2xN matrix with the NxN, it should be the symmetrical K matrix
    ## but calculated only over the actual range of interpolation (like in the faces 2-3 above)

    #face5
    Kfull = (Kt.bmm(K))
    plt.imshow((Kfull / Kfull.max()).numpy()[0],
            cmap = parula_map)
    labels= np.array([0, 5, 10, 15, 19])
    plt.xticks(labels, labels)
    plt.yticks(labels, labels)
    import matplotlib2tikz
    matplotlib2tikz.save(Path("pose-kernel/paper/fig/video-interpolation/kernel_face{}".format(face_id)))

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    import imageio

    imgs = np.zeros((5,1024,1024))
    k=456
    for j in range(5):
        im = imageio.imread(Path('C:/Users/arihe/git/PINE_SNG/PINE/src/PINE/face02/std_test_sigma01/0{}_{}.png'.format(k,j)))
        imgs[j] = rgb2gray(im)

    print(im.shape)
    
    #1
    plt.imshow(imgs.std(axis=0), cmap=plt.get_cmap('gray')) #, vmin=0, vmax=1)

    #256
    plt.imshow(imgs.std(axis=0), cmap=plt.get_cmap('gray'), vmin=0, vmax=200)

    #456
    plt.imshow(imgs.std(axis=0), cmap=plt.get_cmap('gray')) #, vmin=0, vmax=1)

    imageio.imwrite('C:/Users/arihe/git/PINE_SNG/PINE/src/PINE/face02/std1.png', np.round(imgs.std(axis=0)*3).astype('uint8'))

    spf = 5
    frame_interval = 5
    img_indices = np.arange(1,696+1,frame_interval)
    imgs = np.zeros((spf,1024,1024))

    root = 'C:/Users/arihe/git/PINE_SNG/PINE/src/PINE/face02/sigma_10'

    for img_i in img_indices:
        for j in range(5):
            p = Path('{}/{}_{}.png'.format(root, str(img_i).zfill(4),j))
            print(p)
            im = imageio.imread(p)
            imgs[j] = rgb2gray(im)
        op = Path('{}/std_{}.png'.format(root, str(img_i).zfill(4)))
        imageio.imwrite(op, np.round(imgs.std(axis=0)*3).astype('uint8'))
        
