import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--id',type=str)
parser.add_argument('--dir0', type=str, default='') #'./imgs/ex_dir0')
parser.add_argument('--dir1', type=str, default='') #./imgs/ex_dir1')
parser.add_argument('--ext', type=str, default='png') #./imgs/ex_dir1')
parser.add_argument('--step', type=int, default=-1) #./imgs/ex_dir1')
parser.add_argument('--save_dir', type=str, default='') #./imgs/ex_dir1')
parser.add_argument('--samename', action='store_true')
parser.add_argument('--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--start',type=int, default=-1)
parser.add_argument('--end',type=int, default=-1)
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'a')
files = os.listdir(opt.dir0)

dists = []

if opt.dir1 == '':
	opt.dir1 = opt.dir0
"""
if opt.save_dir != '':
	opt.dir1 = '{}/{}'.format(opt.dir1, opt.save_dir)
	opt.dir0 = '{}/{}'.format(opt.dir0, opt.save_dir)
if opt.step != -1:
	opt.dir1 = '{}/{}'.format(opt.dir1, opt.step)
	opt.dir0 = '{}/{}'.format(opt.dir0, opt.step)
	if opt.save_dir != '':
		opt.out = '{}_{}_res.txt'.format(opt.save_dir, opt.step)
"""
print(opt)
print('...')

# file1 index #i corresponsd to file2 index #[start + i]

extL = len(opt.ext)


for i,file in enumerate(files):
    img_id = int(file[:4]) if (file[4:4+4] == '.png' or file[4:4+5] == '.jpeg') else 0
    if not opt.samename:
        file2 = "xat{}".format(file[4:])
    else:
        if opt.end != -1:
            file2 = str(img_id - opt.start +1).zfill(4) + '.' + opt.ext
        else:
            file2 = str(img_id).zfill(4) + '.' + opt.ext

#    print('{} / {}. Try {}'.format(file, file2, os.path.join(opt.dir1,file2)))

    if(os.path.exists(os.path.join(opt.dir1,file2)) and (file[0:4] == "xate" or opt.samename)):
            if file == 'cropped' or file2 == 'cropped':
                continue

            if opt.start != -1 and opt.end != -1 and (img_id < opt.start or img_id > opt.end):
                continue

    # Load images
            img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
            img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file2)))

            # Compute distance
            dist01 = model.forward(img0,img1)

            if i%10 == 0:
                print('{0} / {1} = {2}'.format(file, file2,dist01))

            if i%50 == 0:
                if i > 2:
                    print('%s / %s: %.9f --- Current MEAN * 1e8: %.3f and std: %.3f'%(file,file2,dist01,np.mean(dists)*1e8, np.std(dists)*1e8))
            id = file.split('.')[0]
            f.writelines('%s,%s,%s,%s,%.9f\n'%(opt.id, id, file,file2,dist01))

            dists = dists + [dist01]

print('{} distances received'.format(len(dists)))

logf = open('persim.log','a')
logf.write('{},{},{}\n'.format(opt.id, np.mean(dists), np.std(dists)))
logf.close()
print("MEAN: {} and std: {}".format(np.mean(dists), np.std(dists)))
print("N = {}".format(len(dists)))

f.close()
