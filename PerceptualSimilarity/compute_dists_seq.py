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
parser.add_argument('--step', type=int, default=-1) #./imgs/ex_dir1')
parser.add_argument('--save_dir', type=str, default='') #./imgs/ex_dir1')
parser.add_argument('--samename', action='store_true')
parser.add_argument('--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--start',type=int, default=-1)
parser.add_argument('--end',type=int, default=-1)
parser.add_argument('--frame_delta',type=int, default=5)
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

if True:
    for i,file in enumerate(files):
         if file[:4].isdigit():
                file2 = format(int(file[0:4]) + opt.frame_delta, '04d') + file[4:]
                if file == 'cropped' or file2 == 'cropped':
                    continue

                if not os.path.exists(os.path.join(opt.dir0,file2)):
                    continue

                img_id = int(file[:4])
                if opt.start != -1 and opt.end != -1 and (img_id < opt.start or img_id > opt.end):
                    continue

#                if i%10 == 0:
#                    print('{} / {}'.format(file, file2))

		# Load images
                img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
                img1 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file2)))

                # Compute distance
                dist01 = model.forward(img0,img1)
                if i%50 == 0:
                    if i > 2:
                        print('%s / %s: %.9f --- Current MEAN * 1e8: %.3f and std: %.3f'%(file,file2,dist01,np.mean(dists)*1e8, np.std(dists)*1e8))
                id = file.split('.')[0]
                f.writelines('%s,%s,%s,%s,%.9f\n'%(opt.id, id, file,file2,dist01))

                dists = dists + [dist01]

logf = open('persim.log','a')
logf.write('{},{},{}\n'.format(opt.id, np.mean(dists), np.std(dists)))
logf.close()
print("MEAN: {} and std: {}".format(np.mean(dists), np.std(dists)))
print("N = {}".format(len(dists)))

f.close()
