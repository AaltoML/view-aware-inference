import pandas as pd
import numpy as np
import sys

f = int(sys.argv[1])
print(f'Face {f} only')

data = pd.read_csv('PerceptualSimilarity/lpips_inter')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains('reco')]['val'].mean()
v2 = df2[df2['id'].str.contains('reco')]['val'].std()
v3 = df2[df2['id'].str.contains('reco')]['val'].median()

print("1-by-1 StyleGAN projections: mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains(f'smooth{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'smooth{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'smooth{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'smooth{f}')]['val'].count()

print("GP interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'int_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'int_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'int_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'int_{f}')]['val'].count()

print("GP interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))


data = pd.read_csv('PerceptualSimilarity/lpips_base_inter')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains(f'quat_smoothing_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'quat_smoothing_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'quat_smoothing_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'quat_smoothing_{f}')]['val'].count()

print("Quat interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'quat_interpolate_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'quat_interpolate_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'quat_interpolate_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'quat_interpolate_{f}')]['val'].count()

print("Quat interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'euler_smoothing_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'euler_smoothing_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'euler_smoothing_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'euler_smoothing_{f}')]['val'].count()

print("Euler interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'euler_interpolate_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'euler_interpolate_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'euler_interpolate_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'euler_interpolate_{f}')]['val'].count()

print("Euler interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))


#data = pd.read_csv('PerceptualSimilarity/lpips_lin')
#df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains(f'lin_{f}')][f'val'].mean()
v2 = df2[df2['id'].str.contains(f'lin_{f}')][f'val'].std()
v3 = df2[df2['id'].str.contains(f'lin_{f}')][f'val'].median()

print("Linear interpolation: mean = {} std = {} median = {}".format(v1,v2,v3))


data = pd.read_csv('PerceptualSimilarity/lpips_intra')
df3 = pd.DataFrame(data)

v1 = df3[df3['id'].str.contains(f'reco_{f}')][f'val'].mean()
v2 = df3[df3['id'].str.contains(f'reco_{f}')][f'val'].std()
v3 = df3[df3['id'].str.contains(f'reco_{f}')][f'val'].median()
v4 = df3[df3['id'].str.contains(f'reco_{f}')][f'val'].count()

print("LPIPS-delta: 1-by-1 StyleGAN projections: mean = {} std = {} median = {}".format(v1,v2,v3,v4))


v1 = df3[df3['id'].str.contains(f'smooth_{f}')][f'val'].mean()
v2 = df3[df3['id'].str.contains(f'smooth_{f}')][f'val'].std()
v3 = df3[df3['id'].str.contains(f'smooth_{f}')][f'val'].median()
v4 = df3[df3['id'].str.contains(f'smooth_{f}')][f'val'].count()

print("LPIPS-delta: GP interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df3[df3['id'].str.contains(f'int_{f}')][f'val'].mean()
v2 = df3[df3['id'].str.contains(f'int_{f}')][f'val'].std()
v3 = df3[df3['id'].str.contains(f'int_{f}')][f'val'].median()
v4 = df3[df3['id'].str.contains(f'int_{f}')][f'val'].count()

print("LPIPS-delta: GP interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))


v1 = df3[df3['id'].str.contains(f'lin_{f}')]['val'].mean()
v2 = df3[df3['id'].str.contains(f'lin_{f}')]['val'].std()
v3 = df3[df3['id'].str.contains(f'lin_{f}')]['val'].median()
v4 = df3[df3['id'].str.contains(f'lin_{f}')]['val'].count()

print("LPIPS-delta: Linear interpolation: mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))




data = pd.read_csv('PerceptualSimilarity/lpips_base_intra')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains(f'quat_smooth_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'quat_smooth_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'quat_smooth_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'quat_smooth_{f}')]['val'].count()

print("LPIPS-delta: Quat interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'quat_int_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'quat_int_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'quat_int_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'quat_int_{f}')]['val'].count()

print("LPIPS-delta: Quat interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'euler_smooth_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'euler_smooth_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'euler_smooth_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'euler_smooth_{f}')]['val'].count()

print("LPIPS-delta: Euler interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains(f'euler_int_{f}')]['val'].mean()
v2 = df2[df2['id'].str.contains(f'euler_int_{f}')]['val'].std()
v3 = df2[df2['id'].str.contains(f'euler_int_{f}')]['val'].median()
v4 = df2[df2['id'].str.contains(f'euler_int_{f}')]['val'].count()

print("LPIPS-delta: Euler interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

