import pandas as pd
import numpy as np

data = pd.read_csv('PerceptualSimilarity/lpips_inter')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains('reco')]['val'].mean()
v2 = df2[df2['id'].str.contains('reco')]['val'].std()
v3 = df2[df2['id'].str.contains('reco')]['val'].median()

print("1-by-1 StyleGAN projections: mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains('smooth')]['val'].mean()
v2 = df2[df2['id'].str.contains('smooth')]['val'].std()
v3 = df2[df2['id'].str.contains('smooth')]['val'].median()

print("GP interpolation (all frames): mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains('int_')]['val'].mean()
v2 = df2[df2['id'].str.contains('int_')]['val'].std()
v3 = df2[df2['id'].str.contains('int_')]['val'].median()

print("GP interpolation (first-last only): mean = {} std = {} median = {}".format(v1,v2,v3))


data = pd.read_csv('PerceptualSimilarity/lpips_base_inter')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains('quat_smooth')]['val'].mean()
v2 = df2[df2['id'].str.contains('quat_smooth')]['val'].std()
v3 = df2[df2['id'].str.contains('quat_smooth')]['val'].median()
v4 = df2[df2['id'].str.contains('quat_smooth')]['val'].count()

print("Quat interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains('quat_int')]['val'].mean()
v2 = df2[df2['id'].str.contains('quat_int')]['val'].std()
v3 = df2[df2['id'].str.contains('quat_int')]['val'].median()
v4 = df2[df2['id'].str.contains('quat_int')]['val'].count()


print("Quat interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains('euler_smooth')]['val'].mean()
v2 = df2[df2['id'].str.contains('euler_smooth')]['val'].std()
v3 = df2[df2['id'].str.contains('euler_smooth')]['val'].median()
v4 = df2[df2['id'].str.contains('euler_smooth')]['val'].count()

print("Euler interpolation (all frames): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))

v1 = df2[df2['id'].str.contains('euler_int')]['val'].mean()
v2 = df2[df2['id'].str.contains('euler_int')]['val'].std()
v3 = df2[df2['id'].str.contains('euler_int')]['val'].median()
v4 = df2[df2['id'].str.contains('euler_int')]['val'].count()

print("Euler interpolation (first-last only): mean = {} std = {} median = {} count = {}".format(v1,v2,v3,v4))


v1 = df2[df2['id'].str.contains('lin')]['val'].mean()
v2 = df2[df2['id'].str.contains('lin')]['val'].std()
v3 = df2[df2['id'].str.contains('lin')]['val'].median()

print("Linear interpolation: mean = {} std = {} median = {}".format(v1,v2,v3))


data = pd.read_csv('PerceptualSimilarity/lpips_intra')
df3 = pd.DataFrame(data)

v1 = df3[df3['id'].str.contains('reco')]['val'].mean()
v2 = df3[df3['id'].str.contains('reco')]['val'].std()
v3 = df3[df3['id'].str.contains('reco')]['val'].median()

print("LPIPS-delta: 1-by-1 StyleGAN projections: mean = {} std = {} median = {}".format(v1,v2,v3))


v1 = df3[df3['id'].str.contains('smooth')]['val'].mean()
v2 = df3[df3['id'].str.contains('smooth')]['val'].std()
v3 = df3[df3['id'].str.contains('smooth')]['val'].median()

print("LPIPS-delta: GP interpolation (all frames): mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df3[df3['id'].str.contains('int_')]['val'].mean()
v2 = df3[df3['id'].str.contains('int_')]['val'].std()
v3 = df3[df3['id'].str.contains('int_')]['val'].median()

print("LPIPS-delta: GP interpolation (first-last only): mean = {} std = {} median = {}".format(v1,v2,v3))


v1 = df3[df3['id'].str.contains('lin')]['val'].mean()
v2 = df3[df3['id'].str.contains('lin')]['val'].std()
v3 = df3[df3['id'].str.contains('lin')]['val'].median()

print("LPIPS-delta: Linear interpolation: mean = {} std = {} median = {}".format(v1,v2,v3))



data = pd.read_csv('PerceptualSimilarity/lpips_base_intra')
df2 = pd.DataFrame(data)

v1 = df2[df2['id'].str.contains('quat_smooth')]['val'].mean()
v2 = df2[df2['id'].str.contains('quat_smooth')]['val'].std()
v3 = df2[df2['id'].str.contains('quat_smooth')]['val'].median()

print("LPIPS-delta: Quat interpolation (all frames): mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains('quat_int')]['val'].mean()
v2 = df2[df2['id'].str.contains('quat_int')]['val'].std()
v3 = df2[df2['id'].str.contains('quat_int')]['val'].median()

print("LPIPS-delta: Quat interpolation (first-last only): mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains('euler_smooth')]['val'].mean()
v2 = df2[df2['id'].str.contains('euler_smooth')]['val'].std()
v3 = df2[df2['id'].str.contains('euler_smooth')]['val'].median()

print("LPIPS-delta: Euler interpolation (all frames): mean = {} std = {} median = {}".format(v1,v2,v3))

v1 = df2[df2['id'].str.contains('euler_int')]['val'].mean()
v2 = df2[df2['id'].str.contains('euler_int')]['val'].std()
v3 = df2[df2['id'].str.contains('euler_int')]['val'].median()

print("LPIPS-delta: Euler interpolation (first-last only): mean = {} std = {} median = {}".format(v1,v2,v3))

