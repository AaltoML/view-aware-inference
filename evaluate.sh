#!/bin/bash

# 1. Load frames.csv, convert into matrices for GP operations, and execute the interpolations you want. [here]

srcroot=`pwd`
gpdataroot=$1

for kernel in viewaware euler quat;
do
 python eval.py --data_path $gpdataroot/data  --face_id=2  --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=3  --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=4  --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=5  --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=2 --full_smoothing  --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=3 --full_smoothing --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=5 --full_smoothing --kernel_mode=$kernel
 python eval.py --data_path $gpdataroot/data  --face_id=4 --full_smoothing  --kernel_mode=$kernel;
done;

# 2. Linear Baselines

#For the baseline separable (Euler) kernels and the quaternion kernels, run with --kernel_mode=[quat|euler], e.g.:
#python eval.py --data_path ./data  --face_id=5 --kernel_mode quat

mkdir $gpdataroot/data/face02/linear
mkdir $gpdataroot/data/face03/linear
mkdir $gpdataroot/data/face04/linear
mkdir $gpdataroot/data/face05/linear

cd stylegan

python project_z.py --z_input_path $gpdataroot/data/face02 --img_out_path $gpdataroot/data/face02/linear --z_interpolate_range=605,5,976
python project_z.py --z_input_path $gpdataroot/data/face03 --img_out_path $gpdataroot/data/face03/linear --z_interpolate_range=725,5,1076
python project_z.py --z_input_path $gpdataroot/data/face04 --img_out_path $gpdataroot/data/face04/linear --z_interpolate_range=150,5,506 ###1080 ?
python project_z.py --z_input_path $gpdataroot/data/face05 --img_out_path $gpdataroot/data/face05/linear --z_interpolate_range=400,5,806

# 3. Decode the Z array with StyleGAN to x^, and crop:

for f in 2 3 4 5;
do
    mkdir -p $gpdataroot/data/face0$f/full_interpolation
    mkdir -p $gpdataroot/data/face0$f/first_last_interpolation
    mkdir -p $gpdataroot/data/face0$f/euler_smoothing
    mkdir -p $gpdataroot/data/face0$f/euler_interpolate
    mkdir -p $gpdataroot/data/face0$f/quat_smoothing
    mkdir -p $gpdataroot/data/face0$f/quat_interpolate

    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan_full_smoothing.z2.npy --img_out_path $gpdataroot/data/face0$f/full_interpolation
    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan.z2.npy --img_out_path $gpdataroot/data/face0$f/first_last_interpolation
    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan_full_smoothing_euler.z2.npy --img_out_path $gpdataroot/data/face0$f/euler_smoothing
    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan_euler.z2.npy --img_out_path $gpdataroot/data/face0$f/euler_interpolate
    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan_full_smoothing_quat.z2.npy --img_out_path $gpdataroot/data/face0$f/quat_smoothing
    python project_z.py --z_input_file $gpdataroot/data/face0$f/stylegan_quat.z2.npy --img_out_path $gpdataroot/data/face0$f/quat_interpolate;
done

cd ..

# All c

for f in 2 3 4 5;
do
    for p in full_interpolation first_last_interpolation stylegan quat_smoothing quat_interpolate euler_smoothing euler_interpolate linear;
    do
        echo $gpdataroot/data/face0$f/$p
        cd $gpdataroot/data/face0$f/$p 
        mkdir cropped
        for i in *.png; do convert $i -crop 512x512+256+384 cropped/$i; done
        for i in *.jpeg; do convert $i -crop 512x512+256+384 cropped/$i; done
        cd ../../..;
    done
done

# Create LPIPS measurements

cd ${srcroot}/PerceptualSimilarity

rm lpips_inter lpips_base_inter  lpips_intra  lpips_base_intra

python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face02 --dir1=${gpdataroot}/data/face02/first_last_interpolation/cropped    --out=lpips_inter --samename  --id=gpint_2 --start=606 --end=976
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face02 --dir1=${gpdataroot}/data/face02/full_interpolation/cropped    --out=lpips_inter --samename --id=smooth2 --start=606 --end=976
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face02 --dir1=${gpdataroot}/data/face02/stylegan    --out=lpips_inter --samename --id=reco2
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face02 --dir1=${gpdataroot}/data/face02/linear/cropped  --out=lpips_base_inter --samename --id=lin_2

python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face03 --dir1=${gpdataroot}/data/face03/first_last_interpolation/cropped    --out=lpips_inter --samename  --id=gpint_3 --start=726 --end=1076 
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face03 --dir1=${gpdataroot}/data/face03/full_interpolation/cropped    --out=lpips_inter --samename --id=smooth3 --start=726 --end=1076 
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face03 --dir1=${gpdataroot}/data/face03/stylegan    --out=lpips_inter --samename --id=reco3
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face03 --dir1=${gpdataroot}/data/face03/linear/cropped  --out=lpips_base_inter --samename --id=lin_3


python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face04 --dir1=${gpdataroot}/data/face04/first_last_interpolation/cropped    --out=lpips_inter --samename  --id=gpint_4 --start=151 --end=506  
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face04 --dir1=${gpdataroot}/data/face04/full_interpolation/cropped    --out=lpips_inter --samename --id=smooth4 --start=151 --end=506  
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face04 --dir1=${gpdataroot}/data/face04/stylegan    --out=lpips_inter --samename --id=reco4
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face04 --dir1=${gpdataroot}/data/face04/linear/cropped  --out=lpips_base_inter --samename --id=lin_4

python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face05 --dir1=${gpdataroot}/data/face05/first_last_interpolation/cropped    --out=lpips_inter --samename  --id=gpint_5 --start=401 --end=806 
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face05 --dir1=${gpdataroot}/data/face05/full_interpolation/cropped    --out=lpips_inter --samename --id=smooth5 --start=401 --end=806 
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face05 --dir1=${gpdataroot}/data/face05/stylegan    --out=lpips_inter --samename --id=reco5
python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face05 --dir1=${gpdataroot}/data/face05/linear/cropped  --out=lpips_base_inter --samename --id=lin_5

for p in quat_smoothing quat_interpolate euler_smoothing euler_interpolate;
do
 python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face02 --dir1=${gpdataroot}/data/face02/$p/cropped   --id=${p}_2 --out=lpips_base_inter --samename --start=606 --end=976 
 python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face03 --dir1=${gpdataroot}/data/face03/$p/cropped   --id=${p}_3 --out=lpips_base_inter --samename --start=726 --end=1076
 python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face04 --dir1=${gpdataroot}/data/face04/$p/cropped   --id=${p}_4 --out=lpips_base_inter --samename --start=151 --end=506 
 python ./compute_dists_dirs_comp.py --dir0=${gpdataroot}/data/face05 --dir1=${gpdataroot}/data/face05/$p/cropped   --id=${p}_5 --out=lpips_base_inter --samename --start=401 --end=806 
done

sed -i '1iid,foo1,foo2,foo3,val' lpips_inter
sed -i '1iid,foo1,foo2,foo3,val' lpips_base_inter

#LPIPS-delta

python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face02/first_last_interpolation/cropped  --id=gpint_2 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face02/full_interpolation/cropped   --id=smooth_2 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face02/stylegan   --id=reco_2 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face02/linear/cropped   --id=lin_2 --out=lpips_intra --samename

python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face03/first_last_interpolation/cropped   --id=gpint_3 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face03/full_interpolation/cropped   --id=smooth_3 --out=lpips_intra --samename 
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face03/stylegan   --id=reco_3 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face03/linear/cropped   --id=lin_3 --out=lpips_intra --samename

python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face04/first_last_interpolation/cropped   --id=gpint_4 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face04/full_interpolation/cropped   --id=smooth_4 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face04/stylegan   --id=reco_4 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face04/linear/cropped   --id=lin_4 --out=lpips_intra --samename

python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face05/first_last_interpolation/cropped   --id=gpint_5 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face05/full_interpolation/cropped   --id=smooth_5 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face05/stylegan   --id=reco_5 --out=lpips_intra --samename
python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face05/linear/cropped   --id=lin_5 --out=lpips_intra --samename

for f in 2 3 4 5;
do
 python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face0$f/quat_smoothing/cropped   --id=quat_smooth_$f --out=lpips_base_intra --samename
 python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face0$f/quat_interpolate/cropped   --id=quat_int_$f --out=lpips_base_intra --samename
 python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face0$f/euler_smoothing/cropped   --id=euler_smooth_$f --out=lpips_base_intra --samename
 python ./compute_dists_seq.py --dir0=${gpdataroot}/data/face0$f/euler_interpolate/cropped   --id=euler_int_$f --out=lpips_base_intra --samename;
done

sed -i '1iid,foo1,foo2,foo3,val' lpips_intra
sed -i '1iid,foo1,foo2,foo3,val' lpips_base_intra

cd ..

python lpips_metrics.py

