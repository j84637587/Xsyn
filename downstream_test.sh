#!/bin/bash

GEN_DATA_PATH=$1 
DET_PATH=$2
MAKE_ANNO_PATH=$3
file0=$4
file1=$5
file2=$6
file3=$7
exp_name=$8

ANNO_PATH="$GEN_DATA_PATH/annotation"
DET_CONFIG_PATH="$DET_PATH/configs"
annotation_file="$ANNO_PATH/annotation_refine.txt"
img_path="$GEN_DATA_PATH/data/"
output_json="$ANNO_PATH/anno_refine.json"

#pidray
#input_config="$DET_CONFIG_PATH/00_xray_pidray/sjl_dino-4scale_r50_8xb2-12e_pidray.py"

#opixray
input_config="$DET_CONFIG_PATH/00_xray_pidray/sjl_dino-4scale_r50_8xb2-12e_opixray.py" 

GLIGEN_PATH="/home/ct/sjl/research/GLIGEN"

# 1. 构建标注文件


bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$ANNO_PATH"
echo "annotation_refine.txt done."

cd $MAKE_ANNO_PATH 

python gen_anno.py \
    --annotation_file "$annotation_file" \
    --img_path "$img_path" \
    --output_json "$output_json"

echo "anno_refine.json done."

# 2. 激活检测环境
source /home/ct/sjl/anaconda3/bin/activate /home/ct/sjl/anaconda3/envs/openmmlab

# 3.1 创建检测文件
cd $GLIGEN_PATH
bash create_config.sh "$input_config" "$GEN_DATA_PATH" "$exp_name"
echo "config file created."

new_config="${input_config%.py}_$exp_name.py"

# 3.2 执行检测程序
cd $DET_PATH

echo "start detection training..."
PORT=28502 CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh $new_config 4