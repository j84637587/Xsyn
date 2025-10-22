#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53489  gligen_inference.py --no_plms

# downstream test

# gen1_1%_filter_full_scar_single_top1_multi_range1
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_convex"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_range1_convex"

# gen3_0.1%_filter_full_scar_range1
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_full_scar_range1"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen3_0.1%_filter_full_scar_range1"

# # gen1_1%_filter_full_scar_range1_nonzero
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_nonzero"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_range1_nonzero"

# gen1_1%_filter_full_scar_range1_ablation
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_ablation"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_range1_ablation"

# gen1_3_1%_filter_best
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_1%_filter_best"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_1%_filter_best"

# # gen3_1%_filter_best
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_1%_filter_best"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen3_1%_filter_best"


# bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

# # gen1_3_0.1%_filter_best
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best"

# gen1_3_0.1%_filter_best_newer
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_newer"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_newer"


#bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

# source /home/ct/sjl/anaconda3/bin/activate /home/ct/sjl/anaconda3/envs/openmmlab
# cd $DET_PATH
# bash sjl_train.sh

# bash sample.sh
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53489  gligen_inference.py --no_plms

# gen1_1%_3_0.1%_filter_best_newer
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_3_0.1%_filter_best_newer"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_3_0.1%_filter_best_newer"

#gen1_3_0.1%_filter_best_30%_prob
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_30%_prob"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_30%_prob"

# bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53489  gligen_inference.py --no_plms

#gen1_3_0.1%_filter_best_20%_prob
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_20%_prob"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_20%_prob"

# gen1_3_0.5%_filter_best_hard
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen1_3_0.5%_filter_best_hard"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.5%_filter_best_hard"

# gen3_0.1%_filter_best_ablation
GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_0.1%_filter_best_ablation"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen3_0.1%_filter_best_ablation"


#bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

file0="0_annotation.txt"
file1="1_annotation.txt"
file2="2_annotation.txt"
file3="3_annotation.txt"
# file4="4_annotation.txt"
# file5="5_annotation.txt"
# file6="6_annotation.txt"
# file7="7_annotation.txt"

# w_o_scar
# ANNO_PATH="$GEN_DATA_PATH/annotation"
img_path="$GEN_DATA_PATH/data/ori"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

# cd /home/ct/sjl/research/GLIGEN

# range1
# ANNO_PATH="$GEN_DATA_PATH/annotation/range1"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

# cd /home/ct/sjl/research/GLIGEN

# # range2
# ANNO_PATH="$GEN_DATA_PATH/annotation/range2"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

# cd /home/ct/sjl/research/GLIGEN

# # range3
# ANNO_PATH="$GEN_DATA_PATH/annotation/range3"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

# cd /home/ct/sjl/research/GLIGEN

# range4
ANNO_PATH="$GEN_DATA_PATH/annotation/range4"
bash merge_anno.sh "$file0" "$file1" "$file2" "$file3"  "$ANNO_PATH"
echo "annotation.txt done."

cd $MAKE_ANNO_PATH 

annotation_file="$ANNO_PATH/annotation.txt"
output_json="$ANNO_PATH/anno.json"

python gen_anno.py \
    --annotation_file "$annotation_file" \
    --img_path "$img_path" \
    --output_json "$output_json"

echo "anno.json done."

cd /home/ct/sjl/research/GLIGEN

# # # only_box
# # ANNO_PATH="$GEN_DATA_PATH/annotation/only_box"
# # bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# # echo "annotation.txt done."

# # cd $MAKE_ANNO_PATH 

# # annotation_file="$ANNO_PATH/annotation.txt"
# # output_json="$ANNO_PATH/anno.json"

# # python gen_anno.py \
# #     --annotation_file "$annotation_file" \
# #     --img_path "$img_path" \
# #     --output_json "$output_json"

# # echo "anno.json done."

# # cd /home/ct/sjl/research/GLIGEN

# # # mode
# # ANNO_PATH="$GEN_DATA_PATH/annotation/mode"
# # bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# # echo "annotation.txt done."

# # cd $MAKE_ANNO_PATH 

# # annotation_file="$ANNO_PATH/annotation.txt"
# # output_json="$ANNO_PATH/anno.json"

# # python gen_anno.py \
# #     --annotation_file "$annotation_file" \
# #     --img_path "$img_path" \
# #     --output_json "$output_json"

# # echo "anno.json done."

# # cd /home/ct/sjl/research/GLIGEN

# # topk
# ANNO_PATH="$GEN_DATA_PATH/annotation/top16"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

# cd /home/ct/sjl/research/GLIGEN

# sam
# ANNO_PATH="$GEN_DATA_PATH/annotation/sam"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3" "$file4" "$file5" "$file6" "$file7" "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."
