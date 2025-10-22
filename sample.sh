#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53479  gligen_inference.py --no_plms

# downstream test

# gen1_1%_filter_full_scar_single_top1_multi_range1
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_single_top1_multi_range1"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_single_top1_multi_range1"

# gen2_1%_filter_full_scar_range1
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen2_1%_filter_full_scar_range1"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen2_1%_filter_full_scar_range1"

# # gen1_1%_filter_full_scar_range1_rand_fg_hidden
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_rand_fg_hidden"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_range1_rand_fg_hidden"

# # gen3_0.1%_filter_full_scar_range1_same_bg_hidden_best
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_full_scar_range1_same_bg_hidden_best"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen3_0.1%_filter_full_scar_range1_same_bg_hidden_best"

# gen3_0.1%_filter_full_scar_best_hidden_newer
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_full_scar_best_hidden_newer"
#DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen3_0.1%_filter_full_scar_best_hidden_newer"

# bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

# source /home/ct/sjl/anaconda3/bin/activate /home/ct/sjl/anaconda3/envs/openmmlab
# cd $DET_PATH
# bash sjl_test.sh
# bash sjl_train.sh

# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_100000/gen1_0.4%_filter_best_no_scar"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation.txt"
# file1="1_annotation.txt"
# file2="2_annotation.txt"
# file3="3_annotation.txt"
# exp_name="gen1_0.4%_filter_best_no_scar"

# bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_100000/gen3_0.4%_filter_best_no_scar"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation.txt"
# file1="1_annotation.txt"
# file2="2_annotation.txt"
# file3="3_annotation.txt"
# exp_name="gen3_0.4%_filter_best_no_scar"

# bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

# gen1_1%_filter_full_scar_ablation_sam
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_ablation_sam"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_1%_filter_full_scar_ablation_sam"

# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_100000/gen1_3_0.4%_filter_best"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.4%_filter_best"

# gen1_3_0.4%_filter_best_50%_prob
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_100000/gen1_3_0.4%_filter_best_50%_prob"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.4%_filter_best_50%_prob"

#gen1_3_0.1%_filter_best_10%_prob
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_10%_prob"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_10%_prob"

#bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"

#CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port 53479  gligen_inference.py --no_plms

# #gen1_3_0.1%_filter_best_100%_prob_hard
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_100%_prob_hard"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_100%_prob_hard"


# # gen1_3_0.1%_filter_best_group
# GEN_DATA_PATH="/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_3_0.1%_filter_best_group"
# DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
# MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
# file0="0_annotation_refine.txt"
# file1="1_annotation_refine.txt"
# file2="2_annotation_refine.txt"
# file3="3_annotation_refine.txt"
# exp_name="gen1_3_0.1%_filter_best_group"

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53479  gligen_inference.py --no_plms

GEN_DATA_PATH="/home/ct/data/sjl/gligen_official/generated_images/text_box_180000/gligen_gen1/"
DET_PATH="/home/ct/data/lwz/workplace/xray_detection/mmdetection"
MAKE_ANNO_PATH="/home/ct/sjl/research/diffusers/examples/research_projects/gligen/data"
file0="0_annotation.txt"
file1="1_annotation.txt"
file2="2_annotation.txt"
file3="3_annotation.txt"

# range0
# ANNO_PATH="$GEN_DATA_PATH/annotation"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3"  "$ANNO_PATH"
# echo "annotation.txt done."

# cd $MAKE_ANNO_PATH 

# annotation_file="$ANNO_PATH/annotation.txt"
# output_json="$ANNO_PATH/anno.json"
# img_path="$GEN_DATA_PATH/data/ori"

# python gen_anno.py \
#     --annotation_file "$annotation_file" \
#     --img_path "$img_path" \
#     --output_json "$output_json"

# echo "anno.json done."

#cd /home/ct/sjl/research/GLIGEN

# # range1
# ANNO_PATH="$GEN_DATA_PATH/annotation/range1"
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3"  "$ANNO_PATH"
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
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3"  "$ANNO_PATH"
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
# bash merge_anno.sh "$file0" "$file1" "$file2" "$file3"  "$ANNO_PATH"
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
img_path="$GEN_DATA_PATH/data/ori"

python gen_anno.py \
    --annotation_file "$annotation_file" \
    --img_path "$img_path" \
    --output_json "$output_json"

echo "anno.json done."

#bash downstream_test.sh "$GEN_DATA_PATH" "$DET_PATH" "$MAKE_ANNO_PATH" "$file0" "$file1" "$file2" "$file3" "$exp_name"
