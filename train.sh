# no_inpaint
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53449 main.py  --yaml_file configs/pidray_box_text.yaml  --name text_box --OUTPUT_ROOT /home/ct/data/sjl/gligen_official_opiray --inpaint_mode=False   --batch_size=2

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53459 main.py  --yaml_file configs/pidray_box_text.yaml  --name text_box --OUTPUT_ROOT /home/ct/data/sjl/gligen_official_hixray_inpaint --inpaint_mode=True   --batch_size=2 --ckpt /home/ct/data/sjl/gligen_official_hixray/text_box/tag00/checkpoint_00180000_inpaint_start.pth

#inpaint
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53449 main.py  --yaml_file configs/pidray_box_text.yaml  --name text_box_180000 --OUTPUT_ROOT /home/ct/data/sjl/gligen_official_inpaint  --inpaint_mode=True  --batch_size=2

# depth 
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53449 main.py  --yaml_file configs/pidray_box_text_depth.yaml  --name depth_exp1 --OUTPUT_ROOT /home/ct/data/sjl/gligen_official --inpaint_mode=False   --batch_size=2

#depth inpaint
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port 53449 main.py  --yaml_file configs/pidray_box_text_depth.yaml  --name depth_inpaint_exp1 --OUTPUT_ROOT /home/ct/data/sjl/gligen_official_inpaint --inpaint_mode=True   --batch_size=2 --ckpt /home/ct/data/sjl/gligen_official/depth_exp1/tag00/checkpoint_00080000_inpaint_start.pth