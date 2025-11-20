import os
import json
from PIL import Image
import numpy as np
import torch
import argparse
import pdb
from pycocotools.coco import COCO

def create_coco_from_annotation(annotation_file, img_path, output_json):
    # Initialize counters for image and annotation IDs
    image_id = 0
    annotation_id = 0
    
    # 创建一个新的字典来存储筛选后的标注
    filtered_annotations = {
        "images": [],
        "annotations": [],
        "categories": categories  # 假设类别信息不需要筛选
    }
    
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            filename = line[0]
            img = Image.open(os.path.join(img_path, filename))
            width, height = img.size
            filtered_annotations["images"].append(
                {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id
                }
            )
            label = np.array([float(x) for x in line[1:] if x != '']).reshape(-1, 5)
            for single_label in label:
                category_id = int(single_label[0])
                bbox = list(single_label[1:])
                x1, y1, w, h = bbox
                area = w * h
                filtered_annotations["annotations"].append(
                    {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id, 
                    "bbox": [x1, y1, w, h], 
                    'area': area,
                    'segmentation': None,
                    "iscrowd": 0
                    }
                )
                annotation_id += 1
            image_id += 1
    
    # 保存新的标注文件
    with open(output_json, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)

def create_coco_json_filter(original_json, filter_file, output_json):
    # Initialize counters for image and annotation IDs
    image_id = 0
    annotation_id = 0
    filter_filenames = []
    
    # with open(filter_file, 'r') as f:
    #     for line in f.readlines():
    #         filename = line.strip('\n')
    #         filter_filenames.append(filename)
    filter_gligen_data = torch.load(filter_file)
    filter_filenames = [data["file_path"] for data in filter_gligen_data]
    
    with open(original_json, 'r') as f:
        original_data = json.load(f)
    
    # 创建一个新的字典来存储筛选后的标注
    filtered_annotations = {
        "images": [],
        "annotations": [],
        "categories": original_data['categories']  # 假设类别信息不需要筛选
    }

    # 筛选图像和标注，并重新分配id
    for image in original_data['images']:
        if image['file_name'] in filter_filenames:
            # 复制图像信息并更新id
            new_image = image.copy()
            new_image['id'] = image_id
            filtered_annotations['images'].append(new_image)
            
            # 筛选与该图像相关的标注，并更新annotation的id和image_id
            for anno in original_data['annotations']:
                if anno['image_id'] == image['id']:
                    # 复制标注信息并更新id和image_id
                    new_anno = anno.copy()
                    new_anno['id'] = annotation_id
                    new_anno['image_id'] = new_image['id']
                    filtered_annotations['annotations'].append(new_anno)
                    annotation_id += 1
            
            image_id += 1

    # 保存新的标注文件
    with open(output_json, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)

def merge_coco_annotations(file1, file2, output_file):
    with open(file1, 'r') as f1:
        coco1 = json.load(f1)
    with open(file2, 'r') as f2:
        coco2 = json.load(f2)

    merged = {
        "images": [],
        "annotations": [],
        "categories": coco1["categories"]  # 假设 categories 相同
    }

    # 拿到 ID 的最大值
    max_image_id = max(img["id"] for img in coco1["images"])
    max_ann_id = max(ann["id"] for ann in coco1["annotations"])

    # 拷贝第一个数据集
    merged["images"].extend(coco1["images"])
    merged["annotations"].extend(coco1["annotations"])

    # 映射旧的 image_id -> 新的 image_id
    image_id_map = {}
    for img in coco2["images"]:
        old_id = img["id"]
        max_image_id += 1
        img["id"] = max_image_id
        image_id_map[old_id] = img["id"]
        merged["images"].append(img)

    for ann in coco2["annotations"]:
        max_ann_id += 1
        ann["id"] = max_ann_id
        ann["image_id"] = image_id_map[ann["image_id"]]  # 更新 image_id
        merged["annotations"].append(ann)

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=4)
    
    print(f"✅ 合并完成，输出文件: {output_file}")

def shift_category_ids(anno_path, output_path, categories, dataset='opixray'):
    # 读取原始标注文件
    with open(anno_path, 'r') as f:
        coco_data = json.load(f)
    
    id_mapping = {}
    '''
    # 创建新旧ID映射字典
    if dataset == 'opixray':
        # opixray
        # id_mapping = {cat['id']: cat['id'] + shift for cat in coco_data['categories'] if cat['name'] != 'Scissor'}
        # id_mapping[1] = 1 
        # id_mapping[2] = 1 
        # id_mapping[3] = 1 
        # id_mapping[4] = 1 
        # id_mapping[5] = 2
        id_mapping[1] = 11 
        id_mapping[2] = 11
        id_mapping[3] = 11 
        id_mapping[4] = 11
        id_mapping[5] = 5
    
    elif dataset == 'hixray':

        # hixray
        #id_mapping = {cat['id']: cat['id'] + shift for cat in coco_data['categories'] }
        #pdb.set_trace()
        id_mapping[1] = 3 
        id_mapping[2] = 4
        id_mapping[3] = 5 
        id_mapping[4] = 6 
        id_mapping[5] = 7
        id_mapping[6] = 8
        id_mapping[7] = 9
    '''
    # 更新categories部分
    coco_data['categories'] = categories

    # 更新annotations部分
    '''
    if dataset != 'pidray':
        for ann in coco_data['annotations']:
            ann['category_id'] = id_mapping[ann['category_id']]
    '''

    # 保存修改后的标注文件
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print('done')

def filter_category(input_json, output_json, target_category_ids):
    '''
    input_json: 原始 COCO JSON 文件
    output_json: 筛选后输出的新 JSON 文件
    target_category_ids: 想保留的类别 ID, [id1, id2,...]
    '''

    # 读取原始文件
    with open(input_json, 'r') as f:
        coco = json.load(f)

    # 筛选 annotation 中类别符合条件的
    filtered_annotations = [ann for ann in coco['annotations'] if ann['category_id'] in target_category_ids]

    # 提取保留 annotation 所涉及的 image_id
    valid_image_ids = set(ann['image_id'] for ann in filtered_annotations)

    # 只保留对应 image
    filtered_images = [img for img in coco['images'] if img['id'] in valid_image_ids]

    # 只保留目标类别
    filtered_categories = [cat for cat in coco['categories'] if cat['id'] in target_category_ids]

    # 构建新的 COCO JSON
    new_coco = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    # 写入输出文件
    with open(output_json, 'w') as f:
        json.dump(new_coco, f, indent=4)

    print(f"筛选完成！共保留 {len(filtered_images)} 张图像，{len(filtered_annotations)} 个标注。")

if __name__ == '__main__':

    # Define categories
    # unified
    '''
    categories = [
        {"id": 1, "name": "Baton"},
        {"id": 2, "name": "Pliers"},
        {"id": 3, "name": "Hammer"},
        {"id": 4, "name": "Powerbank"},
        {"id": 5, "name": "Scissors"},
        {"id": 6, "name": "Wrench"},
        {"id": 7, "name": "Gun"},
        {"id": 8, "name": "Bullet"},
        {"id": 9, "name": "Sprayer"},
        {"id": 10, "name": "HandCuffs"},
        {"id": 11, "name": "Knife"},
        {"id": 12, "name": "Lighter"},
        {"id": 13, "name": "Utility_Knife"},
        {"id": 14, "name": "Multi-tool_Knife"},
        {"id": 15, "name": "Folding_Knife"},
        {"id": 16, "name": "Straight_Knife"},
        {"id": 17, "name": "Portable_Charger_1"},
        {"id": 18, "name": "Portable_Charger_2"},
        {"id": 19, "name": "Water"},
        {"id": 20, "name": "Laptop"},
        {"id": 21, "name": "Mobile_Phone"},
        {"id": 22, "name": "Tablet"},
        {"id": 23, "name": "Cosmetic"},
        {"id": 24, "name": "Nonmetallic_Lighter"}
    ]
    '''
    # pidray   
    
    categories = [
        {"id": 1, "name": "Baton"},
        {"id": 2, "name": "Pliers"},
        {"id": 3, "name": "Hammer"},
        {"id": 4, "name": "Powerbank"},
        {"id": 5, "name": "Scissors"},
        {"id": 6, "name": "Wrench"},
        {"id": 7, "name": "Gun"},
        {"id": 8, "name": "Bullet"},
        {"id": 9, "name": "Sprayer"},
        {"id": 10, "name": "HandCuffs"},
        {"id": 11, "name": "Knife"},
        {"id": 12, "name": "Lighter"}
    ]
    
    
    # opixray
    '''
    categories = [
        {"id": 1, "name": "Utility_Knife"}, 
        {"id": 2, "name": "Multi-tool_Knife"},
        {"id": 3, "name": "Folding_Knife"},
        {"id": 4, "name": "Straight_Knife"},
        {"id": 5, "name": "Scissor"}
    ]
    '''
    # opi-hi-continuous-unified
    # categories = [
    #     {"id": 1, "name": "Knife"}, 
    #     {"id": 2, "name": "Scissors"},
    #     {"id": 3, "name": "Charger"},
    #     {"id": 4, "name": "Water"},
    #     {"id": 5, "name": "Laptop"},
    #     {"id": 6, "name": "Mobile_Phone"},
    #     {"id": 7, "name": "Tablet"},
    #     {"id": 8, "name": "Cosmetic"},
    #     {"id": 9, "name": "Lighter"}
    # ]
    
    
    # hixray
    '''
    categories = [
        {"id": 1, "name": "Portable_Charger_1"},
        {"id": 2, "name": "Portable_Charger_2"},
        {"id": 3, "name": "Water"},
        {"id": 4, "name": "Laptop"},
        {"id": 5, "name": "Mobile_Phone"},
        {"id": 6, "name": "Tablet"},
        {"id": 7, "name": "Cosmetic"},
        {"id": 8, "name": "Nonmetallic_Lighter"}
    ]
    '''
    # categories = [
    #     {"id": 1, "name": "Charger"},
    #     {"id": 2, "name": "Water"},
    #     {"id": 3, "name": "Laptop"},
    #     {"id": 4, "name": "Mobile_Phone"},
    #     {"id": 5, "name": "Tablet"},
    #     {"id": 6, "name": "Cosmetic"},
    #     {"id": 7, "name": "Lighter"}
    # ]
    
    # categories = [
    #     {"id": 1, "name": "Baton"},
    #     {"id": 2, "name": "Pliers"},
    #     {"id": 3, "name": "Hammer"},
    #     {"id": 4, "name": "Powerbank"},
    #     {"id": 5, "name": "Scissors"},
    #     {"id": 6, "name": "Wrench"},
    #     {"id": 7, "name": "Gun"},
    #     {"id": 8, "name": "Bullet"},
    #     {"id": 9, "name": "Sprayer"},
    #     {"id": 10, "name": "HandCuffs"},
    #     {"id": 11, "name": "Knife"},
    #     {"id": 12, "name": "Lighter"},
    #     {"id": 13, "name": "Charger"},
    #     {"id": 14, "name": "Water"},
    #     {"id": 15, "name": "Laptop"},
    #     {"id": 16, "name": "Mobile_Phone"},
    #     {"id": 17, "name": "Tablet"},
    #     {"id": 18, "name": "Cosmetic"}
    # ]
    
    # Define COCO structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', default='/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_best_ablation_3/annotation/range1/annotation.txt')
    parser.add_argument('--img_path', default='/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_best_ablation_3/data/0.3')
    parser.add_argument('--output_json', default='/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_best_ablation_3/annotation/range1/anno.json')

    args = parser.parse_args()
    
    # annotation_file = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_h_sample/annotation/3/annotation.txt'
    # img_path = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_h_sample/data/'
    # output_json = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_h_sample/annotation/3/gen1_anno_refine.json'
    
    create_coco_from_annotation(args.annotation_file, args.img_path, args.output_json)
    '''
    for ratio in [1, 5]:
        original_json = '/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/train_positive.json'
        filter_file = f'/home/ct/data/sjl/diffusion/pidray_low_resolution/gligen/pidray_train_{ratio}%.pt'
        output_json = f'/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/pidray_train_positive_{ratio}%.json'
        create_coco_json_filter(original_json, filter_file, output_json)
        print(f'{ratio} done')
    

    print('annotation done')
    '''

    # opixray
    # anno_path = '/home/ct/data/lwz/dataset/OHP/annotations/opi_train_continuous.json'
    # output_path = '/home/ct/data/lwz/dataset/OHP/annotations/opi_train_continuous.json'
    # shift_category_ids(anno_path, output_path, categories, dataset='opixray')

    #hixray
    # anno_path = '/home/ct/data/lwz/dataset/OHP/annotations/hi-anno-train_continuous.json'
    # output_path = '/home/ct/data/lwz/dataset/OHP/annotations/hi-anno-train_continuous.json'
    # shift_category_ids(anno_path, output_path, categories, dataset='hixray')


    #pidray
    #anno_path = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_best_ablation_3/annotation/range4/anno.json'
    #output_path = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_best_ablation_3/annotation/range4/anno_unified.json'
    # anno_path = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_same_bg_hidden/annotation/anno_refine.json'
    # output_path = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_same_bg_hidden/annotation/anno_refine_continuous_unified.json'
    # shift_category_ids(anno_path, output_path, categories, dataset='pidray')

    # annotation_file_path = '/home/ct/data/lwz/dataset/PIDray/annotations/test_positive_unified.json'
    # coco = COCO(annotation_file_path)
    # print(coco.getCatIds())  # 输出所有合法类别ID

    # opi_anno_file = '/home/ct/data/lwz/dataset/OHP/annotations/corrected_order_test_ann_continuous_unified.json'
    # hi_anno_file = '/home/ct/data/lwz/dataset/OHP/annotations/hi-anno-val_continuous_unified.json'
    # pi_anno_file = '/home/ct/data/lwz/dataset/OHP/annotations/test_positive_continuous_unified.json'
    # opi_hi_unified_file = '/home/ct/data/lwz/dataset/OHP/annotations/opi_hi_continuous_unified_val.json'
    # opi_hi_pi_unified_file = '/home/ct/data/lwz/dataset/OHP/annotations/opi_hi_pi_continuous_unified_val.json'
    # merge_coco_annotations(opi_anno_file, hi_anno_file, opi_hi_pi_unified_file)
    # merge_coco_annotations(opi_hi_pi_unified_file, pi_anno_file, opi_hi_pi_unified_file)
    
    # for i in range(1, 12, 2): 
    #     task_id = i // 2 + 1
    #     target_category_ids_seq = [i, i + 1]
    #     target_category_ids_joint = list(range(1, i+2))

    #     train_input_json = '/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/train_positive.json'
    #     test_input_json = '/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/test_positive.json'
    #     train_output_json_seq = f'/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/class_incremental/train_positive_task{task_id}.json'
    #     test_output_json_joint = f'/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/class_incremental/test_positive_task{task_id}.json'
    #     if task_id > 1 and task_id < 6:
    #         train_output_json_joint = f'/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/annotations/class_incremental/train_positive_task1-{task_id}.json'
    #         filter_category(train_input_json, train_output_json_joint, target_category_ids_joint)
        
    #     filter_category(train_input_json, train_output_json_seq, target_category_ids_seq)
    #     if task_id < 6:
    #         filter_category(test_input_json, test_output_json_joint, target_category_ids_joint)
    #     print(f'task_id = {task_id}, target_category_ids_seq = {target_category_ids_seq}, target_category_ids_joint = {target_category_ids_joint}')
            
        



    



