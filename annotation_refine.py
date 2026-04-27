from torch.utils.data import Dataset
from pycocotools.coco import COCO


class Caption_coco(Dataset):
    def __init__(self, args):
        coco_api = COCO(args.annotation_file)
        img_ids = coco_api.getImgIds()
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.loadAnns(coco_api.getAnnIds(imgIds=[img_id])) for img_id in img_ids]
        self.data = imgs
        self.annos = anns
        self.coco = coco_api

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]['file_name']
        anno = self.annos[index]
        return {'image_name': image_name, 'annos': anno}

    @staticmethod
    def collate_fn(batch):
        return [res for res in batch]
