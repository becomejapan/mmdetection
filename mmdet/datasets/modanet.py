from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class ModanetDataset(CocoDataset):

    CLASSES = ('bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses',
               'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie')

