from mmdet.apis import init_detector, inference_detector, show_result
import glob

# config_file = 'configs/htc/htc_r50_fpn_20e.py'
# checkpoint_file = 'checkpoints/htc_r50_fpn_20e_20190408-c03b7015.pth'
config_file = 'my_configs/cascade_mask_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/modanet_cascade_mask_rcnn_r50_fpn_1x/latest.pth'
score_thr = 0.85

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# img = 'test.jpg'
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES, score_thr=score_thr, show=False, out_file='result.jpg')

imgs = glob.glob('test_manual/*.jpg')
print(imgs)

for img in imgs:
    print("Inferencing {} ...".format(img))
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, score_thr=score_thr, show=False, out_file='{}_result.jpg'.format(img[:-4]))

# for i, result in enumerate(inference_detector(model, imgs)):
# 	img = imgs[i]
# 	show_result(img, result, model.CLASSES, score_thr=score_thr, show=False, out_file='{}_result.jpg'.format(img[:-4]))
