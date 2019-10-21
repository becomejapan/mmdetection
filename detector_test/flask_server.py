from flask import Flask, request, send_file
import requests
import shutil
import hashlib
import os
from tempfile import mkstemp
from mmdet.apis import init_detector, inference_detector, show_result


def gethex(url):
    m = hashlib.sha256()
    m.update(url.encode('utf-8'))
    return m.hexdigest()


def download_image(url, fname):
    resp = requests.get(url, stream=True)
    with open(fname, 'wb') as ofile:
        shutil.copyfileobj(resp.raw, ofile)


def detect_masks(img, out_file):
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, score_thr=score_thr, show=False, out_file=out_file)

# Cascade Mask-RCNN
config_file = '../my_configs/cascade_mask_rcnn_r50_fpn_1x.py'
checkpoint_file = '../work_dirs/modanet_cascade_mask_rcnn_r50_fpn_1x/latest.pth'

# Mask-RCNN
# config_file = '../my_configs/mask_rcnn_r50_fpn_1x.py'
# checkpoint_file = '../work_dirs/modanet_mask_rcnn_r50_fpn_1x/latest.pth'

score_thr = 0.85
model = init_detector(config_file, checkpoint_file, device='cuda:0')

app = Flask(__name__)


@app.route("/modanet-test")
def fetch_and_cache():
    """This function gets an image from a URL and saves a local copy."""
    imageurl = request.args.get('image', None)
    if not imageurl:
        return "ERROR"

    urlhash = gethex(imageurl)
    ofile = 'cache/' + urlhash + '.jpg'
    if os.path.isfile(ofile):
        return send_file(ofile, mimetype='image/jpeg')

    tmpfile = mkstemp()[1]
    download_image(imageurl, tmpfile)
    detect_masks(tmpfile, ofile)
    os.remove(tmpfile)
    return send_file(ofile, mimetype='image/jpeg')
