from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = '../configs/_base_/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:7')

# test a single image and show the results
img = '../demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='../demo/result.jpg')

# test a video and show the results
video = mmcv.VideoReader('../demo/demo.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)

