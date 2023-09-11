# !export TORCH_USE_CUDA_DSA=1

import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
# import torch

# torch.backends.cuda.matmul.allow_tf32 = False

face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd')
# 支持 url image and abs dir image path
#img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg'
img_path = 'images/woman/pose2.png'
result = face_detection(img_path)

# 提供可视化结果
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_draw = draw_face_detection_result('srcImg.jpg', result)
import matplotlib.pyplot as plt
plt.imshow(img_draw)
