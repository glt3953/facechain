# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
import numpy as np
import torch
from modelscope import snapshot_download

#from facechain.inference import GenPortrait
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, cloth_prompt

torch.backends.cuda.matmul.allow_tf32 = False

training_threadpool = ThreadPoolExecutor(max_workers=1)

training_done_count = 0

HOT_MODELS = [
    "\N{fire}数字身份(Digital Identity)",
]

class UploadTarget(enum.Enum):
    PERSONAL_PROFILE = 'Personal Profile'
    LORA_LIaBRARY = 'LoRA Library'
    
def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def train_lora_fn(foundation_model_path=None, revision=None, output_img_dir=None, work_dir=None):
    os.system(
        f'PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py --pretrained_model_name_or_path={foundation_model_path} '
        f'--revision={revision} --sub_path="film/film" '
        f'--output_dataset_name={output_img_dir} --caption_column="text" --resolution=512 '
        f'--random_flip --train_batch_size=1 --num_train_epochs=200 --checkpointing_steps=5000 '
        f'--learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 --seed=42 --output_dir={work_dir} '
        f'--lora_r=32 --lora_alpha=32 --lora_text_encoder_r=32 --lora_text_encoder_alpha=32')

class Trainer:
    def __init__(self):
        pass

    def run(
            self,
            uuid: str,
            instance_images: list,
    ) -> str:

        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if instance_images is None:
            raise gr.Error('您需要上传训练图片(Please upload photos)！')
        if len(instance_images) > 10:
            raise gr.Error('您需要上传小于10张训练图片(Please upload at most 10 photos)！')
        if not uuid:
            if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                return "请登陆后使用(Please login first)! "
            else:
                uuid = 'qw'

        output_model_name = 'personalizaition_lora'

        # mv user upload data to target dir
        instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)
        print("--------uuid: ", uuid)

        if not os.path.exists(f"/tmp/{uuid}"):
            os.makedirs(f"/tmp/{uuid}")
        work_dir = f"/tmp/{uuid}/{output_model_name}"
        print("----------work_dir: ", work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(instance_data_dir, ignore_errors=True)

        prepare_dataset([img['name'] for img in instance_images], output_dataset_dir=instance_data_dir)
        data_process_fn(instance_data_dir, True)

        # train lora
        train_lora_fn(foundation_model_path='ly261666/cv_portrait_model',
                      revision='v2.0',
                      output_img_dir=instance_data_dir,
                      work_dir=work_dir)
        print(f'模型路径：{work_dir}，图片路径：{output_img_dir}')

        message = f'训练已经完成！请切换至 [形象体验] 标签体验模型效果(Training done, please switch to the inference tab to generate photos.)'
        print(message)
        return message


def flash_model_list(uuid):
    folder_path = f"/tmp/{uuid}"
    folder_list = []
    print("------flash_model_list folder_path: ", folder_path)
    if not os.path.exists(folder_path):
        print('--------The folder_path is missing.')
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/output/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    print("-------folder_list + HOT_MODELS: ", folder_list + HOT_MODELS)
    return gr.Radio.update(choices=HOT_MODELS + folder_list)


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def train_input():
    trainer = Trainer()

    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                    file_count="multiple")

                    clear_button = gr.Button("清空图片(Clear photos)")
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                    gr.Markdown('''
                        - Step 1. 上传计划训练的图片，3~10张头肩照（注意：请避免图片中出现多人脸、脸部遮挡等情况，否则可能导致效果异常）
                        - Step 2. 点击 [开始训练] ，启动形象定制化训练，约需15分钟，请耐心等待～
                        - Step 3. 切换至 [形象体验] ，生成你的风格照片
                        ''')
                    gr.Markdown('''
                        - Step 1. Upload 3-10 headshot photos of yours (Note: avoid photos with multiple faces or face obstruction, which may lead to non-ideal result).
                        - Step 2. Click [Train] to start training for customizing your Digital-Twin, this may take up-to 15 mins.
                        - Step 3. Switch to [Inference] Tab to generate stylized photos.
                        ''')

        run_button = gr.Button('开始训练（等待上传图片加载显示出来再点，否则会报错）'
                               'Start training (please wait until photo(s) fully uploaded, otherwise it may result in training failure)')

        with gr.Box():
            gr.Markdown('''
            请等待训练完成
            
            Please wait for the training to complete.
            ''')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验。
            
            安装方法请参考：https://github.com/modelscope/facechain .
            
            If you are experiencing prolonged waiting time, you may try on [ModelScope NoteBook](https://modelscope.cn/my/mynotebook/preset) to prepare your dedicated environment.
                        
            You may refer to: https://github.com/modelscope/facechain for installation instruction.
            ''')

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             instance_images,
                         ],
                         outputs=[output_message])

    return demo
    
with gr.Blocks(css='style.css') as demo:
    train_input()

demo.queue(status_update_rate=1, api_open=False).launch(debug=True, share=True, show_error=True)
