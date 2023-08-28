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

from facechain.inference import GenPortrait
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, cloth_prompt

inference_threadpool = ThreadPoolExecutor(max_workers=5)

inference_done_count = 0

HOT_MODELS = [
    "\N{fire}数字身份(Digital Identity)",
]

class UploadTarget(enum.Enum):
    PERSONAL_PROFILE = 'Personal Profile'
    LORA_LIaBRARY = 'LoRA Library'

def update_cloth(style_index):
    prompts = []
    if style_index == 0:
        example_prompt = generate_pos_prompt(styles[style_index]['name'],
                                             cloth_prompt[0]['prompt'])
        for prompt in cloth_prompt:
            prompts.append(prompt['name'])
    else:
        example_prompt = generate_pos_prompt(styles[style_index]['name'],
                                             styles[style_index]['add_prompt_style'])
        prompts.append(styles[style_index]['cloth_name'])
    return gr.Radio.update(choices=prompts, value=prompts[0]), gr.Textbox.update(value=example_prompt)


def update_prompt(style_index, cloth_index):
    if style_index == 0:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         cloth_prompt[cloth_index]['prompt'])
    else:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         styles[style_index]['add_prompt_style'])
    return gr.Textbox.update(value=pos_prompt)

def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image

def generate_pos_prompt(style_model, prompt_cloth):
    if style_model == styles[0]['name'] or style_model is None:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    return pos_prompt


def launch_pipeline(uuid,
                    pos_prompt,
                    user_models,
                    num_images=1,
                    style_model=None,
                    multiplier_style=0.25
                    ):
    base_model = 'ly261666/cv_portrait_model'
    before_queue_size = inference_threadpool._work_queue.qsize()
    before_done_count = inference_done_count
    style_model = styles[style_model]['name']

    if style_model == styles[0]['name']:
        style_model_path = None
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
        style_model_path = os.path.join(model_dir, matched['bin_file'])

    print("-------user_models: ", user_models)
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    output_model_name = 'personalizaition_lora'
    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)

    lora_model_path = f'/tmp/{uuid}/{output_model_name}'

    gen_portrait = GenPortrait(pos_prompt, neg_prompt, style_model_path, multiplier_style, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)
    future = inference_threadpool.submit(gen_portrait, instance_data_dir,
                                         num_images, base_model, lora_model_path, 'film/film', 'v2.0')

    while not future.done():
        is_processing = future.running()
        if not is_processing:
            cur_done_count = inference_done_count
            to_wait = before_queue_size - (cur_done_count - before_done_count)
            yield ["排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5), None]
        else:
            yield ["生成中, 请耐心等待(Generating)...", None]
        time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    image_path = './lora_result.png'
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        cv2.imwrite(image_path, result)

        yield ["生成完毕(Generating done)！", outputs_RGB]
    else:
        yield ["生成失败，请重试(Generating failed, please retry)！", outputs_RGB]

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

def inference_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                user_models = gr.Radio(label="模型选择(Model list)", choices=HOT_MODELS, type="value",
                                       value=HOT_MODELS[0])
                style_model_list = []
                for style in styles:
                    style_model_list.append(style['name'])
                style_model = gr.Dropdown(choices=style_model_list, value=styles[0]['name'], 
                                          type="index", label="风格模型(Style model)")
                
                prompts=[]
                for prompt in cloth_prompt:
                    prompts.append(prompt['name'])
                cloth_style = gr.Radio(choices=prompts, value=cloth_prompt[0]['name'],
                                       type="index", label="服装风格(Cloth style)")

                with gr.Accordion("高级选项(Expert)", open=False):
                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3,
                                        value=generate_pos_prompt(None, cloth_prompt[0]['prompt']), interactive=True)
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量(Number of photos)', value=6, precision=1, minimum=1, maximum=6)
                    gr.Markdown('''
                    注意：最多支持生成6张图片!(You may generate a maximum of 6 photos at one time!)
                        ''')

        display_button = gr.Button('开始生成(Start!)')

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果(Result)')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")
                                                                               
        style_model.change(update_cloth, style_model, [cloth_style, pos_prompt], queue=False)
        cloth_style.change(update_prompt, [style_model, cloth_style], [pos_prompt], queue=False)
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, pos_prompt, user_models, num_images, style_model, multiplier_style],
                             outputs=[infer_progress, output_images])

    return demo


with gr.Blocks(css='style.css') as demo:
    inference_input()

demo.queue(status_update_rate=1, api_open=False).launch(debug=True, share=True, show_error=True)
