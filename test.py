from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
checkpoint = "/inspire/hdd/project/embodied-multimodality/public/pywang/resources/video_hf/models/video_mllama"

model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

images=[
        "/inspire/hdd/project/embodied-multimodality/public/pywang/data/image_data/DenseFusion-1M/images/DenseFusion-1M/2042121004267.png",
        "/inspire/hdd/project/embodied-multimodality/public/pywang/data/image_data/DenseFusion-1M/images/DenseFusion-1M/2294043002950.png",
        "/inspire/hdd/project/embodied-multimodality/public/pywang/data/image_data/DenseFusion-1M/images/DenseFusion-1M/2294043002950.png"
    ]

videos= ["/inspire/hdd/project/embodied-multimodality/public/pywang/youtube_data/v3/Sports/freeride/within_4min/EEw506878lI.mp4"]
# 或者
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe the images.\n<image><image><image>"},       
    ]}
]


input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# import debugpy; debugpy.connect(('localhost', 9998))
inputs = processor(
    images = images,
    # videos = videos,
    text = input_text,
    add_special_tokens=False,
    return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=2048,do_sample=False)
print(processor.decode(output[0]))