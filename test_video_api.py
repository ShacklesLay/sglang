from openai import OpenAI

client = OpenAI(base_url=f"http://localhost:{30058}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/pys/data/1066674931.mp4"
                    },
                },
                {
                    "type": "text",
                    "text": "Describe the video in detail, output in markdown format",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "And describe the image in detail, output in markdown format",
                },
            ],
        }
    ],
    max_tokens=1000,
)

print(response.choices[0].message.content)