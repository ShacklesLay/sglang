import asyncio
import time
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor

# 创建client
client = OpenAI(base_url=f"http://localhost:{30058}/v1", api_key="None")

def send_single_request():
    """发送单个请求"""
    
    response = client.chat.completions.create(
        model="Llama-3.2-11B-Vision-Instruct",  # 使用启动时的模型名
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/000271252.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe the images?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg"
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,  # 减少token数量以便更快看到结果
    )
    print(response.choices[0].message.content)
    return response

def test_batch_processing():
    """测试batch处理 - 并发发送多个请求"""
    print("=== 测试Batch处理 - 并发请求 ===")
    
    # 准备多个请求
    requests_data = [
        (1, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "What objects do you see in this image?"),
        (2, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "Describe the colors in this image"),
        (3, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "What is the main subject of this image?"),
        (4, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "Count the number of people in this image"),
        (5, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "What is the weather like in this image?"),
    ]
    
    print(f"准备发送 {len(requests_data)} 个并发请求...")
    start_time = time.time()
    
    # 使用ThreadPoolExecutor并发发送请求
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for req_id, image_path, question in requests_data:
            future = executor.submit(send_single_request, req_id, image_path, question)
            futures.append(future)
            # 稍微延迟一下，让请求几乎同时到达但有微小间隔
            time.sleep(0.1)
        
        # 等待所有请求完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Request failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n所有请求完成，总耗时: {total_time:.2f}秒")

def test_sequential_requests():
    """测试顺序请求 - 对比用"""
    print("\n=== 测试顺序请求（对比用） ===")
    
    requests_data = [
        (1, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "What objects do you see in this image?"),
        (2, "/inspire/hdd/project/embodied-multimodality/public/pywang/cktan/reps/Streaming/test_data/1.jpg", "Describe the colors in this image"),
    ]
    
    start_time = time.time()
    
    for req_id, image_path, question in requests_data:
        send_single_request(req_id, image_path, question)
    
    total_time = time.time() - start_time
    print(f"\n顺序请求完成，总耗时: {total_time:.2f}秒")

if __name__ == "__main__":
    # 运行batch测试
    # test_batch_processing()
    send_single_request()
    
    # 可选：运行顺序请求对比
    # test_sequential_requests() 