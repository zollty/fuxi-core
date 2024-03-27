import sys
import os

# 获取当前脚本的绝对路径
__current_script_path = os.path.abspath(__file__)
# 将项目根目录添加到sys.path
get_runtime_root_dir = os.path.dirname(os.path.dirname(__current_script_path))
sys.path.append(get_runtime_root_dir)
print(get_runtime_root_dir)

from typing import List
from fuxi.utils.thread_helper import run_in_executor


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed search docs."""
    print(f"-----start: {texts[0]}")
    print(texts)
    time.sleep(5)
    print(f"-----end: {texts[0]}")
    return [[1232, 4545.4]]


async def aembed_documents(texts: List[str]) -> List[List[float]]:
    """Asynchronous Embed search docs."""
    return await run_in_executor(None, embed_documents, texts)


async def main(embed):
    sentences = [
        '我是AI智能机器人，你可以称呼我为聊天助手或者知识顾问。我是由邹天涌先生所指导的，负责回答和解决各种问题，提供信息和帮助。如果你有任何问题，尽管问吧！',
        'sentence_1']
    embeddings = await embed.async_encode(sentences)

    start_time = time.time()
    for i in range(tn):
        embeddings = await embed.async_encode(sentences)

    end_time = time.time() - start_time
    print("--------------BCE_Sentence_Embeddings------------------")
    print(end_time)
    print(embeddings[0][0:10])


if __name__ == '__main__':
    import asyncio

    import platform
    import time
    from fuxi.utils.torch_helper import detect_device
    from hpdeploy.embeddings import EmbeddingsService, BCE_Native_Embeddings, BCE_Sentence_Embeddings, \
        BCE_Huggingface_Embeddings, BCE_Transformer_Embeddings, \
        Jina_Sentence_Embeddings, Jina_Transformer_Embeddings

    os_type = platform.system()
    print(os_type)
    if os_type == "Windows":
        path = "G:/50-TEMP/models/embed/bce-embedding-base_v1"
        jina_path = "G:/50-TEMP/models/embed/jina-embeddings-v2-base-zh"
    else:
        path = "/ai/models/bce-embedding-base_v1"
        jina_path = "/ai/models/jina-embeddings-v2-base-zh"

    print(detect_device())
    tn: int = 100
    device = 'cuda:0'  # if no GPU, set "cpu"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    coros = [main(BCE_Native_Embeddings(path, device)),
             main(BCE_Sentence_Embeddings(path, device)),
             main(BCE_Huggingface_Embeddings(path, device)),
             main(BCE_Transformer_Embeddings(path, device)),
             main(Jina_Sentence_Embeddings(jina_path, device)),
             main(Jina_Transformer_Embeddings(jina_path, device)),
             ]

    tasks = [loop.create_task(coro) for coro in coros]
    try:
        for task in tasks:
            loop.run_until_complete(task)
    except KeyboardInterrupt:
        loop.stop()
    finally:
        loop.close()
