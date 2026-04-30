import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends

from api.model_manager import ModelManager
from api.security import verify_api_key, create_rate_limiter
from api.util import success, error
from src.utils.logger import get_logger

logger = get_logger()
model_manager = ModelManager()

# 异步 GPU 互斥：Semaphore(1) 保证同一时刻只有一个 GPU 操作
# run_in_executor 将阻塞的推理调用放到线程池，不阻塞事件循环
gpu_semaphore = asyncio.Semaphore(1)
executor = ThreadPoolExecutor(max_workers=1)
rate_limiter = create_rate_limiter()


async def _load_default_model():
    """启动时加载默认模型（通过环境变量或配置指定）"""
    serve_config = os.environ.get("ATOMLORA_SERVE_CONFIG")
    if serve_config:
        try:
            from src.config.parser import parse_config
            config = parse_config(serve_config, mode="serve")
            model_name = config["exp_id"]
            async with gpu_semaphore:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    executor,
                    lambda: model_manager.load_model(model_name, config_path=serve_config),
                )
            logger.info(f"默认模型已加载: {model_name} (from {serve_config})")
        except FileNotFoundError as e:
            logger.error(f"[ERROR] 模型或配置文件不存在: {e}")
            logger.error("[HINT] 请检查配置中的 model.path 和 data 路径是否正确")
        except Exception as e:
            logger.error(f"[ERROR] 默认模型加载失败: {type(e).__name__}: {e}")
        return

    default_model = os.environ.get("ATOMLORA_DEFAULT_MODEL")
    if not default_model:
        logger.info("未设置默认模型（通过 atomlora serve --config 或 ATOMLORA_DEFAULT_MODEL 环境变量指定）")
        return
    try:
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor,
                lambda: model_manager.load_model(default_model),
            )
        logger.info(f"默认模型已加载: {default_model}")
    except Exception as e:
        logger.error(f"默认模型加载失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _load_default_model()
    yield


app = FastAPI(title="Ai模型主页", lifespan=lifespan)


@app.get("/index")
def index():
    """健康检查接口"""
    return success()


@app.post("/load")
async def load_model(request: Request, _: None = Depends(verify_api_key)):
    """显式加载指定模型"""
    await rate_limiter.check(request)

    try:
        body = await request.json()
    except Exception:
        return error("请求体必须是合法的 JSON")

    config_path = body.get("config_path")
    if not config_path:
        return error("config_path 必填")

    try:
        from src.config.parser import parse_config
        config = parse_config(config_path, mode="serve")
        model_name = config["exp_id"]
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor,
                lambda: model_manager.load_model(model_name, config_path=config_path),
            )
        return success(msg=f"模型已加载: {model_name}")
    except FileNotFoundError as e:
        return error(f"模型或配置文件不存在: {e}")
    except Exception as e:
        return error(f"{type(e).__name__}: {e}")


@app.post("/predict")
async def predict(request: Request, _: None = Depends(verify_api_key)):
    """预测接口：接收模型名和样本，返回预测结果"""
    await rate_limiter.check(request)

    try:
        body = await request.json()
    except Exception:
        return error("请求体必须是合法的 JSON")

    model_name = body.get("model_name")
    sample = body.get("sample")

    if not model_name:
        return error("model_name 必填")

    if not sample or not isinstance(sample, dict):
        return error("sample 必填且必须是对象")

    try:
        text_col = model_manager.get_text_col(model_name)
        content = sample.get(text_col)
        if not content or not isinstance(content, str):
            return error(f"sample.{text_col} 必填且必须是字符串")
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                executor,
                lambda: model_manager.predict(model_name=model_name, sample=sample),
            )
    except FileNotFoundError as e:
        return error(f"模型不存在: {e}")
    except Exception as e:
        return error(f"{type(e).__name__}: {e}")

    result = json.loads(json.dumps(result, ensure_ascii=False, indent=2))
    return success(data=result)


@app.get("/model_info/{model_name}")
def model_info(model_name: str):
    """获取模型详细信息"""
    try:
        info = model_manager.get_model_info(model_name)
        return success(data=info)
    except Exception as e:
        return error(str(e))


@app.post("/unload")
async def unload_models(request: Request, _: None = Depends(verify_api_key)):
    """释放所有已加载模型的显存"""
    await rate_limiter.check(request)
    async with gpu_semaphore:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, model_manager.unload_all)
    return success(msg="所有模型已卸载")
