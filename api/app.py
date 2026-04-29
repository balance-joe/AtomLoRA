import json
import os
from threading import Lock
from fastapi import FastAPI, Request

from api.model_manager import ModelManager
from api.util import success, error

app = FastAPI(title="Ai模型主页")

model_manager = ModelManager()
# 注意：当前仅支持单 worker 模式（uvicorn 默认就是单 worker）
# 多 worker (--workers N) 时每个进程有独立的 GPU 锁，无法互斥，会导致 OOM
gpu_lock = Lock()


@app.on_event("startup")
def startup_event():
    # 优先从 ATOMLORA_SERVE_CONFIG 加载（CLI 传入的配置路径）
    serve_config = os.environ.get("ATOMLORA_SERVE_CONFIG")
    if serve_config:
        try:
            from src.config.parser import parse_config
            config = parse_config(serve_config)
            model_name = config["exp_id"]
            with gpu_lock:
                model_manager.load_model(model_name, config_path=serve_config)
            print(f"默认模型已加载: {model_name} (from {serve_config})")
        except FileNotFoundError as e:
            print(f"[ERROR] 模型或配置文件不存在: {e}")
            print("[HINT] 请检查配置中的 model.path 和 data 路径是否正确")
        except Exception as e:
            print(f"[ERROR] 默认模型加载失败: {type(e).__name__}: {e}")
        return

    # 回退：从环境变量指定的模型名加载
    default_model = os.environ.get("ATOMLORA_DEFAULT_MODEL")
    if not default_model:
        print("未设置默认模型（通过 atomlora serve --config 或 ATOMLORA_DEFAULT_MODEL 环境变量指定）")
        return
    try:
        with gpu_lock:
            model_manager.load_model(default_model)
        print(f"默认模型已加载: {default_model}")
    except Exception as e:
        print(f"默认模型加载失败: {e}")


@app.get("/index")
def index():
    return success()


@app.post("/predict")
async def predict(request: Request):
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

    content = sample.get("content")
    if not content or not isinstance(content, str):
        return error("sample.content 必填且必须是字符串")

    try:
        with gpu_lock:
            result = model_manager.predict(
                model_name=model_name,
                sample=sample
            )
    except FileNotFoundError as e:
        return error(f"模型不存在: {e}")
    except Exception as e:
        return error(f"{type(e).__name__}: {e}")

    # 保持你原来的输出格式
    result = json.loads(json.dumps(result, ensure_ascii=False, indent=2))
    return success(data=result)


@app.get("/model_info/{model_name}")
def model_info(model_name: str):
    try:
        info = model_manager.get_model_info(model_name)
        return success(data=info)
    except Exception as e:
        return error(str(e))


@app.post("/unload")
def unload_models():
    """释放所有已加载模型的显存"""
    with gpu_lock:
        model_manager.unload_all()
    return success(msg="所有模型已卸载")
