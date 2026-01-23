import json
from fastapi import FastAPI, Request

from api.model_manager import ModelManager
from api.util import success, error

app = FastAPI(title="Ai模型主页")

MODEL_ROOT = "configs"

model_manager = ModelManager(model_root=MODEL_ROOT)


@app.on_event("startup")
def startup_event():
    default_model = "macbert_yq_class_0.2"
    try:
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
        result = model_manager.predict(
            model_name=model_name,
            sample=sample
        )
    except Exception as e:
        return error(str(e))

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
