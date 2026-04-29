import argparse
import json
import sys
import os

# Windows 下强制 UTF-8 输出，防止中文日志乱码
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def _resolve_config(config_path: str) -> str:
    """Resolve --config argument. Supports 'latest' shorthand."""
    if config_path == "latest":
        resolved = os.path.join("outputs", "latest", "config.yaml")
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                "outputs/latest/config.yaml 不存在。请先运行一次训练。"
            )
        return resolved
    return config_path


def cmd_train(args):
    args.config = _resolve_config(args.config)
    from atomlora.engine import main
    main(args.config)


def cmd_eval(args):
    args.config = _resolve_config(args.config)
    from atomlora.engine import evaluate
    evaluate(args.config)


def cmd_predict(args):
    args.config = _resolve_config(args.config)
    from src.config.parser import parse_config
    from src.utils.logger import init_logger
    from src.predict.predictor import TextAuditPredictor

    config = parse_config(args.config)
    init_logger(config["exp_id"], config["task_type"])

    predictor = TextAuditPredictor(config=config)
    text_col = config["data"]["text_col"]
    result = predictor.predict({text_col: args.text})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    predictor.close()


# 常见错误的用户友好提示
_ERROR_HINTS = {
    "FileNotFoundError": "文件不存在。请检查配置中的路径是否正确，模型是否已下载。",
    "RuntimeError:CUDA": "GPU 显存不足。尝试减小 batch_size 或用 CPU 模式。",
    "KeyError:label": "标签映射错误。请检查 label_col / label_mapping 配置是否与数据匹配。",
    "ModuleNotFoundError": "缺少依赖包。运行 pip install -r requirements.txt 或 bash install.sh。",
    "No such file": "配置文件不存在。请检查 --config 参数路径。",
}


def _friendly_error(e: Exception, config_path: str = None):
    """将异常转为用户友好的错误信息"""
    err_type = type(e).__name__
    err_msg = str(e)

    print(f"\n{'='*50}")
    print(f"[ERROR] {err_type}: {err_msg}")

    # 匹配提示
    for key, hint in _ERROR_HINTS.items():
        if key.lower() in (err_type + err_msg).lower():
            print(f"[HINT] {hint}")
            break

    if config_path:
        print(f"[INFO] 配置文件: {config_path}")
    print(f"{'='*50}\n")


def cmd_serve(args):
    args.config = _resolve_config(args.config)
    import uvicorn
    # 传递配置文件路径，让 API 服务直接加载指定配置
    os.environ["ATOMLORA_SERVE_CONFIG"] = os.path.abspath(args.config)
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        prog="atomlora",
        description="AtomLoRA - Lightweight LoRA fine-tuning framework",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--config", required=True, help="YAML config path")
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_predict = sub.add_parser("predict", help="Run single-text prediction")
    p_predict.add_argument("--config", required=True, help="YAML config path")
    p_predict.add_argument("--text", required=True, help="Text to predict")
    p_predict.set_defaults(func=cmd_predict)

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI inference server")
    p_serve.add_argument("--config", required=True, help="YAML config path")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    try:
        args.func(args)
    except Exception as e:
        config_path = getattr(args, 'config', None)
        _friendly_error(e, config_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
