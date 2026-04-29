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


def cmd_train(args):
    from atomlora.engine import main
    main(args.config)


def cmd_eval(args):
    from atomlora.engine import evaluate
    evaluate(args.config, data_path=args.data_path)


def cmd_predict(args):
    from src.config.parser import parse_config
    from src.utils.logger import init_logger
    from src.predict.predictor import TextAuditPredictor

    config = parse_config(args.config, mode="predict")
    init_logger(config["exp_id"], config["task_type"])

    predictor = TextAuditPredictor(config=config)
    text_col = config["data"]["text_col"]
    result = predictor.predict({text_col: args.text})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    predictor.close()


# 常见错误的用户友好提示（key 精确匹配异常类型名）
_ERROR_HINTS = {
    "FileNotFoundError": "文件不存在。请检查配置中的路径是否正确，模型是否已下载。",
    "RuntimeError": "运行时错误。如果涉及 CUDA，尝试减小 batch_size 或用 CPU 模式。",
    "KeyError": "键错误。请检查配置中的字段名是否正确（如 label_col / label_mapping）。",
    "ModuleNotFoundError": "缺少依赖包。运行 pip install -r requirements.txt 或 bash install.sh。",
    "ValueError": "值错误。请检查配置文件中的参数类型和取值范围。",
}


def _friendly_error(e: Exception, config_path: str = None):
    """将异常转为用户友好的错误信息"""
    err_type = type(e).__name__
    err_msg = str(e)

    print(f"\n{'='*50}")
    print(f"[ERROR] {err_type}: {err_msg}")

    # 按异常类型名精确匹配提示
    hint = _ERROR_HINTS.get(err_type)
    if hint:
        print(f"[HINT] {hint}")

    if config_path:
        print(f"[INFO] 配置文件: {config_path}")
    print(f"{'='*50}\n")


def cmd_serve(args):
    import uvicorn
    from src.config.parser import parse_config, resolve_runtime_config_path
    from src.utils.device import resolve_device

    config = parse_config(args.config, mode="serve")
    device = resolve_device(config)
    if args.reload and device.type == "cuda":
        raise ValueError("GPU 推理服务不支持 --reload，请使用单进程模式启动。")

    os.environ["ATOMLORA_SERVE_CONFIG"] = resolve_runtime_config_path(args.config, mode="serve")
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
    p_eval.add_argument("--data-path", help="Optional evaluation dataset path")
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
