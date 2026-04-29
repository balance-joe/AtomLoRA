import argparse
import json
import sys
import os


def cmd_train(args):
    from main import main
    main(args.config)


def cmd_eval(args):
    from main import evaluate
    evaluate(args.config)


def cmd_predict(args):
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


def cmd_serve(args):
    import uvicorn
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
    args.func(args)


if __name__ == "__main__":
    main()
