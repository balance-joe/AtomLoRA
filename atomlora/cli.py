import argparse
import json
import sys
import os

from src.utils.logger import get_logger, ensure_utf8_stdio

ensure_utf8_stdio()
logger = get_logger()


def cmd_train(args):
    """执行模型训练"""
    from atomlora.engine import main
    main(args.config)


def cmd_eval(args):
    """评估已训练的模型"""
    from atomlora.engine import evaluate
    evaluate(args.config, data_path=args.data_path)


def cmd_predict(args):
    """单条文本预测"""
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


# 异常类型名 → 用户友好的排查建议
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

    logger.error(f"\n{'='*50}")
    logger.error(f"[ERROR] {err_type}: {err_msg}")

    hint = _ERROR_HINTS.get(err_type)
    if hint:
        logger.error(f"[HINT] {hint}")

    if config_path:
        logger.error(f"[INFO] 配置文件: {config_path}")
    logger.error(f"{'='*50}\n")


def cmd_serve(args):
    """启动 FastAPI 推理服务"""
    import uvicorn
    from src.config.parser import parse_config, resolve_runtime_config_path
    from src.utils.device import resolve_device

    config = parse_config(args.config, mode="serve")
    device = resolve_device(config)
    if args.reload and device.type == "cuda":
        raise ValueError("GPU 推理服务不支持 --reload，请使用单进程模式启动。")

    os.environ["ATOMLORA_SERVE_CONFIG"] = resolve_runtime_config_path(args.config, mode="serve")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_split(args):
    """切分数据集为 train/dev/test"""
    from src.data.splitter import split_data

    # 支持从配置文件读取字段名，命令行参数优先级更高
    input_path = args.input
    output_dir = args.output
    text_col = args.text_col
    label_col = args.label_col
    label_mapping = None
    label_subset = None

    if args.config:
        from src.config.parser import parse_config
        config = parse_config(args.config, mode="train")
        data_cfg = config["data"]
        text_col = text_col or data_cfg.get("text_col")
        # 配置中 label_col 可能是 {任务名: 字段名} 的字典，需要提取原始字段名
        cfg_label_col = data_cfg.get("label_col")
        if not label_col:
            if isinstance(cfg_label_col, dict):
                label_col = list(cfg_label_col.values())[0]
            else:
                label_col = cfg_label_col
        label_mapping = data_cfg.get("label_mapping")
        label_subset = data_cfg.get("label_subset")
        # 未指定 --input 时，从配置的 train_path 读取源数据
        if not input_path:
            input_path = data_cfg.get("train_path")
        if not output_dir:
            output_dir = os.path.dirname(input_path) if input_path else "."

    if not input_path:
        logger.error("[ERROR] 必须指定 --input 或 --config（config 中需有 train_path）")
        sys.exit(1)
    if not text_col or not label_col:
        logger.error("[ERROR] 必须指定 --text-col 和 --label-col，或通过 --config 提供")
        sys.exit(1)

    report = split_data(
        input_path=input_path,
        output_dir=output_dir,
        text_col=text_col,
        label_col=label_col,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=args.stratify,
        label_mapping=label_mapping,
        label_subset=label_subset,
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"[OK] 切分完成 - 共 {report['total_samples']} 条样本")
    for name, info in report["splits"].items():
        logger.info(f"  {name}: {info['count']} 条 -> {info['path']}")
    logger.info(f"配置文件: {report['config_path']}")
    logger.info(f"报告: {os.path.join(os.path.dirname(report['config_path']), 'split_report.json')}")
    logger.info(f"\n下一步: atomlora train --config {report['config_path']}")
    logger.info(f"{'='*50}\n")


def cmd_doctor(args):
    """诊断数据集质量"""
    import yaml as _yaml
    from src.data.doctor import run_doctor, format_report_markdown

    # 优先用完整配置解析，解析失败时降级为纯 YAML 读取（兼容最小配置）
    try:
        from src.config.parser import parse_config, OUTPUTS_ROOT
        config = parse_config(args.config, mode="train")
    except (ValueError, FileNotFoundError):
        with open(args.config, "r", encoding="utf-8") as f:
            config = _yaml.safe_load(f)
        OUTPUTS_ROOT = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
        )

    exp_id = config.get("exp_id", "unknown")

    report = run_doctor(config)

    output_dir = os.path.join(OUTPUTS_ROOT, exp_id)
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "dataset_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(output_dir, "dataset_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_report_markdown(report))

    logger.info(f"\n{'='*50}")
    status_label = {"PASS": "PASS", "WARN": "WARNING", "FAIL": "FAIL"}[report["status"]]
    logger.info(f"[{status_label}] 数据诊断完成")
    if report["errors"]:
        logger.info(f"  ERROR ({report['error_count']}):")
        for e in report["errors"]:
            logger.info(f"    {e}")
    if report["warnings"]:
        logger.info(f"  WARNING ({report['warning_count']}):")
        for w in report["warnings"]:
            logger.info(f"    {w}")
    logger.info(f"  INFO ({report['info_count']})")
    logger.info(f"报告: {json_path}")
    logger.info(f"Markdown: {md_path}")
    logger.info(f"{'='*50}\n")

    # 严格模式下发现 ERROR 级别问题时退出码为 1
    if args.strict and report["error_count"] > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="atomlora",
        description="AtomLoRA - Lightweight LoRA fine-tuning framework",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- 子命令定义 ----
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--config", required=True, help="YAML config path")
    p_eval.add_argument("--data-path", help="Optional evaluation dataset path")
    p_eval.set_defaults(func=cmd_eval)

    p_predict = sub.add_parser("predict", help="Run single-text prediction")
    p_predict.add_argument("--config", required=True, help="YAML config path")
    p_predict.add_argument("--text", required=True, help="Text to predict")
    p_predict.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", help="Start FastAPI inference server")
    p_serve.add_argument("--config", required=True, help="YAML config path")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    p_split = sub.add_parser("split", help="Split raw JSONL into train/dev/test")
    p_split.add_argument("--input", help="Input JSONL file path (or use --config)")
    p_split.add_argument("--output", help="Output directory (default: same as input)")
    p_split.add_argument("--config", help="YAML config path (reads text_col, label_col, label_mapping)")
    p_split.add_argument("--text-col", help="Text field name in JSONL")
    p_split.add_argument("--label-col", help="Label field name in JSONL")
    p_split.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    p_split.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio (default: 0.1)")
    p_split.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_split.add_argument("--stratify", action="store_true", default=True, help="Stratified split by label (default: True)")
    p_split.add_argument("--no-stratify", action="store_false", dest="stratify", help="Disable stratified split")
    p_split.set_defaults(func=cmd_split)

    p_doctor = sub.add_parser("doctor-data", help="Diagnose dataset quality")
    p_doctor.add_argument("--config", required=True, help="YAML config path")
    p_doctor.add_argument("--strict", action="store_true", help="Exit 1 if any ERROR-level issue found")
    p_doctor.set_defaults(func=cmd_doctor)

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
