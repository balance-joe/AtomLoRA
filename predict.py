import json
import sys
import os
import argparse

from src.model.text_dataset import create_dataloader
from src.predict.predictor import TextAuditPredictor
# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger

def main(config_path):
    # 1. Config
    config = parse_config(config_path)
    exp_id = config["exp_id"]
    logger = init_logger(exp_id, config["task_type"])
    
    logger.info(f"加载配置，文件是 {config_path}")
    
    # 1. 创建预测器
    predictor = TextAuditPredictor(config=config)
    
    text = {"text":"创业指导中心和学院的大力支持。河南大学就业创业指导中心副主任王科、经济学院党委副书记李世平、副院长赵艳忠、教育部首批全国创新 [SEP] 王科 -> 王可 [SEP] 领导人错误 领导人姓名错误。领导人[王可]现任职：全国人大常委会委员、全国人大外事委员会副主任委员、中国红十字会党组书记、中国红十字会常务副会长。","is_misreport_label":1,"risk_label":0}
    
    # 2. 打印模型信息
    print("模型信息:", json.dumps(predictor.get_model_info(), ensure_ascii=False, indent=2))
    
    # 3. 单条预测
    sample = {"text": "这是一条测试文本"}
    result = predictor.predict(sample)
    print("单条预测结果:", json.dumps(result, ensure_ascii=False, indent=2))
    
    # 4. 批量预测
    batch_samples = [
        {"text": "测试文本1"},
        {"text": "测试文本2"},
        {"text": "测试文本3"}
    ]
    batch_results = predictor.predict_batch(batch_samples, batch_size=2)
    print("批量预测结果:", json.dumps(batch_results, ensure_ascii=False, indent=2))
    
    # 5. 释放资源
    predictor.close()
    
if __name__ == "__main__":
    # 可以用 argparse，这里为了简便直接调用
    # main(r"D:\python\AtomLoRA\configs\bert_text_audit_multi.yaml")
    # main(r"D:\python\AtomLoRA\configs\bert_text_audit_single.yaml")
    # main(r"D:\python\AtomLoRA\configs\ernie_text_audit_multi.yaml")
    main(r"D:\python\AtomLoRA\configs\ernie_text_audit_single.yaml")