# main.py
import argparse
import sys
import os

from src.model.model_factory import load_tokenizer

# 添加项目根目录到Python路径（确保模块导入）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.parser import parse_config
from src.utils.logger import init_logger, get_logger
from src.data.data_processor import load_dataset  # 模拟数据模块


def main(config_path: str):
    try:
        # ====================== 1. 解析配置 ======================
        print(f"[Start] Loading config from: {config_path}")
        config = parse_config(config_path)
        exp_id = config["exp_id"]
        task_type = config["task_type"]
        
        # # ====================== 2. 初始化实验日志 ======================
        logger = init_logger(experiment_id=exp_id, task_type=task_type)
        logger.info("="*50 + f" Experiment {exp_id} Start " + "="*50)
        logger.info(f"Task type: {task_type}")
        logger.info(f"Model arch: {config['model']['arch']}")
        logger.info(f"Train epochs: {config['train']['num_epochs']}")
        logger.debug(f"Full config: {config}")  # DEBUG级别仅记录到文件
        
      # ====================== 3. 加载Tokenizer ======================
        tokenizer = load_tokenizer(config)  # 从配置加载Tokenizer
        
           # 3. 示例文本（来自你的数据集）
        sample_texts = [
            "“刷单返利”，从“虚假投资”到“杀猪盘”，从传统的电话、短信[ERROR]广[\/ERROR]撒网式诈骗发展到利用AI换脸、语音克隆等高科技手段实施精准诈 [SEP] 广 -> 广告 [SEP] 疑似错误",
            "密[ERROR]的[\/ERROR]树林，在蔚蓝的天空下，太阳光从树叶的缝隙中透出来把树叶照的透 [SEP] 的 -> 得 [SEP] 的地得错误",
            "他无法分[ERROR]辨[/ERROR]是非，总是说错话 [SEP] 辨 -> 辩 [SEP] 错别字"
        ]
        
        # 4. 逐个Tokenize并展示结果
        for idx, text in enumerate(sample_texts):
            print(f"\n===== 示例文本 {idx+1} Tokenize结果 =====")
            print(f"原始文本：{text[:60]}...（总字符数：{len(text)}）")
            
            # Tokenize（按你的配置参数）
            encoded = tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=config["data"]["max_len"],
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # 转换为token列表（查看切割细节）
            input_ids = encoded["input_ids"].squeeze(0).tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoded["attention_mask"].squeeze(0).tolist()
            
            # 展示前60个有效token（非padding）
            print("\nToken切割详情（前60个有效token）：")
            valid_tokens = [(i, tok, idx_) for i, (tok, idx_, mask) in enumerate(zip(tokens, input_ids, attention_mask)) if mask == 1][:60]
            for i, tok, idx_ in valid_tokens:
                # 高亮特殊token
                if tok in ["[ERROR]", "[/ERROR]", "[SEP]", "NOSUGGEST"]:
                    print(f"  [{i}] \033[32m{tok:<12}\033[0m (ID: {idx_:<4})")
                else:
                    print(f"  [{i}] {tok:<12} (ID: {idx_:<4})")
            
            # 统计特殊token
            special_token_stats = {
                "[ERROR]": tokens.count("[ERROR]"),
                "[/ERROR]": tokens.count("[/ERROR]"),
                "[SEP]": tokens.count("[SEP]"),
                "有效token长度": sum(attention_mask)
            }
            print(f"\n特殊token统计：{special_token_stats}")

        
        # # ====================== 3. 数据加载 ======================
        # logger.info("加载训练，测试，评估数据集...")
        # train_data = load_dataset(
        #     config, 
        #     path=config['data']['test_path'],
        #     tokenizer=tokenizer
        # )
        # print(train_data)
        # dev_data = load_dataset(config, split="dev")
        # test_data = load_dataset(config, split="test")
        # logger.info(f"Data loaded: train({len(train_data)}), dev({len(dev_data)}), test({len(test_data)})")
        # # ====================== 4. 模型构建 ======================
        
        # logger.info("="*50 + f" Experiment {exp_id} Finish " + "="*50)
        
    except Exception as e:
        # 异常日志记录（含堆栈）
        if 'logger' in locals():
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        else:
            print(f"Error before logger init: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":

    # 启动实验
    main(r"D:\python\AtomLoRA\configs\text_audit.yaml")
    
    # 新数据格式
    # {"text":"他无法分[ERROR]辨[\/ERROR]是非，总是说错话 [SEP] 辨 -> 辩 [SEP] 错别字","label_is_misreport":0,"label_risk_level":1}