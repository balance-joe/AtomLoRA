# src/predict/predictor.py
import os
import torch

from typing import Dict, List, Any, Union, Optional
from tqdm import tqdm
from src.utils.logger import get_logger
from src.model.model_factory import TaskTextClassifier, load_tokenizer
from peft import PeftModel
from src.utils.paths import resolve_adapter_path, resolve_classifier_path, resolve_tokenizer_path

# 与训练代码对齐的默认参数（统一管理）
DEFAULT_CONFIG = {
    "data": {
        "max_len": 256,
        "text_col": "text"
    },
    "model": {
        "arch": "bert-base-chinese",
        "fp16": False
    },
    "train": {
        "batch_size": 20
    },
    "task_type": "single_cls"
}


class TextAuditPredictor:
    """
    文本审计预测器 - 兼容单/多任务，与训练代码深度对齐
    """
    
    def __init__(self, config: Dict[str, Any], device: Union[str, torch.device, None] = None):
        """
        初始化预测器
        
        Args:
            config: 完整配置字典（与训练配置一致）
            device: 预测设备 (None=自动选择, "cuda:0"/"cpu"=手动指定)
        """
        # 合并配置与默认值（兜底）
        self.config = self._merge_config(config)
        self.exp_id = self.config['exp_id']
        
        # 灵活的设备配置
        self.device = self._setup_device(device)
        
        # 初始化日志
        self.logger = get_logger(f"predict_{self.exp_id}")
        
        # 实验产物目录
        self.exp_dir = os.path.join("outputs", self.exp_id)
        if not os.path.exists(self.exp_dir):
            raise FileNotFoundError(f"实验目录不存在: {self.exp_dir}")
        
        # 加载实验产物和模型
        self._load_experiment_artifacts()
        self._initialize_model()
        
        self.logger.info(f"✅ 预测器初始化完成 | 设备: {self.device} | 任务类型: {self.task_type}")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并用户配置与默认配置"""
        def recursive_merge(default: Dict, user: Dict) -> Dict:
            merged = default.copy()
            for k, v in user.items():
                if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                    merged[k] = recursive_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged
        
        return recursive_merge(DEFAULT_CONFIG, user_config)
    
    def _setup_device(self, device: Union[str, torch.device, None]) -> torch.device:
        """灵活设置计算设备"""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, torch.device):
            return device
        else:
            raise ValueError(f"无效的设备类型: {type(device)}")
    
    def _validate_config(self):
        """验证关键配置项"""
        required_keys = [
            ("exp_id", str),
            ("model", dict),
            ("data", dict)
        ]
        
        for key, expected_type in required_keys:
            if key not in self.config or not isinstance(self.config[key], expected_type):
                raise ValueError(f"配置缺失或类型错误: {key} (期望{expected_type.__name__})")
    
    def _load_experiment_artifacts(self):
        """加载实验产物（兼容训练代码的保存格式）"""
        try:
            # 验证配置
            self._validate_config()
            
            # 1. 确定任务类型
            self.task_type = self.config.get("task_type", "single_cls")
            
            # 2. 处理标签映射（对齐训练代码的任务名）
            self.label_map = self.config["data"]["label_mapping"]
            if self.task_type == "single_cls":
                # 单任务下：default对应第一个标签映射
                first_task_key = next(iter(self.label_map.keys()))
                self.label_map = {"default": self.label_map[first_task_key]}
            
            # 3. 加载tokenizer
            self.tokenizer = self._load_tokenizer()
            
            self.logger.info(f"📁 实验产物加载成功 | 任务类型: {self.task_type}")
            
        except Exception as e:
            self.logger.error(f"❌ 实验产物加载失败: {e}")
            raise
    
    def _load_tokenizer(self):
        """加载Tokenizer（优先加载训练好的，回退到配置）"""
        tokenizer_path = resolve_tokenizer_path(self.exp_dir)
        if os.path.exists(tokenizer_path):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.logger.warning("⚠️ 未找到训练好的Tokenizer，从配置加载")
            return load_tokenizer(self.config)

    def _initialize_model(self):
        """
        初始化模型（严格匹配训练代码结构）
        新增：模型初始化后校验LoRA权重完整性
        """
        try:
            # 1. 创建模型结构
            self.model = TaskTextClassifier(self.config, self.tokenizer)
            self.model.to(self.device)

            # 2. 加载模型权重（顺序：完整模型 -> LoRA -> 分类头）
            self._load_model_weights()

            # 3. 评估/预测模式
            self.model.eval()
            self.logger.info(f"✅ 模型初始化完成 | 设备: {self.device} | 任务类型: {self.task_type}")

            # ========== 新增：模型初始化后LoRA权重全局校验 ==========
            self.logger.info("🔍 开始全局LoRA权重校验...")
            try:
                from peft.utils import get_peft_model_state_dict
                # 尝试提取LoRA权重
                lora_state_dict = get_peft_model_state_dict(self.model.bert)
                self.logger.info(f"📊 LoRA权重键总数: {len(lora_state_dict.keys())}")

                # 打印前5个LoRA权重键（直观验证）
                # if lora_state_dict:
                    # top5_keys = list(lora_state_dict.keys())[:5]
                    # for idx, k in enumerate(top5_keys):
                        # self.logger.info(f"   第{idx + 1}个LoRA权重键: {k}")
                # else:
                #     self.logger.error("❌ 未提取到任何LoRA权重键！")

            except ImportError:
                self.logger.error("❌ 缺少peft库，无法提取LoRA权重！")
            except AttributeError as e:
                self.logger.error(f"❌ 模型无LoRA相关属性: {e}")
            except Exception as e:
                self.logger.error(f"❌ 提取LoRA权重失败: {str(e)}")
            # ======================================================

        except Exception as e:
            self.logger.error(f"❌ 模型初始化失败: {e}")
            raise


    def _load_model_weights(self, torch=None):
        """
        加载训练好的模型权重（PEFT 标准 LoRA 模式）
        顺序：
            1️⃣ LoRA 适配器 (adapter_model.safetensors/bin)
            2️⃣ 分类头 (classifiers.pt)

        关键修正：
        1. 兼容加载 .bin 和 .safetensors 格式。
        2. 移除 model.pt 加载。
        3. 关键：若 self.model.bert 已是 PeftModel (通过 _inject_lora)，则使用 load_adapter 加载权重。
        """

        # 1. 移除 model.pt 加载：
        self.logger.info("ℹ️ LoRA模式，跳过完整模型 model.pt 的加载。")

        # 2. LoRA 适配器加载
        lora_path = resolve_adapter_path(self.exp_dir)

        # 核心修正：检查 adapter_model.safetensors 和 adapter_model.bin
        adapter_bin_path = os.path.join(lora_path, "adapter_model.bin")
        adapter_safetensors_path = os.path.join(lora_path, "adapter_model.safetensors")

        # 检查适配器文件是否存在
        if os.path.exists(adapter_safetensors_path) or os.path.exists(adapter_bin_path):
            try:
                from peft import PeftModel

                # 【关键逻辑修正】
                # 如果 TaskTextClassifier 已经在 __init__ 中注入了 LoRA 结构 (PeftModel)，
                # 我们只需要加载权重到这个已存在的结构中，避免双重包装。
                if not isinstance(self.model.bert, PeftModel):
                    self.logger.error("❌ 模型加载错误：self.model.bert 不是 PeftModel 实例。")
                    self.logger.error("请检查 Predictor/Evaluator 初始化逻辑，确保 LoRA 结构已通过 _inject_lora 注入。")
                    # 如果不是 PeftModel，则尝试使用 from_pretrained 进行包装（这是为了兼容旧逻辑，但会产生警告）
                    # 确保 dtype 匹配
                    dtype_config = self.config["train"].get("mixed_precision")
                    dtype = torch.float16 if dtype_config in ["fp16"] else \
                        torch.bfloat16 if dtype_config in ["bf16"] else torch.float32

                    self.logger.warning(
                        "⚠️ 模型未预先包装。尝试使用 PeftModel.from_pretrained 进行加载和包装，可能产生 'Found missing adapter keys' 警告。")
                    self.model.bert = PeftModel.from_pretrained(
                        self.model.bert,
                        lora_path,
                        torch_dtype=dtype
                    )
                else:
                    # 推荐的加载方式：使用 load_adapter 加载权重到已存在的 PeftModel 槽位
                    self.logger.info("✅ 检测到 PeftModel 实例，使用 load_adapter 加载权重。")
                    self.model.bert.load_adapter(lora_path, adapter_name="default")

                # ========== LoRA权重精细化校验 ==========
                self.logger.info("🔍 开始LoRA权重有效性校验...")

                lora_A_count = 0
                lora_B_count = 0
                invalid_lora_A = 0
                untrained_lora_B = 0

                # 遍历所有参数进行校验
                for name, param in self.model.bert.named_parameters():
                    if "lora_A" in name:
                        lora_A_count += 1
                        # 检查 lora_A 是否接近零（加载失败的迹象）
                        if param.mean().abs().cpu().item() < 1e-8:
                            invalid_lora_A += 1
                            self.logger.warning(
                                f"⚠️ LoRA权重 {name} 均值接近零 ({param.mean().abs().cpu().item():.8f})，可能加载失败。")

                    elif "lora_B" in name:
                        lora_B_count += 1
                        # 检查 lora_B 是否接近零（训练不充分的警示）
                        if param.mean().abs().cpu().item() < 1e-6:
                            untrained_lora_B += 1

                # 总结与判断
                if lora_A_count == 0:
                    self.logger.error("❌ LoRA适配器加载失败：未检测到任何 lora_A 权重！")
                elif invalid_lora_A > 0:
                    self.logger.error(f"❌ LoRA适配器加载失败：{invalid_lora_A} 个 lora_A 权重异常，请检查训练保存逻辑！")
                else:
                    self.logger.info("✅ LoRA适配器加载成功（所有 lora_A 权重有效）")
                    if untrained_lora_B == lora_B_count and lora_B_count > 0:
                        self.logger.warning(
                            f"⚠️ LoRA-B ({lora_B_count}个) 权重均值仍接近 0，模型可能欠拟合或训练步数过少。")

                # ==============================================

            except ImportError:
                self.logger.error("❌ LoRA加载失败：缺少peft库，请执行 pip install peft")
            except Exception as e:
                self.logger.error(f"⚠️ LoRA适配器加载失败: {str(e)}")
        else:
            self.logger.warning(f"⚠️ 标准PEFT适配器文件 (adapter_model.bin 或 .safetensors) 不存在于 {lora_path}")

        # 3. 分类头加载
        classifier_path = resolve_classifier_path(self.exp_dir)
        if os.path.exists(classifier_path):
            try:
                # 【关键修复】确保 torch 在此作用域可用
                import torch

                clf_state = torch.load(classifier_path, map_location=self.device)
                self.model.classifiers.load_state_dict(clf_state)
                self.logger.info("✅ 分类头权重加载成功")
            except Exception as e:
                self.logger.error(f"⚠️ 分类头加载失败: {e}")
        else:
            self.logger.warning(f"⚠️ 分类头权重文件 {classifier_path} 不存在。")
        # 确保模型在设备上
        self.model.to(self.device)
        self.model.eval()


    def predict(self, data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测单条数据
        
        Args:
            data_sample: 包含text字段的字典
            
        Returns:
            Dict: 预测结果（含标签、置信度等）
        """
        # 输入验证
        text_col = self.config["data"]["text_col"]
        if text_col not in data_sample:
            raise ValueError(f"数据缺少必要字段: {text_col}")
        
        text = data_sample[text_col]
        
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config["data"]["max_len"],
            return_tensors='pt',
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, labels=None)

        # 后处理结果
        return self._postprocess_result(outputs, data_sample)
    
    def predict_batch(self, data_samples: List[Dict[str, Any]],
                     batch_size: int = None, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        高效批量预测（GPU并行加速）

        Args:
            data_samples: 数据样本列表
            batch_size: 批次大小（None=使用训练配置的batch_size）
            show_progress: 是否显示进度条

        Returns:
            List[Dict]: 批量预测结果
        """
        if not data_samples:
            self.logger.warning("⚠️ 空数据样本列表")
            return []

        # 使用训练配置的batch_size作为默认值
        batch_size = batch_size or self.config["train"]["batch_size"]
        all_results = []
        text_col = self.config["data"]["text_col"]

        # 进度条
        iterator = range(0, len(data_samples), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"批量预测 (batch={batch_size})")

        self.model.eval()
        with torch.no_grad():
            for batch_start in iterator:
                batch_end = batch_start + batch_size
                batch_samples = data_samples[batch_start:batch_end]

                # 提取批量文本
                batch_texts = []
                valid_indices = []
                for idx, sample in enumerate(batch_samples):
                    if text_col in sample:
                        batch_texts.append(sample[text_col])
                        valid_indices.append(idx)
                    else:
                        self.logger.warning(f"跳过无效样本 {batch_start+idx}: 缺少{text_col}字段")
                        all_results.append({"error": f"缺少{text_col}字段", "sample": sample})

                if not batch_texts:
                    continue

                # 批量Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config["data"]["max_len"],
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # 批量前向传播
                outputs = self.model(input_ids, attention_mask, labels=None)

                # 批量后处理
                for idx, text_idx in enumerate(valid_indices):
                    sample = batch_samples[text_idx]
                    try:
                        # 提取单条输出
                        single_output = self._extract_single_output(outputs, idx)
                        result = self._postprocess_result(single_output, sample)
                        all_results.append(result)
                    except Exception as e:
                        error_msg = f"样本{batch_start+text_idx}处理失败: {str(e)}"
                        self.logger.warning(error_msg)
                        all_results.append({"error": error_msg, "sample": sample})

        return all_results

    def _extract_single_output(self, outputs: Union[Dict[str, torch.Tensor], torch.Tensor], idx: int):
        """从批量输出中提取单条结果"""
        if isinstance(outputs, dict):
            return {k: v[idx:idx+1] for k, v in outputs.items()}
        else:
            return outputs[idx:idx+1]

    def _postprocess_result(self, outputs: Union[Dict[str, torch.Tensor], torch.Tensor], 
                           data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """后处理预测结果（兼容单/多任务）"""
        if self.task_type == "single_cls":
            return self._process_single_task(outputs, data_sample)
        else:
            return self._process_multi_task(outputs, data_sample)
    
    def _process_single_task(self, outputs: Union[Dict[str, torch.Tensor], torch.Tensor], 
                            data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """处理单任务预测结果"""
        # 获取logits
        logits = outputs.get("default", outputs) if isinstance(outputs, dict) else outputs
        
        # 计算概率和预测
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).cpu().item()
        confidence = probs[0][pred_idx].cpu().item()
        
        # 获取标签文本
        label_map = self.label_map["default"]
        pred_label = label_map.get(str(pred_idx), label_map.get(pred_idx, str(pred_idx)))
        
        # 构建结果
        result = {
            "text": data_sample.get(self.config["data"]["text_col"], ""),
            "prediction": pred_label,
            "label_id": int(pred_idx),
            "confidence": round(float(confidence), 4),
            "probabilities": {}
        }
        
        # 输出所有类别概率（最多显示10个）
        for idx, prob in enumerate(probs[0].cpu().numpy()):
            label = label_map.get(str(idx), label_map.get(idx, str(idx)))
            result["probabilities"][label] = round(float(prob), 4)
            if idx >= 9:
                break
        
        # 添加原始数据（可选）
        if "raw_data" in data_sample:
            result["raw_data"] = data_sample["raw_data"]
        
        return result
    
    def _process_multi_task(self, outputs: Dict[str, torch.Tensor], 
                           data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """处理多任务预测结果"""
        if not isinstance(outputs, dict):
            raise ValueError("多任务模型必须返回字典格式输出")
        
        result = {
            "text": data_sample.get(self.config["data"]["text_col"], ""),
            "tasks": {}
        }
        
        for task_name, logits in outputs.items():
            if task_name not in self.label_map:
                continue
            
            # 计算概率和预测
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).cpu().item()
            confidence = probs[0][pred_idx].cpu().item()
            
            # 获取任务标签
            task_label_map = self.label_map[task_name]
            pred_label = task_label_map.get(str(pred_idx), task_label_map.get(pred_idx, str(pred_idx)))
            
            # 任务结果
            result["tasks"][task_name] = {
                "prediction": pred_label,
                "label_id": int(pred_idx),
                "confidence": round(float(confidence), 4),
                "probabilities": {}
            }
            
            # 任务概率
            for idx, prob in enumerate(probs[0].cpu().numpy()):
                label = task_label_map.get(str(idx), task_label_map.get(idx, str(idx)))
                result["tasks"][task_name]["probabilities"][label] = round(float(prob), 4)
                if idx >= 9:
                    break
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "exp_id": self.exp_id,
            "task_type": self.task_type,
            "model_arch": self.config["model"]["arch"],
            "max_length": self.config["data"]["max_len"],
            "label_mapping": self.label_map,
            "device": str(self.device),
            "lora_enabled": hasattr(self.model.bert, "peft_config")
        }
    
    def close(self):
        """清理资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🔌 预测器资源已释放")
