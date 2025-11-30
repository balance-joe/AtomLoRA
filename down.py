from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,BertTokenizer, AutoConfig,AutoModelForSequenceClassification , AutoModel



# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# # 官方模型名 + 本地保存路径
# model_name = "nghuyong/ernie-3.0-base-zh"
# save_path = r"D:\python\AtomLoRA\models\ernie-3.0-base-zh"

# # 下载并保存分词器和MLM模型（官方原版）
# tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
# model = AutoModelForMaskedLM.from_pretrained(model_name)
# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path)

# # return 


# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# # 本地路径（用原始字符串避免转义）
# ernie_model_path = r"D:\python\AtomLoRA\models\ernie-3.0-base-zh"
# bert_model_path = r"D:\python\AtomLoRA\models\bert-base-chinese"

# # 关键：加载MaskedLM模型（带掩码预测head），并设置分词器参数
# ernie_tokenizer = AutoTokenizer.from_pretrained(ernie_model_path, do_lower_case=False)
# ernie_model = AutoModelForMaskedLM.from_pretrained(ernie_model_path)  # 必须是ForMaskedLM

# bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
# bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_path)

# # 推理模式+禁用梯度
# ernie_model.eval()
# bert_model.eval()
# torch.set_grad_enabled(False)

# def predict_mask(text, tokenizer, model, top_k=5):
#     """修复预测逻辑：确保MASK位置正确识别"""
#     inputs = tokenizer(
#         text, 
#         return_tensors="pt", 
#         padding=True, 
#         truncation=True
#     )
#     # 找到所有MASK的位置
#     mask_positions = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
#     if len(mask_positions) == 0:
#         return ["未找到[MASK]标记"]
    
#     # 模型预测
#     outputs = model(**inputs)
#     logits = outputs.logits
    
#     # 提取每个MASK位置的top-k结果
#     results = []
#     for pos in mask_positions:
#         top_tokens = torch.topk(logits[0, pos, :], top_k).indices.tolist()
#         results.extend([tokenizer.decode([t]).strip() for t in top_tokens])
#     return results

# # ========== 测试案例1：单字掩码 ==========
# text1 = "北京是中国的[MASK]都"
# print("===== 单字掩码测试：", text1)
# print("ERNIE预测结果：", predict_mask(text1, ernie_tokenizer, ernie_model))
# print("BERT预测结果：", predict_mask(text1, bert_tokenizer, bert_model))

# # ========== 测试案例2：实体掩码 ==========
# text2 = "[MASK][MASK]是唐代著名诗人"
# print("\n===== 实体掩码测试：", text2)
# print("ERNIE预测结果：", predict_mask(text2, ernie_tokenizer, ernie_model))
# print("BERT预测结果：", predict_mask(text2, bert_tokenizer, bert_model))

# # ========== 测试案例3：短语掩码 ==========
# text3 = "我喜欢学习[MASK][MASK][MASK][MASK]技术"
# print("\n===== 短语掩码测试：", text3)
# print("ERNIE预测结果：", predict_mask(text3, ernie_tokenizer, ernie_model))
# print("BERT预测结果：", predict_mask(text3, bert_tokenizer, bert_model))

# # ========== 测试案例4：知识关联掩码 ==========
# text4 = "珠穆朗玛峰是世界上最高的[MASK]"
# print("\n===== 知识关联掩码测试：", text4)
# print("ERNIE预测结果：", predict_mask(text4, ernie_tokenizer, ernie_model))
# print("BERT预测结果：", predict_mask(text4, bert_tokenizer, bert_model))