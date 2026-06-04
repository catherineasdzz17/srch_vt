import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HunyuanMT:
    def __init__(self):
        # model_name = "model/hunyuan-mt"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        # self.model = self.model.eval()

        model_name = "model/hunyuan-mt"
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        print("加载Hunyuan模型（8bit量化）...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model = self.model.eval()
        
        print("Hunyuan加载完成")

    
    def trans2zh(self, src_text: str) -> str:
        """
        调用Hunyuan-MT-7B模型将输入文本翻译为中文
        """
        messages = [
            {
                "role": "system", 
                "content": "你是专业翻译助手，将任何语言翻译为简体中文。直接输出译文，不要重复指令或原文。"
            },
            {
                "role": "user", 
                "content": src_text
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs, 
            max_new_tokens=2048,
            do_sample=False
        )
        new_tokens = outputs[0][inputs.shape[1]:]
        trans_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return trans_text