import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QWEN25:
    def __init__(self):
        # model_name = "model/qwen2.5"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        # self.model = self.model.eval()
        model_name = "model/qwen2.5"
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        print("加载Qwen模型（8bit量化）...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model = self.model.eval()
        
        print("Qwen加载完成")
    

    # def generate_elements_str(self, query_sentence: str) -> str:
    #     """
    #     通过包装Prompt，调用Qwen2.5-7B-Instruct模型生成三元组的结构化信息
    #     Args:
    #         query_sentence(str): 输入的查询句子
    #     Returns:
    #         str: 包含三元组的结构化字符串
    #     """

    #     prompt_template = """
    #     以下是一个查询句子，请从中提取涉及的实体信息，按严格按照以下格式返回：
    #     [实体类别]:[实体名称];[实体类别]:[实体名称]。
    #     请注意，所有相同实体请你合并同类项之后，再分别输出
    #     实体类别包括：
    #     - 人物（如国家领导人、公众人物等）。
    #     - 组织机构（如公司、政府、国际组织等）。
    #     - 国家或地区。
    #     - 事件（如特定行动、活动等）。

    #     示例：
    #     查询句子：特朗普将在2024年访问乌克兰，与泽连斯基讨论能源合作。
    #     输出：
    #     人物:特朗普，泽连斯基; 国家或地区:乌克兰; 事件:能源合作。

    #     现在请处理以下查询句子：
    #     查询句子：{query_sentence}
    #     输出：
    #     """

    #     # 将输入句子嵌入到Prompt中
    #     prompt = prompt_template.format(query_sentence=query_sentence)

    #     messages = [
    #         {"role": "system", "content": "你的任务是信息提取，请严格按照要求的格式进行输出，不要输出多于的任何内容"},
    #         {"role": "user", "content": prompt}
    #     ]
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

    #     generated_ids = self.model.generate(
    #         **model_inputs,
    #         max_new_tokens=512
    #     )
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]

    #     response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    #     if response == "无识别结果":
    #         return None
        
    #     return response.replace('\n',';')


    def generate_elements_str(self, query_sentence: str) -> str:
        import re
        """
        调用 Qwen2.5 模型提取实体三元组。
        """

        prompt_template = """
        以下是一个查询句子，请从中提取涉及的实体信息，按严格格式返回：
        [实体类别]:[实体名称];[实体类别]:[实体名称]。
        请注意，同类合并，保持原文，不要扩写或杜撰。

        实体类别包括：
        - 人物（如国家领导人、公众人物等）。
        - 组织机构（如公司、政府、国际组织等）。
        - 国家或地区。
        - 事件（如特定行动、活动等）。

        输出规则（二选一，不要输出额外内容、不加解释、不换行）：
        1) 若识别到至少 1 个实体，按以下格式返回（未涉及的类别省略）：
        人物:xxx;组织机构:xxx;国家或地区:xxx;事件:xxx。
        2) 若未识别到任何实体，只返回：
        无

        现在请处理以下查询句子：
        查询句子：{query_sentence}
        输出：
        """.strip()

        prompt = prompt_template.format(query_sentence=query_sentence)

        messages = [
            {"role": "system", "content": "你的任务是信息提取，请严格按照要求的格式进行输出，不要输出多余内容"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None)
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception:
            return "无"

        # 去掉可能的代码块包裹
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())

        # 模型输出“无”或空串
        if not response or response == "无" or "无" in response:
            return "无"

        # 只接受“类别:内容”对
        cleaned = response.rstrip("。.")
        parts = [p.strip() for p in cleaned.split(";") if p.strip()]
        if not parts:
            return "无"

        valid_cats = {"人物", "组织机构", "国家或地区", "事件"}
        kept_pairs = []

        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
            elif "：" in p:
                k, v = p.split("：", 1)
            else:
                continue

            k = k.strip()
            v = v.strip().strip("；;，, ")
            if not k or not v or k not in valid_cats:
                continue

            # 仅保留原句中真实出现的实体，去重保序
            ents = [e.strip() for e in re.split(r"[，,、\s]+", v) if e.strip()]
            seen = set()
            ents_in_query = []
            for e in ents:
                if e in query_sentence and e not in seen:
                    seen.add(e)
                    ents_in_query.append(e)

            if ents_in_query:
                kept_pairs.append(f"{k}:{'，'.join(ents_in_query)}")

        if not kept_pairs:
            return "无"

        # 与版本一格式保持一致：分号无空格、不强制句号
        return ";".join(kept_pairs).replace("\n", ";")

    
    def judge_sim_tra(self, sim_que, tra_que):
        """
        判断两个句子哪个更通顺
        Args:
            sim_que(str), tra_que(str): 输入的查询句子
        Returns:
            str: 返回更准确的文本句子
        """

        prompt_template = f"""
        给你两个句子，分别是用简体和繁体的ocr模型从一张图片上ocr下来的
        首先，如果有一句话不是中文的（比如说None这种），你可以直接返回另一句子的标号（1或者2）
        两个给你的句子都是简体中文，但是其中会有一句十分不通顺，你需要选择更通顺的那一句话
        你只需要返回1或者2，表明你认为是第一句还是第二句更通顺，不需要返回任何其他内容

        示例1：
        句子1：乌克兰仍在持续抵抗
        句子2：鸟克兰扔在持续抵抗
        输出：1

        示例2：
        句子1：None
        句子2：哈里斯承认败选
        输出：2
        解释：句子1是None，因此直接返回2

        现在请回答下列句子对：
        句子1：{sim_que}
        句子2：{tra_que}
        输出：
        """

        prompt = prompt_template.format(sim_que=sim_que, tra_que=tra_que)

        messages = [
            {"role": "system", "content": "你是一个语言学家，你的任务是句子对判断，请严格按照示例输出"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return sim_que if response=='1' else tra_que


    def add_punctuation(self, raw_text):
        """
        为文字分段并添加标点
        Args:
            raw_text(str): 输入的文本
        Returns:
            str: 处理后的文本
        """

        prompt = f"""
        请将以下文字按照语意停顿适当分段，并补充合适的标点符号（如逗号、句号等）。注意：
        1. 内容不得修改，仅调整停顿并放置标点符号；
        2. 确保标点符号使用正确，不增减文字内容。

        文字如下：
        {raw_text}

        输出格式：
        - 标点符号正确且符合语意停顿；
        - 内容完整无误。
        """

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    def judge_similarity(self, a, b):
        """
        判断两个句子是否匹配
        Args:
            a(str), b(str): 输入的句子
        Returns:
            bool: True/False
        """

        prompt_template = f"""
        给你两个句子，第一个句子是输入的检索内容，第二句是得到的检索结果，你需要判断第二句是否可以作为第一句的检索结果
        请注意，在我们的场景里，只要两段文字包含有相同的人物/国家/事件，就可以视为正例，你的判定可以宽松一些
        如果可以就回答True，否则就回答False，请注意，你只允许输出这两个词，任何其他内容都不需要

        示例1：
        句子1：乌克兰仍在持续抵抗
        句子2：俄罗斯正在高歌猛进进军基辅
        输出：True

        示例2：
        句子1：台湾选举结束
        句子2：哈里斯承认败选
        输出：False

        现在请回答下列句子对：
        句子1：{a}
        句子2：{b}
        输出：
        """

        prompt = prompt_template.format(sim_que=a, tra_que=b)

        messages = [
            {"role": "system", "content": "你是一个语言学家，你的任务是句子对进行判断，请严格按照示例输出"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        return eval(response)


    def generate_yinyu_fengci(self, query_sentence: str) -> str:
        """
        识别输入文本中的隐喻讽刺内容
        Args:
            query_sentence(str): 输入的文本
        Returns:
            str: 结构化的隐喻讽刺
        """

        prompt_template = """
        你是一位高级自然语言理解助手，专注于分析文本中的修辞手法。你的任务是从以下文本中判断是否含有“隐喻”或“讽刺”特点的实体，如果有的话请分类为以下两种类型之一：  
        1. 隐喻性实体：通过某种形式的比喻或象征性表达，间接描述实际事物的词语或短语。  
        2. 讽刺性实体：以反语、嘲讽、或幽默方式表达批评或否定意义的词语或短语。  
        ### 输入文本：  
        {query_sentence}
        
        ### 输出要求：  
        
        1. 对文本中所有含有“隐喻”和“讽刺”特征的实体进行识别。
        2. 对于每个隐喻或讽刺，明确指出其指代的真实人物、组织或群体。
        3. 识别的隐喻可能不仅限于直接的人物或事物，可能还涉及社会现象、政治事件等
        4. 讽刺表达通常是通过对比、反语或夸张等手法批评或讥讽某个对象
        5. 注意只需要识别出含有“隐喻”或“讽刺”特点的实体或元素
        
        ### 输出格式：  
        - 类别：隐喻 / 讽刺  \n
        - 实体：[具体的实体]  \n

        ### 如果有隐喻或讽刺实体请严格按照输出格式进行输出，如果没有隐喻讽刺实体直接输出:"无隐喻或讽刺实体"
        """
        
        # 将输入句子嵌入到Prompt中
        prompt = prompt_template.format(query_sentence=query_sentence)

        messages = [
            {"role": "system", "content": "你是一位高级自然语言理解助手，专注于分析文本中的修辞手法，请严格按照要求的格式进行输出，不要输出多于的任何内容"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.replace('\n', ';')


    def generate_summary(self, query_sentence, elements):
        """
        根据视频文本和视觉元素生成报告
        """
        import torch
        import gc
        
        # ============ 1. 推理前显存检查和清理 ============
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 检查可用显存
        free_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory -= torch.cuda.memory_reserved(0)
        free_gb = free_memory / 1024**3
        
        print(f"[显存检查] 推理前可用: {free_gb:.2f} GB")
        
        if free_gb < 2.5:
            print(f"显存不足 ({free_gb:.2f} GB)，强制清理...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # ============ 2. 处理输入文本（不压缩）============
        compressed_text = query_sentence if query_sentence else "（无文本内容）"
        
        # ============ 3. 处理视觉元素 ============
        element_list_text = ""
        if elements and isinstance(elements, list):
            element_counter = {}
            
            # 只处理前30个元素
            for elem in elements[:30]:
                if not elem or not isinstance(elem, dict):
                    continue
                
                elem_names = elem.get("elements", [])
                
                for name in elem_names:
                    if name:
                        element_counter[name] = element_counter.get(name, 0) + 1
            
            # 取前15个高频元素
            top_elements = sorted(element_counter.items(), key=lambda x: x[1], reverse=True)[:15]
            
            if top_elements:
                element_names = [name for name, _ in top_elements]
                element_list_text = "、".join(element_names)
        
        # ============ 4. 构建 Prompt ============
        prompt = f"""请对以下视频内容进行详细的客观描述。

        【文本内容】
        {compressed_text}

        【视觉元素】
        {element_list_text if element_list_text else "（本视频未识别到明确的视觉元素标签）"}

        【要求】
        1. **字数要求：至少200字，建议200-300字**
        2. **内容要求如下：**
        - 第一段：概述视频的核心主题和主要内容（约80-100字）
        - 第二段：详细说明文本中提到的关键信息、数据、观点或事实（约80-100字）
        - 第三段：详细列出视频中出现的视觉元素，并说明它们的作用与内容的关联（约80-100字）
        3. 使用客观陈述句，避免"精彩"、"重要"、"有趣"等主观评价
        4. 保持中立专业的叙述语气

        请直接输出描述内容（不要包含"描述："等前缀）："""

        messages = [
            {"role": "system", "content": "你是专业的视频内容分析员。你的任务是对视频进行客观、详细的描述，字数必须在200字以上，建议200-300字。描述要分段、有条理、信息丰富。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # ============ 5. 应用模板 ============
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # ============ 6. Tokenize（关键修改）============
            model_inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=3000,
                padding=False
            ).to(self.model.device)
            
            input_length = model_inputs.input_ids.shape[1]
            print(f"[报告生成] 输入tokens: {input_length}")
            
            # ============ 7. 生成（关键修改）============
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # ============ 8. 解码 ============
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            print(f"[报告生成] 输出长度: {len(response)}字")
            
            # ============ 9. 立即清理显存 ============
            del model_inputs, generated_ids, output_ids, text
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            return response if response else ""
        
        except Exception as e:
            print(f"[报告生成] 错误: {type(e).__name__}: {e}")
            
            # 清理显存
            if 'model_inputs' in locals():
                del model_inputs
            if 'generated_ids' in locals():
                del generated_ids
            gc.collect()
            torch.cuda.empty_cache()
            
            return ""


    def generate_summary_chunk(self, query_sentence, elements):
        """
        提取关键信息，仅用于长文本分块压缩时的中间步骤
        """
        import torch
        import gc
        
        # 推理前清理
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 截断输入文本（chunk 方法只需要提取关键信息）
        text_input = query_sentence[:1000] if len(query_sentence) > 1000 else query_sentence
        
        prompt = f"""请从以下文本中提取3-5个最重要的关键信息点：

        {text_input}

        要求：
        - 只提取核心观点、数据、事实
        - 每个信息点30-50字
        - 保持客观陈述

        关键信息："""

        messages = [
            {"role": "system", "content": "你是信息提取助手。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=1500, 
                padding=False
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=250,
                    temperature=0.5,
                    do_sample=False,
                    use_cache=False,  
                    num_beams=1,      
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            output_ids = outputs[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # 立即清理
            del model_inputs, outputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            return response if response else query_sentence[:200]
        
        except Exception as e:
            print(f"[Chunk摘要] 错误: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return query_sentence[:200]