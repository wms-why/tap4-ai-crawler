import os
import logging
from transformers import AutoTokenizer
from util.common_util import CommonUtil
import replicate

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
util = CommonUtil()
# 初始化LLaMA模型的Tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

class LLMUtil:
    def __init__(self):
        self.detail_sys_prompt = os.getenv('DETAIL_SYS_PROMPT')
        self.tag_selector_sys_prompt = os.getenv('TAG_SELECTOR_SYS_PROMPT')
        self.language_sys_prompt = os.getenv('LANGUAGE_SYS_PROMPT')
        self.replicate_model = os.getenv('REPLICATE_MODEL')
        self.replicate_max_tokens = int(os.getenv('REPLICATE_MAX_TOKENS', 8000))

    def process_detail(self, user_prompt):
        logger.info("正在处理Detail...")
        return util.detail_handle(self.process_prompt(self.detail_sys_prompt, user_prompt))

    def process_tags(self, user_prompt):
        logger.info(f"正在处理tags...")
        result = self.process_prompt(self.tag_selector_sys_prompt, user_prompt)
        # 将result（逗号分割的字符串）转为数组
        if result:
            tags = [element.strip() for element in result.split(',')]
        else:
            tags = []
        logger.info(f"tags处理结果:{tags}")
        return tags

    def process_language(self, language, user_prompt):
        logger.info(f"正在处理多语言:{language}, user_prompt:{user_prompt}")
        # 如果language 包含 English字符，则直接返回
        if 'english'.lower() in language.lower():
            result = user_prompt
        else:
            result = self.process_prompt(self.language_sys_prompt.replace("{language}", language), user_prompt)
            if result and not user_prompt.startswith("#"):
                # 如果原始输入没有包含###开头的markdown标记，则去掉markdown标记
                result = result.replace("### ", "").replace("## ", "").replace("# ", "").replace("**", "")
        logger.info(f"多语言:{language}, 处理结果:{result}")
        return result

    def process_prompt(self, sys_prompt, user_prompt):
        if not sys_prompt:
            logger.info(f"LLM无需处理，sys_prompt为空:{sys_prompt}")
            return None
        if not user_prompt:
            logger.info(f"LLM无需处理，user_prompt为空:{user_prompt}")
            return None

        logger.info("LLM正在处理")
        try:
            tokens = tokenizer.encode(user_prompt)
            if len(tokens) > self.replicate_max_tokens:
                logger.info(f"用户输入长度超过{self.replicate_max_tokens}，进行截取")
                truncated_tokens = tokens[:self.replicate_max_tokens]
                user_prompt = tokenizer.decode(truncated_tokens)
            r = []
            for event in replicate.stream(
                "meta/meta-llama-3-8b-instruct",
                input={
                    "top_k": 0,
                    "top_p": 0.95,
                    "prompt": user_prompt,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "system_prompt": sys_prompt,
                    "length_penalty": 1,
                    "max_new_tokens": 512,
                    "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "presence_penalty": 0,
                    "log_performance_metrics": False
                },
            ):
                r.append(str(event))
            
            return "".join(r)
        except Exception as e:
            logger.error(f"LLM处理失败", e)
            return None