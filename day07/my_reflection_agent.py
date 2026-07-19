# my_reflection_agent.py
import re
from typing import Optional, Dict
from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message


# 定义自定义的Reflection Agent的提示模板
MY_REFLECTION_INITIAL_PROMPT = """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
"""

MY_REFLECTION_REFLECT_PROMPT = """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
"""

MY_REFLECTION_REFINE_PROMPT = """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""


class MyReflectionAgent(ReflectionAgent):
    """
    重写的Reflection Agent - 反思与迭代优化的智能体

    工作流程:
    1. 初始执行: 对任务进行第一次尝试
    2. 反思阶段: 审查当前回答，找出不足
    3. 优化阶段: 根据反馈改进回答
    4. 迭代循环: 重复步骤2-3，直到"无需改进"或达到最大迭代次数
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config, max_iterations, custom_prompts)
        self.max_iterations = max_iterations
        self.current_history = []

        # 支持自定义提示模板，否则使用默认模板
        if custom_prompts:
            self.initial_prompt = custom_prompts.get("initial", MY_REFLECTION_INITIAL_PROMPT)
            self.reflect_prompt = custom_prompts.get("reflect", MY_REFLECTION_REFLECT_PROMPT)
            self.refine_prompt = custom_prompts.get("refine", MY_REFLECTION_REFINE_PROMPT)
        else:
            self.initial_prompt = MY_REFLECTION_INITIAL_PROMPT
            self.reflect_prompt = MY_REFLECTION_REFLECT_PROMPT
            self.refine_prompt = MY_REFLECTION_REFINE_PROMPT

        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行Reflection Agent"""
        self.current_history = []

        print(f"\n🤖 {self.name} 开始处理任务: {input_text[:100]}...")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.initial_prompt.format(task=input_text)
        current_response = self._call_llm(initial_prompt, **kwargs)
        self.current_history.append(f"[初始回答]: {current_response[:200]}...")
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(current_response, "assistant"))

        # --- 2. 迭代循环: 反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("-> 正在进行反思...")
            reflect_prompt = self.reflect_prompt.format(
                task=input_text,
                content=current_response
            )
            feedback = self._call_llm(reflect_prompt, **kwargs)
            self.current_history.append(f"[第{i+1}轮反思]: {feedback[:200]}...")

            # b. 检查停止条件
            if "无需改进" in feedback:
                print("\n✅ 反思认为回答已无需改进，任务完成。")
                break

            # c. 优化
            print("-> 正在进行优化...")
            refine_prompt = self.refine_prompt.format(
                task=input_text,
                last_attempt=current_response,
                feedback=feedback
            )
            current_response = self._call_llm(refine_prompt, **kwargs)
            self.current_history.append(f"[第{i+1}轮优化]: {current_response[:200]}...")
            self.add_message(Message(current_response, "assistant"))

        # --- 3. 返回最终结果 ---
        final_answer = current_response
        print(f"\n🎯 {self.name} 任务完成")
        return final_answer

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """调用LLM并返回响应文本"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.invoke(messages, **kwargs)
        return response_text if response_text else ""

    def get_history(self) -> str:
        """获取完整的执行历史"""
        return "\n".join(self.current_history)
