# my_plan_solve_agent.py
import re
from typing import Optional, Dict, List
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message


# 定义自定义的Plan-and-Solve Agent的提示模板
MY_PLAN_SOLVE_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

MY_PLAN_SOLVE_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

MY_PLAN_SOLVE_SUMMARY_PROMPT = """
你是一位顶级的AI总结专家。请根据以下所有步骤的执行结果，对原始问题给出一个完整、准确的最终回答。

# 原始问题:
{question}

# 完整计划:
{plan}

# 各步骤执行结果:
{step_results}

请综合以上所有信息，给出最终答案:
"""


class MyPlanSolveAgent(PlanAndSolveAgent):
    """
    重写的Plan-and-Solve Agent - 先规划后执行的智能体

    工作流程:
    1. 规划阶段: 将复杂问题分解为有序的子任务列表
    2. 执行阶段: 按顺序逐步执行每个子任务，积累中间结果
    3. 汇总阶段: 综合所有步骤结果，生成最终答案
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 10,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_steps = max_steps
        self.current_history: List[str] = []

        # 支持自定义提示模板，否则使用默认模板
        if custom_prompts:
            self.planner_prompt = custom_prompts.get("planner", MY_PLAN_SOLVE_PLANNER_PROMPT)
            self.executor_prompt = custom_prompts.get("executor", MY_PLAN_SOLVE_EXECUTOR_PROMPT)
            self.summary_prompt = custom_prompts.get("summary", MY_PLAN_SOLVE_SUMMARY_PROMPT)
        else:
            self.planner_prompt = MY_PLAN_SOLVE_PLANNER_PROMPT
            self.executor_prompt = MY_PLAN_SOLVE_EXECUTOR_PROMPT
            self.summary_prompt = MY_PLAN_SOLVE_SUMMARY_PROMPT

        print(f"✅ {name} 初始化完成，最大执行步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行Plan-and-Solve Agent"""
        self.current_history = []

        print(f"\n🤖 {self.name} 开始处理问题: {input_text[:100]}...")

        # --- 1. 规划阶段 ---
        print("\n--- 正在生成行动计划 ---")
        plan = self._generate_plan(input_text, **kwargs)

        if not plan:
            print("\n⚠️ 无法生成有效的行动计划，任务终止。")
            return "无法生成有效的行动计划。"

        print(f"📋 计划已生成，共 {len(plan)} 个步骤:")
        for i, step in enumerate(plan, 1):
            print(f"   {i}. {step}")

        # --- 2. 执行阶段 ---
        print(f"\n--- 开始执行计划 ---")
        step_results: List[str] = []

        for i, step in enumerate(plan, 1):
            if i > self.max_steps:
                print(f"\n⚠️ 已达到最大执行步数限制 ({self.max_steps})，停止执行。")
                break

            print(f"\n--- 执行步骤 {i}/{len(plan)}: {step[:80]}... ---")

            # 构建历史记录
            history_str = self._build_history(step_results)

            # 执行当前步骤
            executor_prompt = self.executor_prompt.format(
                question=input_text,
                plan="\n".join([f"{j}. {s}" for j, s in enumerate(plan, 1)]),
                history=history_str,
                current_step=step
            )
            step_answer = self._call_llm(executor_prompt, **kwargs)
            step_results.append(f"步骤{i} [{step}]: {step_answer}")
            self.current_history.append(f"[步骤{i}结果]: {step_answer[:200]}...")
            print(f"   结果: {step_answer[:150]}...")

        # --- 3. 汇总最终答案 ---
        if len(plan) > 1:
            print(f"\n--- 正在汇总最终答案 ---")
            summary_prompt = self.summary_prompt.format(
                question=input_text,
                plan="\n".join([f"{j}. {s}" for j, s in enumerate(plan, 1)]),
                step_results="\n".join(step_results)
            )
            final_answer = self._call_llm(summary_prompt, **kwargs)
        else:
            # 单步骤计划，直接提取结果
            final_answer = step_results[0] if step_results else "未能完成任何步骤。"
            separator_idx = final_answer.find("]: ")
            if separator_idx != -1:
                final_answer = final_answer[separator_idx + 3:]

        print(f"\n🎯 {self.name} 任务完成")
        print(f"最终答案: {final_answer[:200]}...")
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _generate_plan(self, question: str, **kwargs) -> List[str]:
        """调用LLM生成执行计划"""
        prompt = self.planner_prompt.format(question=question)
        response = self._call_llm(prompt, **kwargs)
        plan = self._parse_plan(response)
        self.current_history.append(f"[计划]: {plan}")
        return plan

    def _parse_plan(self, response: str) -> List[str]:
        """从LLM响应中解析出计划步骤列表"""
        # 策略1: 匹配 Python 列表格式 ["步骤1", "步骤2", ...]
        match = re.search(r'\[([^\]]*)\]', response, re.DOTALL)
        if match:
            items_str = match.group(1)
            steps = re.findall(r'["\']([^"\']*)["\']', items_str)
            if steps:
                return steps

        # 策略2: 按行解析编号格式 "1. xxx" 或 "步骤1: xxx"
        lines = response.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            match = re.match(r'^(?:\d+[\.\)、]\s*|步骤\d+[：:]\s*)(.+)', line)
            if match:
                steps.append(match.group(1).strip())
        return steps if steps else []

    def _build_history(self, step_results: List[str]) -> str:
        """构建历史步骤与结果的文本"""
        if not step_results:
            return "暂无历史步骤。"
        return "\n".join(step_results)

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """调用LLM并返回响应文本"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.invoke(messages, **kwargs)
        return response_text if response_text else ""

    def get_history(self) -> str:
        """获取完整的执行历史"""
        return "\n".join(self.current_history)
