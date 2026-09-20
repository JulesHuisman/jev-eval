import asyncio
from functools import cached_property

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain_typesafe import Noul, TypeSafeClassifier


class JevCorrectnessMetric(BaseMetric):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    @cached_property
    def classifier(self) -> TypeSafeClassifier:
        return TypeSafeClassifier(questions={
            "correctness": Noul(
                instructions="The generated answer is correct compared to the ground truth.",
                criteria={
                    "yes": "The answer is fully correct and aligns with the ground truth.",
                    "no": "The answer is fully incorrect and contradicts the ground truth.",
                }
            ),
        },
    )

    def _build_state(self, test_case: LLMTestCase) -> dict[str, str]:
        """Build the classifier input for a test case."""
        return {
            "question": test_case.input,
            "ground_truth": test_case.expected_output,
            "generated_answer": test_case.actual_output,
        }

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """
        In case we are running in sync mode.
        """
        return asyncio.run(self.a_measure(test_case=test_case, *args, **kwargs))

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """
        Score the generated output based on the ground truth, use Jev noul.
        """
        state = self._build_state(test_case=test_case)
        response = await self.classifier.ainvoke(state)
        self.score = response.nouls["correctness"].noul
        return self.score

    @property
    def __name__(self):
        return "Correctness"