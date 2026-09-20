import asyncio

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from langchain_openai.chat_models import ChatOpenAI

from jev_eval.jev_correctness import JevCorrectnessMetric

llm = ChatOpenAI(model="gpt-5.6-terra")


async def generate_answer(question: str) -> str:
    """Answer a single dataset question."""
    response = await llm.ainvoke(f"Answer the question: {question}")
    return response.content


async def main() -> None:
    """Generate answers for every golden question, then score them using Jev."""
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path="assets/dataset.json")

    answers = await asyncio.gather(
        *(generate_answer(question=golden.input) for golden in dataset.goldens)
    )

    test_cases = [
        LLMTestCase(
            input=golden.input,
            actual_output=answer,
            expected_output=golden.expected_output,
        )
        for golden, answer in zip(dataset.goldens, answers, strict=True)
    ]

    evaluate(test_cases=test_cases, metrics=[JevCorrectnessMetric()])


if __name__ == "__main__":
    asyncio.run(main())
