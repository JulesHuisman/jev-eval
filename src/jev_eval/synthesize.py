from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.synthesizer import Synthesizer

goldens = Synthesizer().generate_goldens_from_docs(
    document_paths=['assets/attention-is-all-you-need.pdf']
)
dataset = EvaluationDataset(goldens=goldens)

dataset.save_as(file_type="json", directory="assets", file_name="dataset")