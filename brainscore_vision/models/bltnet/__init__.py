from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from model import get_model, LAYERS
from brainscore_vision import score
from brainscore_kietzmannlab import PUBLIC_VISION_DATASETS


model_registry['bltnet'] = lambda: ModelCommitment(
    identifier='bltnet',
    activations_model=get_model(),
    layers=LAYERS)

print(model_registry.keys())

model_scores = {}
for dataset in PUBLIC_VISION_DATASETS:
   model_score = score(model_identifier='bltnet', benchmark_identifier=dataset)
   model_scores[dataset] = model_score
for k,v in model_scores.items():
  print(k,v)