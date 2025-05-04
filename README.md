# Lamppost-dataset

Binary classifier to detect lamppost pixels in 100 × 100 aerial tiles  
• **502** annotated source photos → **72 268** tiles (1∶1 pos/neg)  
• Baseline **EfficientNet-B1** finetune → 93 % val / 88 % test accuracy  

## Repo layout
| path | role |
|------|------|
| `src/` | tiler · label GUI · exporter · trainer |
| `notebooks/` | QC & modelling notebooks |
| `models/` | JSON metrics + LFS-tracked `.pt` checkpoints |
| `data/` | **ignored** – raw images + tile exports |
| `tests/` | pytest unit tests |

## Quick start

```bash
git clone https://github.com/mo88kh/lamppost-dataset.git
cd lamppost-dataset
python -m venv .venv && source .venv/bin/activate    #  (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pytest


