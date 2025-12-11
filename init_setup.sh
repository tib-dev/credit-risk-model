#!/usr/bin/env bash

# ==============================================================================
# init_setup.sh
# Project: credit-risk-model
# Description: Bootstrap full project structure (package-based src/credit_risk_model)
# ==============================================================================

set -e
set -o pipefail

echo "=============================================="
echo "Credit Risk Model - Project Structure Setup"
echo "=============================================="

# ----------------------------
# 1. Project root directories
# ----------------------------
dirs=(
    "config"
    "data/raw"
    "data/interim"
    "data/processed"
    "data/external"
    "notebooks"
    "src/credit_risk_model"
    "src/credit_risk_model/data"
    "src/credit_risk_model/features"
    "src/credit_risk_model/models"
    "src/credit_risk_model/api"
    "src/credit_risk_model/pipeline"
    "src/credit_risk_model/utils"
    "tests"
    "docker"
    "scripts"
    "mlruns"
    ".github/workflows"
)

for d in "${dirs[@]}"; do
    mkdir -p "$d"
done

echo "All directories created."

# ----------------------------
# 2. Create empty Python files
# ----------------------------
py_files=(
    # Package __init__
    "src/credit_risk_model/__init__.py"
    "src/credit_risk_model/config_loader.py"
    "src/credit_risk_model/settings.py"
    
    # Data module
    "src/credit_risk_model/data/__init__.py"
    "src/credit_risk_model/data/load_data.py"
    "src/credit_risk_model/data/preprocess.py"
    "src/credit_risk_model/data/rfm.py"
    "src/credit_risk_model/data/pipeline.py"
    "src/credit_risk_model/data/splitter.py"
    
    # Features module
    "src/credit_risk_model/features/__init__.py"
    "src/credit_risk_model/features/aggregate.py"
    "src/credit_risk_model/features/categorical.py"
    "src/credit_risk_model/features/numerical.py"
    "src/credit_risk_model/features/woe_iv.py"
    "src/credit_risk_model/features/feature_builder.py"
    
    # Models module
    "src/credit_risk_model/models/__init__.py"
    "src/credit_risk_model/models/train.py"
    "src/credit_risk_model/models/evaluate.py"
    "src/credit_risk_model/models/tuning.py"
    "src/credit_risk_model/models/predict.py"
    "src/credit_risk_model/models/register.py"
    
    # API module
    "src/credit_risk_model/api/__init__.py"
    "src/credit_risk_model/api/main.py"
    "src/credit_risk_model/api/pydantic_models.py"
    "src/credit_risk_model/api/utils.py"
    
    # Pipeline module
    "src/credit_risk_model/pipeline/__init__.py"
    "src/credit_risk_model/pipeline/dvc_stage_data.py"
    "src/credit_risk_model/pipeline/dvc_stage_features.py"
    "src/credit_risk_model/pipeline/dvc_stage_train.py"
    "src/credit_risk_model/pipeline/dvc_stage_evaluate.py"
    
    # Utils module
    "src/credit_risk_model/utils/__init__.py"
    "src/credit_risk_model/utils/logger.py"
    "src/credit_risk_model/utils/helpers.py"
    "src/credit_risk_model/utils/constants.py"
    
    # Tests
    "tests/__init__.py"
    "tests/test_data_processing.py"
    "tests/test_feature_engineering.py"
    "tests/test_model_training.py"
    "tests/test_api.py"
)

for f in "${py_files[@]}"; do
    touch "$f"
done

echo "All Python files created."

# ----------------------------
# 3. Create YAML config files
# ----------------------------
yaml_files=(
    "config/main.yaml"
    "config/data.yaml"
    "config/features.yaml"
    "config/model.yaml"
    "config/proxy.yaml"
    "config/train.yaml"
    "config/api.yaml"
)

for y in "${yaml_files[@]}"; do
    touch "$y"
done

echo "All YAML config files created."

# ----------------------------
# 4. Create core project files
# ----------------------------
core_files=(
    "README.md"
    ".gitignore"
    ".dvcignore"
    ".env.example"
    "pyproject.toml"
    "requirements.txt"
    "dvc.yaml"
    "params.yaml"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
    "docker/start.sh"
    "notebooks/eda.ipynb"
    "notebooks/rfm.ipynb"
    "notebooks/modeling.ipynb"
    "notebooks/experiments.ipynb"
)

for f in "${core_files[@]}"; do
    touch "$f"
done

echo "All core files created."

# ----------------------------
# 5. Create scripts
# ----------------------------
scripts=(
    "scripts/run_api.sh"
    "scripts/run_training.sh"
    "scripts/run_pipeline.sh"
)

for s in "${scripts[@]}"; do
    touch "$s"
    chmod +x "$s"
done

# Add starter content for scripts
echo '#!/usr/bin/env bash' > scripts/run_api.sh
echo 'echo "Starting FastAPI server..."' >> scripts/run_api.sh
echo 'uvicorn credit_risk_model.api.main:app --host 0.0.0.0 --port 8000 --reload' >> scripts/run_api.sh

echo '#!/usr/bin/env bash' > scripts/run_training.sh
echo 'echo "Running training pipeline..."' >> scripts/run_training.sh
echo 'python -m credit_risk_model.models.train' >> scripts/run_training.sh

echo '#!/usr/bin/env bash' > scripts/run_pipeline.sh
echo 'echo "Running full DVC pipeline..."' >> scripts/run_pipeline.sh
echo 'dvc repro' >> scripts/run_pipeline.sh

echo "All scripts created and made executable."

# ----------------------------
# 6. Initialize virtual environment
# ----------------------------
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "Virtual environment created at .venv"
fi

# ----------------------------
# 7. Final message
# ----------------------------
echo "=============================================="
echo "Package-based project structure successfully created!"
echo "Activate your environment: source .venv/bin/activate"
echo "Run API: scripts/run_api.sh"
echo "Run Training: scripts/run_training.sh"
echo "Run DVC Pipeline: scripts/run_pipeline.sh"
echo "Edit config files in ./config to customize"
echo "=============================================="
