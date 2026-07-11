# explainability subpackage
from src.explainability.shap_utils import (  # noqa: F401
    build_case_studies,
    compute_shap_explanation,
    mean_abs_shap_table,
    prepare_model_frame,
    select_case_study_ids,
    write_explainability_report,
)
