# explainability subpackage
from src.explainability.feature_glossary import (  # noqa: F401
    FEATURE_GLOSSARY,
    enrich_driver,
    feature_label,
    glossary_as_records,
)
from src.explainability.shap_utils import (  # noqa: F401
    build_case_studies,
    compute_shap_explanation,
    mean_abs_shap_table,
    prepare_model_frame,
    select_case_study_ids,
    write_explainability_report,
)
