import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    import sklearn
    import imblearn
except ImportError as e:
    print(f"❌ Validation Failed: Missing dependencies. {e}")
    sys.exit(1)

def test_imbalanced_pipeline():
    try:
        # Generate synthetic imbalanced data
        X, y = make_classification(n_classes=2, class_sep=2,
                                   weights=[0.1, 0.9], n_informative=3, 
                                   n_redundant=1, flip_y=0,
                                   n_features=20, n_clusters_per_class=1,
                                   n_samples=100, random_state=10)

        # Initialize sampler and model
        sampler = SMOTE(random_state=42)
        model = LogisticRegression()
        
        # Build the pipeline
        pipeline = Pipeline(steps=[('sampler', sampler), ('classification', model)])

        # Execute the fit - this is where internal API mismatches surface
        pipeline.fit(X, y)

        print(f"✅ Validation Passed: imblearn {imblearn.__version__} with sklearn {sklearn.__version__}")
        return True

    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imbalanced_pipeline()