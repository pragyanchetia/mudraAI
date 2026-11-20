"""Step 2: train the XGBoost mudra classifier on the 33 features."""
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import config as C


def main():
    train_df = pd.read_csv(C.TRAIN_CSV).dropna()
    test_df = pd.read_csv(C.TEST_CSV).dropna()
    X_train, y_train = train_df.drop("label", axis=1), train_df["label"]
    X_test, y_test = test_df.drop("label", axis=1), test_df["label"]
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # Generic names (f0..f32) keep the model ONNX-compatible
    names = [f"f{i}" for i in range(X_train.shape[1])]
    X_train.columns = names
    X_test.columns = names

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)      # fit on train only
    y_test_enc = encoder.transform(y_test)
    joblib.dump(encoder, C.LABEL_ENCODER_PATH)
    print(f"Label encoder saved ({len(encoder.classes_)} classes)")

    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, gamma=0.1,
        objective="multi:softprob", num_class=len(encoder.classes_),
        eval_metric="mlogloss", random_state=C.RANDOM_STATE,
    )
    print("Training XGBoost...")
    model.fit(X_train, y_train_enc)

    pred = model.predict(X_test)
    print(f"\nAccuracy on unseen test data: {accuracy_score(y_test_enc, pred) * 100:.2f}%\n")
    print(classification_report(y_test_enc, pred, target_names=encoder.classes_, zero_division=0))

    joblib.dump(model, C.CLASSIFIER_PATH)
    print(f"Model saved to {C.CLASSIFIER_PATH}")

    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_xgboost(
            model, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))])
        C.CLASSIFIER_ONNX.write_bytes(onnx_model.SerializeToString())
        print(f"ONNX model saved to {C.CLASSIFIER_ONNX}")
    except Exception as e:  # ONNX export is optional
        print(f"ONNX export skipped: {e}")


if __name__ == "__main__":
    main()
