"""Step 3: One-Class SVM 'gatekeeper' that rejects non-mudra hand poses."""
import joblib
import pandas as pd
from sklearn.svm import OneClassSVM

import config as C


def main():
    train_df = pd.read_csv(C.TRAIN_CSV).dropna()
    test_df = pd.read_csv(C.TEST_CSV).dropna()
    X_train = train_df.drop("label", axis=1).to_numpy()
    X_test = test_df.drop("label", axis=1).to_numpy()
    print(f"Features: {X_train.shape[1]}  Train rows: {len(X_train)}")

    svm = OneClassSVM(gamma="scale", nu=C.GATEKEEPER_NU)
    print("Training One-Class SVM...")
    svm.fit(X_train)

    train_acc = (svm.predict(X_train) == 1).mean() * 100
    test_acc = (svm.predict(X_test) == 1).mean() * 100
    print(f"Valid-mudra acceptance rate - Train: {train_acc:.2f}%  Test: {test_acc:.2f}%")

    joblib.dump(svm, C.GATEKEEPER_PATH)
    print(f"Gatekeeper saved to {C.GATEKEEPER_PATH}")

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            svm, initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=11)
        C.GATEKEEPER_ONNX.write_bytes(onnx_model.SerializeToString())
        print(f"ONNX model saved to {C.GATEKEEPER_ONNX}")
    except Exception as e:  # ONNX export is optional
        print(f"ONNX export skipped: {e}")


if __name__ == "__main__":
    main()
