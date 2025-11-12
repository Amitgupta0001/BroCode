import sys
import pathlib
import pytest
import os
import json
import numpy as np
import tempfile

# Ensure ML package path is importable when running tests from project root
proj_root = pathlib.Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(proj_root / "ML"))

# Import classes under test
from keystroke_model import KeystrokeAuthenticationModel, BilingualAuthenticationSystem, TrainResult
import model_io

def make_dummy_samples_labels():
    """
    TODO: replace this with small realistic keystroke-feature arrays and labels.
    Example structure depends on your KeystrokeAuthenticationModel.train signature:
      - samples: list or numpy array of feature vectors
      - labels: list or numpy array of labels (e.g., 0/1 or user IDs)
    """
    # Placeholder - test will be skipped until real data is provided
    return None, None

@pytest.mark.skip(reason="fill in realistic samples/labels in make_dummy_samples_labels()")
def test_train_returns_trainresult(tmp_path):
    samples, labels = make_dummy_samples_labels()
    model = KeystrokeAuthenticationModel(user_id="test_user", language="english", model_dir=str(tmp_path))
    result = model.train(samples, labels, epochs=1)  # adjust kwargs as appropriate
    assert isinstance(result, TrainResult)
    assert result.success in (True, False)
    assert isinstance(result.accuracy, float)
    # Check save/load roundtrip if your model.save_model/load_model exist:
    model_path = tmp_path / "test_model.joblib"
    model.save_model(str(model_path))
    loaded = KeystrokeAuthenticationModel.load_model(str(model_path))
    assert loaded is not None

@pytest.mark.skip(reason="fill in samples/labels and enable this test")
def test_bilingual_register_user(tmp_path):
    samples, labels = make_dummy_samples_labels()
    system = BilingualAuthenticationSystem(model_dir=str(tmp_path))
    success = system.register_user("user_test", "english", samples, labels, epochs=1)
    assert isinstance(success, bool)
    assert success is True  # or adapt expectation to your data

def test_model_io_roundtrip(tmp_path):
    # create a dummy model payload
    dummy = {"foo": "bar", "weights": [1, 2, 3]}
    meta = {"user_id": "u1", "language": "english"}
    path = str(tmp_path / "m.joblib")
    model_io.save_model(path, dummy, metadata=meta)
    loaded_model, loaded_meta = model_io.load_model(path)
    assert loaded_model == dummy
    assert loaded_meta.get("user_id") == "u1"
    assert "saved_at" in loaded_meta

def test_bilingual_register_user_monkeypatched(tmp_path, monkeypatch):
    # Ensure BilingualAuthenticationSystem uses TrainResult success path.
    saved_paths = []
    def fake_train(self, samples, labels, **kwargs):
        # simulate training by writing a small file to model_dir and returning success
        os.makedirs(self.model_dir, exist_ok=True)
        p = os.path.join(self.model_dir, f"{self.user_id}_{self.language}.joblib")
        model_io.save_model(p, {"rehydrated": True}, metadata={"user_id": self.user_id, "language": self.language})
        saved_paths.append(p)
        return TrainResult(success=True, accuracy=0.92)

    monkeypatch.setattr(KeystrokeAuthenticationModel, "train", fake_train)

    system = BilingualAuthenticationSystem(model_dir=str(tmp_path))
    ok = system.register_user("test_user", "english", samples=[[1]], labels=[1])
    assert ok is True
    assert saved_paths and os.path.exists(saved_paths[0])

def make_synthetic_numeric_samples(n_samples=20):
    # create two users with slightly different mean inter-key intervals
    X = []
    y = []
    for i in range(n_samples):
        if i % 2 == 0:
            base = np.random.normal(loc=100.0, scale=10.0, size=10)  # user A
            y.append("userA")
        else:
            base = np.random.normal(loc=140.0, scale=12.0, size=10)  # userB
            y.append("userB")
        # aggregate to small feature vector is handled by train()
        X.append(base.tolist())
    return X, y

def test_train_and_save_load(tmp_path):
    samples, labels = make_synthetic_numeric_samples(30)
    model_dir = str(tmp_path)
    model = KeystrokeAuthenticationModel(user_id="userA", language="english", model_dir=model_dir)
    result = model.train(samples, labels, save_path=os.path.join(model_dir, "uA_english.joblib"))
    assert isinstance(result, TrainResult)
    assert result.success is True
    # load with classmethod
    loaded = KeystrokeAuthenticationModel.load_model(os.path.join(model_dir, "uA_english.joblib"))
    assert loaded is not None
    # authenticate using one sample from userA
    ok, score = loaded.authenticate(samples[0])
    assert isinstance(ok, bool)
    assert isinstance(score, float)

def test_bilingual_register_and_auth(tmp_path):
    samples, labels = make_synthetic_numeric_samples(20)
    system = BilingualAuthenticationSystem(model_dir=str(tmp_path))
    ok = system.register_user("userA", "english", samples, labels)
    assert ok is True
    # pick a sample and authenticate
    acc, score = system.authenticate_user("userA", "english", samples[0])
    assert isinstance(acc, bool)
    assert isinstance(score, float)