import pytest
from api.quiz4 import app
import json
from utils import *
import numpy as np

@pytest.fixture(scope="module")
def digit_samples():
    x, y = read_digits()
    samples = {}
    for digit in range(10):
        sample_index = np.where(y == digit)[0][0]
        samples[digit] = x[sample_index].reshape(1, -1)
    return samples

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200


@pytest.mark.parametrize("digit, expected_prediction", [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
])

def test_post_predict(digit_samples, digit, expected_prediction):
    sample = digit_samples[digit]
    
    response = app.test_client().post("/predict", json={"image": sample.tolist()})
    response_data = json.loads(response.get_data(as_text=True))
    
    prediction = response_data['prediction'][0]
    print(prediction)

    assert response.status_code == 200
    assert prediction == expected_prediction