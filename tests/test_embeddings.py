import numpy as np

from rag.embeddings import EXPECTED_DIMENSION, describe_model, embed_texts


def test_embeddings_have_expected_shape_and_are_normalized():
    info = describe_model()
    assert info["dimension"] == EXPECTED_DIMENSION
    vectors = embed_texts(["lubricate the ram guide", "spindle warm-up"])
    assert vectors.shape == (2, EXPECTED_DIMENSION)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # Similar maintenance phrases should be closer than unrelated ones.
    query = embed_texts(["routine maintenance lubrication oil"])[0]
    scores = vectors @ query
    assert scores[0] > scores[1]
