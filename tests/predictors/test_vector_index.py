"""VectorIndex and build_vector_index tests."""

import os
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors import load_vector_index
from orient_express.predictors.vector_index import (
    CropSpec,
    VectorIndex,
    build_vector_index,
)


def _assert_labels_match(agg, expected_labels):
    """Assert aggregated labels match expected, ignoring order."""
    assert Counter(agg.labels) == Counter(expected_labels)


def _get_centroid_for(agg, target_label):
    """Look up the centroid vector for a given label."""
    for i, label in enumerate(agg.labels):
        if label == target_label:
            return agg.vectors[i]
    raise ValueError(f"Label {target_label} not found in {agg.labels}")


class TestVectorIndex:
    """Tests for VectorIndex construction, search, aggregation, and serialization."""

    @pytest.fixture
    def normalized_vectors(self):
        np.random.seed(42)
        raw = np.random.randn(6, 64).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    @pytest.fixture
    def single_label_index(self, normalized_vectors):
        labels = ["A", "A", "A", "B", "B", "B"]
        return VectorIndex(vectors=normalized_vectors, labels=labels)

    @pytest.fixture
    def multi_label_index(self, normalized_vectors):
        labels = [("A",), ("A",), ("A", "C"), ("B", "C"), ("B",), ("B",)]
        return VectorIndex(vectors=normalized_vectors, labels=labels)

    @pytest.fixture
    def int_label_index(self, normalized_vectors):
        labels = [101, 102, 103, 104, 105, 106]
        return VectorIndex(vectors=normalized_vectors, labels=labels)

    # -- Construction ---------------------------------------------------------

    def test_construction_with_str_labels(self, normalized_vectors):
        labels = ["x"] * 6
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        assert len(index) == 6
        assert index.dim == 64

    def test_construction_with_int_labels(self, normalized_vectors):
        labels = [1, 2, 3, 4, 5, 6]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        assert len(index) == 6
        assert index.labels == labels

    def test_construction_with_tuple_labels(self, normalized_vectors):
        labels = [("A", "B"), ("C",), ("A", "B"), ("D",), ("C",), ("D",)]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        assert len(index) == 6

    def test_construction_rejects_1d_vectors(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            VectorIndex(vectors=np.array([1.0, 2.0, 3.0]), labels=["a"])

    def test_construction_rejects_mismatched_lengths(self, normalized_vectors):
        with pytest.raises(ValueError, match="labels length"):
            VectorIndex(vectors=normalized_vectors, labels=["a", "b"])

    def test_construction_normalize(self):
        raw = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        index = VectorIndex(vectors=raw, labels=["a", "b"], normalize=True)
        norms = np.linalg.norm(index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_construction_normalize_zero_vector(self):
        raw = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        index = VectorIndex(vectors=raw, labels=["a", "b"], normalize=True)
        np.testing.assert_array_equal(index.vectors[0], [0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(index.vectors[1]), 1.0, atol=1e-6)

    def test_construction_list_labels_are_unhashable(self):
        """Lists are not valid labels because they can't be dict keys."""
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(TypeError):
            VectorIndex(vectors=raw, labels=[["A"], ["B"]])

    def test_construction_duplicate_labels_share_indices(self, normalized_vectors):
        labels = ["A", "A", "A", "B", "B", "B"]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        assert index.label_to_idx["A"] == [0, 1, 2]
        assert index.label_to_idx["B"] == [3, 4, 5]

    # -- Repr -----------------------------------------------------------------

    def test_repr_scalar_labels(self, single_label_index):
        r = repr(single_label_index)
        assert "6 vectors" in r
        assert "dim=64" in r
        assert "2 unique labels" in r

    def test_repr_tuple_labels(self, multi_label_index):
        r = repr(multi_label_index)
        assert "6 vectors" in r
        assert "dim=64" in r

    def test_repr_int_labels(self, int_label_index):
        r = repr(int_label_index)
        assert "6 vectors" in r
        assert "6 unique labels" in r

    # -- Lookup ---------------------------------------------------------------

    def test_get_by_label_int(self, int_label_index, normalized_vectors):
        vec = int_label_index.get_by_label(101)
        np.testing.assert_allclose(vec, normalized_vectors[0:1])

    def test_get_by_labels_int(self, int_label_index, normalized_vectors):
        vecs = int_label_index.get_by_labels([101, 103])
        np.testing.assert_allclose(vecs[0], normalized_vectors[0])
        np.testing.assert_allclose(vecs[1], normalized_vectors[2])

    def test_get_by_label_str(self, single_label_index, normalized_vectors):
        vecs = single_label_index.get_by_label("A")
        assert vecs.shape == (3, 64)
        np.testing.assert_allclose(vecs, normalized_vectors[:3])

    def test_get_by_label_tuple(self, multi_label_index, normalized_vectors):
        vecs = multi_label_index.get_by_label(("A", "C"))
        assert vecs.shape == (1, 64)
        np.testing.assert_allclose(vecs[0], normalized_vectors[2])

    def test_get_by_label_missing_raises(self, single_label_index):
        with pytest.raises(KeyError):
            single_label_index.get_by_label("NONEXISTENT")

    def test_get_by_idx(self, single_label_index, normalized_vectors):
        vec = single_label_index.get_by_idx(0)
        assert vec.shape == (64,)
        np.testing.assert_allclose(vec, normalized_vectors[0])

        vecs = single_label_index.get_by_idxs([0, 1])
        assert vecs.shape == (2, 64)
        np.testing.assert_allclose(vecs[0], normalized_vectors[0])
        np.testing.assert_allclose(vecs[1], normalized_vectors[1])

    # -- Search ---------------------------------------------------------------

    def test_search_self_is_top_match(self, single_label_index, normalized_vectors):
        results = single_label_index.search(normalized_vectors[0], k=1)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[0].labels == "A"

    def test_search_k_larger_than_index(self, single_label_index, normalized_vectors):
        results = single_label_index.search(normalized_vectors[0], k=100)
        assert len(results) == 6

    def test_search_returns_descending_scores(
        self, single_label_index, normalized_vectors
    ):
        results = single_label_index.search(normalized_vectors[0], k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_tuple_label_preserved(self, multi_label_index, normalized_vectors):
        results = multi_label_index.search(normalized_vectors[2], k=1)
        assert results[0].labels == ("A", "C")

    def test_search_int_labels(self, int_label_index, normalized_vectors):
        results = int_label_index.search(normalized_vectors[0], k=1)
        assert results[0].labels == 101

    def test_search_1d_and_2d_query_equivalent(
        self, single_label_index, normalized_vectors
    ):
        query = normalized_vectors[0]
        results_1d = single_label_index.search(query, k=3)
        results_2d = single_label_index.search(query.reshape(1, -1), k=3)
        for r1, r2 in zip(results_1d, results_2d, strict=True):
            assert r1.score == pytest.approx(r2.score)
            assert r1.labels == r2.labels

    def test_search_batch(self, single_label_index, normalized_vectors):
        queries = normalized_vectors[:2]
        batch_results = single_label_index.search_batch(queries, k=2)
        assert len(batch_results) == 2
        assert len(batch_results[0]) == 2
        assert len(batch_results[1]) == 2
        assert batch_results[0][0].score == pytest.approx(1.0, abs=1e-5)
        assert batch_results[1][0].score == pytest.approx(1.0, abs=1e-5)

    # -- Aggregation ----------------------------------------------------------

    def test_aggregate_scalar_labels(self, single_label_index):
        agg = single_label_index.aggregate()
        assert len(agg) == 2
        _assert_labels_match(agg, ["A", "B"])
        norms = np.linalg.norm(agg.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_aggregate_scalar_labels_per_label(self, single_label_index):
        """per_label=True on scalar labels is identical to default."""
        agg = single_label_index.aggregate(per_label=True)
        assert len(agg) == 2
        _assert_labels_match(agg, ["A", "B"])

    def test_aggregate_tuple_labels_default(self, multi_label_index):
        """Default: one centroid per unique tuple label."""
        agg = multi_label_index.aggregate()
        assert len(agg) == 4
        _assert_labels_match(agg, [("A",), ("A", "C"), ("B",), ("B", "C")])

    def test_aggregate_tuple_label_centroid_correctness(
        self, multi_label_index, normalized_vectors
    ):
        """Tuple label ("A", "C") appears only on vector 2.
        Its centroid should equal that vector (already normalized)."""
        agg = multi_label_index.aggregate()
        centroid = _get_centroid_for(agg, ("A", "C"))
        np.testing.assert_allclose(centroid, normalized_vectors[2], atol=1e-5)

    def test_aggregate_shared_label_centroid(
        self, multi_label_index, normalized_vectors
    ):
        """Tuple label ("A",) appears on vectors 0 and 1.
        Its centroid should be their normalized mean."""
        agg = multi_label_index.aggregate()
        centroid = _get_centroid_for(agg, ("A",))
        expected = normalized_vectors[0] + normalized_vectors[1]
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(centroid, expected, atol=1e-5)

    def test_aggregate_tuple_per_label(self, multi_label_index):
        """per_label=True: one centroid per individual element from tuples."""
        agg = multi_label_index.aggregate(per_label=True)
        assert len(agg) == 3
        _assert_labels_match(agg, ["A", "B", "C"])

    def test_aggregate_tuple_per_label_centroid_correctness(
        self, multi_label_index, normalized_vectors
    ):
        """per_label=True: label "C" appears in tuples on vectors 2 and 3.
        Its centroid should be the normalized mean of those two vectors."""
        agg = multi_label_index.aggregate(per_label=True)
        centroid = _get_centroid_for(agg, "C")
        expected = normalized_vectors[2] + normalized_vectors[3]
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(centroid, expected, atol=1e-5)

    def test_aggregate_int_labels(self, normalized_vectors):
        labels = [1, 1, 1, 2, 2, 2]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        agg = index.aggregate()
        assert len(agg) == 2
        _assert_labels_match(agg, [1, 2])

    def test_aggregate_already_unique(self, normalized_vectors):
        labels = ["a", "b", "c", "d", "e", "f"]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        agg = index.aggregate()
        assert len(agg) == 6
        _assert_labels_match(agg, labels)

    def test_aggregate_per_label_noop_on_scalars(self, normalized_vectors):
        """per_label=True with scalar (non-tuple) labels behaves like default."""
        labels = ["A", "A", "B", "B", "C", "C"]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        agg_default = index.aggregate()
        agg_per_label = index.aggregate(per_label=True)
        assert len(agg_default) == len(agg_per_label)
        _assert_labels_match(agg_default, agg_per_label.labels)

    # -- Dump / Load ----------------------------------------------------------

    def test_dump_and_load_roundtrip_str_labels(self, single_label_index, tmp_path):
        single_label_index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert len(loaded) == len(single_label_index)
        assert loaded.labels == single_label_index.labels
        np.testing.assert_allclose(loaded.vectors, single_label_index.vectors)

    def test_dump_and_load_roundtrip_int_labels(self, int_label_index, tmp_path):
        int_label_index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert loaded.labels == int_label_index.labels
        np.testing.assert_allclose(loaded.vectors, int_label_index.vectors)

    def test_dump_and_load_roundtrip_tuple_labels(self, multi_label_index, tmp_path):
        """Tuple labels survive dump/load (JSON serializes tuples as arrays,
        load_vector_index must convert them back)."""
        multi_label_index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert loaded.labels == multi_label_index.labels
        np.testing.assert_allclose(loaded.vectors, multi_label_index.vectors)
        # Verify they're actually tuples, not lists
        for label in loaded.labels:
            if not isinstance(label, (int, str)):
                assert isinstance(label, tuple), f"Expected tuple, got {type(label)}"

    def test_dump_and_load_mixed_scalar_and_tuple_labels(
        self, normalized_vectors, tmp_path
    ):
        """A mix of scalar and tuple labels round-trips correctly."""
        labels = ["A", ("B", "C"), 42, ("D",), "E", 99]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert loaded.labels == labels
        for orig, roundtripped in zip(labels, loaded.labels, strict=True):
            assert type(orig) is type(roundtripped)

    def test_dump_creates_expected_files(self, single_label_index, tmp_path):
        files = single_label_index.dump(str(tmp_path))
        assert len(files) == 2
        assert all(os.path.exists(f) for f in files)
        assert any(f.endswith(".yaml") for f in files)
        assert any(f.endswith(".npz") for f in files)


# -----------------------------------------------------------------------------
# build_vector_index Tests
# -----------------------------------------------------------------------------


class TestBuildVectorIndex:
    """Tests for the build_vector_index factory function."""

    @pytest.fixture
    def mock_feature_extractor(self):
        extractor = MagicMock()

        def fake_predict(images):
            results = []
            for _ in images:
                mock_result = MagicMock()
                mock_result.feature = np.random.randn(64).astype(np.float32)
                results.append(mock_result)
            return results

        extractor.predict.side_effect = fake_predict
        return extractor

    def test_build_scalar_labels(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(4)
        ]
        labels = ["A", "A", "B", "B"]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        assert len(index) == 4
        assert index.labels == ["A", "A", "B", "B"]

    def test_build_int_labels(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(3)
        ]
        labels = [101, 102, 103]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        assert len(index) == 3
        assert index.labels == [101, 102, 103]

    def test_build_tuple_labels(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(3)
        ]
        labels = [("A", "B"), ("B",), ("C",)]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        assert len(index) == 3
        assert index.labels == [("A", "B"), ("B",), ("C",)]

    def test_build_from_file_paths(self, mock_feature_extractor, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"crop_{i}.png"
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(str(p))
            paths.append(str(p))
        labels = ["A", "B", "C"]
        index = build_vector_index(paths, labels, mock_feature_extractor)
        assert len(index) == 3

    def test_build_rejects_mismatched_lengths(self, mock_feature_extractor):
        crops = [Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))]
        with pytest.raises(ValueError, match="crops length"):
            build_vector_index(crops, ["A", "B"], mock_feature_extractor)

    def test_build_rejects_bad_crop_type(self, mock_feature_extractor):
        with pytest.raises(
            TypeError, match="PIL Image, a file path string, or a CropSpec"
        ):
            build_vector_index([12345], ["A"], mock_feature_extractor)

    def test_build_batching(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(5)
        ]
        labels = ["A"] * 5
        build_vector_index(crops, labels, mock_feature_extractor, batch_size=2)
        assert mock_feature_extractor.predict.call_count == 3  # 2 + 2 + 1

    def test_build_normalizes_by_default(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(3)
        ]
        labels = ["A", "B", "C"]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        norms = np.linalg.norm(index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_build_from_crop_specs(self, mock_feature_extractor, tmp_path):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p = tmp_path / "full_image.png"
        img.save(str(p))
        specs = [
            CropSpec(path=str(p), bbox=(0, 0, 50, 50)),
            CropSpec(path=str(p), bbox=(50, 50, 100, 100)),
        ]
        index = build_vector_index(specs, ["A", "B"], mock_feature_extractor)
        assert len(index) == 2

    def test_build_mixed_crop_types(self, mock_feature_extractor, tmp_path):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p = tmp_path / "image.png"
        img.save(str(p))
        crops = [
            img,
            str(p),
            CropSpec(path=str(p), bbox=(10, 10, 50, 50)),
        ]
        index = build_vector_index(crops, ["A", "B", "C"], mock_feature_extractor)
        assert len(index) == 3

    def test_crop_spec_crops_correctly(self, tmp_path):
        """Verify CropSpec actually crops to the specified bbox."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:40, 30:60] = [255, 0, 0]
        img = Image.fromarray(arr)
        p = tmp_path / "image.png"
        img.save(str(p))

        from orient_express.predictors.vector_index import _CropDataset

        dataset = _CropDataset([CropSpec(path=str(p), bbox=(30, 20, 60, 40))])
        crop = dataset[0]
        crop_arr = np.array(crop)
        assert crop.size == (30, 20)
        assert np.all(crop_arr[:, :, 0] == 255)
