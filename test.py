# test.py

import pytest
from sparse_recommender import SparseMatrix  # Adjust the import statement based on your directory structure

def test_set_and_get():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    assert sparse_matrix.get(0, 0) == 5

def test_recommend():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    user_vector = [1]
    recommendations = sparse_matrix.recommend(user_vector)
    assert recommendations == [5]

def test_add_movie():
    sparse_matrix1 = SparseMatrix()
    sparse_matrix1.set(0, 0, 5)
    sparse_matrix2 = SparseMatrix()
    sparse_matrix2.set(1, 1, 3)
    result = sparse_matrix1.add_movie(sparse_matrix2)
    assert result.get(0, 0) == 5
    assert result.get(1, 1) == 3

def test_to_dense():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    dense_matrix = sparse_matrix.to_dense()
    assert dense_matrix == [[5]]

def test_set_and_get_multiple_values():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    sparse_matrix.set(1, 1, 3)
    sparse_matrix.set(2, 2, 7)

    assert sparse_matrix.get(0, 0) == 5
    assert sparse_matrix.get(1, 1) == 3
    assert sparse_matrix.get(2, 2) == 7

def test_recommend_multiple_movies():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    sparse_matrix.set(1, 1, 3)
    sparse_matrix.set(2, 2, 7)

    user_vector = [1, 0, 1]
    recommendations = sparse_matrix.recommend(user_vector)
    assert recommendations == [5, 7]

def test_add_movie_with_overlap():
    sparse_matrix1 = SparseMatrix()
    sparse_matrix1.set(0, 0, 5)
    sparse_matrix1.set(1, 1, 3)

    sparse_matrix2 = SparseMatrix()
    sparse_matrix2.set(1, 1, 2)
    sparse_matrix2.set(2, 2, 7)

    result = sparse_matrix1.add_movie(sparse_matrix2)
    assert result.get(0, 0) == 5
    assert result.get(1, 1) == 5  # Should sum overlapping values (3 + 2)
    assert result.get(2, 2) == 7

def test_to_dense_with_empty_matrix():
    sparse_matrix = SparseMatrix()
    dense_matrix = sparse_matrix.to_dense()
    assert dense_matrix == [[]]  # Empty matrix

def test_to_dense_multiple_values():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 0, 5)
    sparse_matrix.set(1, 1, 3)
    sparse_matrix.set(2, 2, 7)

    dense_matrix = sparse_matrix.to_dense()
    assert dense_matrix == [[5, 0, 0], [0, 3, 0], [0, 0, 7]]

if __name__ == "__main__":
    pytest.main([__file__])
