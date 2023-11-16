# sparse_recommender.py

class SparseMatrix:
    def __init__(self):
        self.matrix = {}

    def set(self, row, col, value):
        if value != 0:
            self.matrix[(row, col)] = value

    def get(self, row, col):
        return self.matrix.get((row, col), 0)

    def recommend(self, user_vector):
        recommendations = []
        for row in range(len(user_vector)):
            if user_vector[row] == 1:
                recommendation = max(
                    (self.get(row, col) for col in range(len(user_vector))),
                    default=0,
                )
                recommendations.append(recommendation)
        return recommendations

    def add_movie(self, other_matrix):
        result = SparseMatrix()

        # Copy the values from the first matrix
        for (row, col), value in self.matrix.items():
            result.set(row, col, value)

        # Add the values from the second matrix, summing overlapping values
        for (row, col), value in other_matrix.matrix.items():
            current_value = result.get(row, col)
            result.set(row, col, current_value + value)

        return result

    def to_dense(self):
        if not self.matrix:
            return [[]]

        # Determine the dimensions of the matrix
        max_row = max(row for row, _ in self.matrix.keys())
        max_col = max(col for _, col in self.matrix.keys())

        # Initialize a dense matrix filled with zeros
        dense_matrix = [[0] * (max_col + 1) for _ in range(max_row + 1)]

        # Fill in the values from the sparse matrix
        for (row, col), value in self.matrix.items():
            dense_matrix[row][col] = value

        return dense_matrix
