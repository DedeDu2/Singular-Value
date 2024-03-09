import numpy as np

# Example dataset (user-item interactions matrix)
data = np.array([[4, 0, 2, 0], [0, 1, 0, 3], [2, 2, 0, 1]])

# Perform SVD
U, sigma, Vt = np.linalg.svd(data)

# Example usage:
user_id = 0
item_id = 3
pred = np.dot(np.dot(U[user_id, :], np.diag(sigma)), Vt[:, item_id])
print("Prediction for user", user_id, "and item", item_id, ":", pred)
