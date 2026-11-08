"""Mục 2: một nơ-ron sigmoid và mạng XOR bằng trọng số tự đặt."""
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z, -700, 700)))


def relu(z):
    return np.maximum(0, z)


if __name__ == "__main__":
    w, b, x = np.array([0.8, -0.5, 0.3]), -0.2, np.array([2., 1., 3.])
    z = w@x+b
    print(f"Mot no-ron: z={z:.2f}, sigmoid(z)={sigmoid(z):.4f}")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    W1, b1 = np.array([[1., 1.], [1., 1.]]), np.array([0., -1.])
    W2, b2 = np.array([1., -2.]), 0.
    H = relu(X@W1+b1)
    Y = H@W2+b2
    print(" x1 x2  h1 h2  y")
    print(np.column_stack([X, H, Y]).astype(int))
    print("Tinh ca bang mot lan:", Y)
    np.testing.assert_array_equal(Y, [0, 1, 1, 0])
    print("h2 chi bat khi hai dau vao deu 1; tru 2*h2 dua dau ra (1,1) ve 0.")
