import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Dot product of Q and Kᵀ
    matmul_qk = np.dot(Q, K.T)

    # Step 2: Scale by √d (d = dimension of key vectors)
    d = K.shape[1]
    scaled_attention_logits = matmul_qk / np.sqrt(d)

    # Step 3: Apply softmax to get attention weights
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scaled_attention_logits)

    # Step 4: Multiply attention weights with V
    output = np.dot(attention_weights, V)

    return output, attention_weights

# Test Inputs
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

# Run attention
output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", weights)
print("\nOutput:\n", output)
