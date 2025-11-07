import tensorflow as tf
import numpy as np

# --- 1️⃣ Basic setup ---
n_users = 5
n_movies = 6
emb_dim = 3

# --- 2️⃣ Embedding layers ---
user_emb = tf.keras.layers.Embedding(input_dim=n_users, output_dim=emb_dim)
movie_emb = tf.keras.layers.Embedding(input_dim=n_movies, output_dim=emb_dim)

# # --- 3️⃣ Example inputs ---
users = np.array([0, 1, 2, 3, 4])
movies = np.array([1, 3, 4, 2, 0])
ratings = np.array([5, 4, 3, 2, 1], dtype=float)

# # --- 4️⃣ Define model ---
user_in = tf.keras.Input(shape=(1,))
movie_in = tf.keras.Input(shape=(1,))
u_vec = user_emb(user_in)
m_vec = movie_emb(movie_in)
u_vec = tf.keras.layers.Flatten()(u_vec)
m_vec = tf.keras.layers.Flatten()(m_vec)
dot = tf.keras.layers.Dot(axes=1)([u_vec, m_vec])

model = tf.keras.Model(inputs=[user_in, movie_in], outputs=dot)
model.compile(optimizer='adam', loss='mse')

# --- 5️⃣ Inspect initial embeddings ---
print("Before training:\n")
print("User embeddings:\n", user_emb.get_weights()[0])
print("\nMovie embeddings:\n", movie_emb.get_weights()[0])
print("="*60)

# --- 6️⃣ Custom callback to log after each epoch ---
class EmbeddingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- Epoch {epoch+1} ---")
        print("Loss:", logs["loss"])
        print("First user vector:", user_emb.get_weights()[0][0])
        print("First movie vector:", movie_emb.get_weights()[0][0])
        print("-"*50)

# --- 7️⃣ Train ---
model.fit([users, movies], ratings, epochs=5, verbose=1, callbacks=[EmbeddingLogger()])

# --- 8️⃣ After training ---
print("\nAfter training:\n")
print("User embeddings:\n", user_emb.get_weights()[0])
print("\nMovie embeddings:\n", movie_emb.get_weights()[0])
