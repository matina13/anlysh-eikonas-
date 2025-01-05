import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO


def extract_features(images, model_name="resnet50"):
    from torchvision.models import ResNet50_Weights
    phi = getattr(models, model_name)(weights=ResNet50_Weights.DEFAULT)
    phi.eval()
    phi = torch.nn.Sequential(*list(phi.children())[:-1])
    psi = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    V = []
    with torch.no_grad():
        for oi in images:
            oi_tensor = psi(oi).unsqueeze(0)
            vi = phi(oi_tensor).squeeze().numpy()
            V.append(vi)
    return np.array(V)


def rank_normalization(S):
    n = S.shape[0]
    S_normalized = np.zeros_like(S)
    for i in range(n):
        sorted_indices = np.argsort(-S[i])  # Sort indices by descending similarity
        S_normalized[i, sorted_indices] = np.arange(n, 0, -1)  # Assign descending ranks
    return S_normalized


def construct_hypergraph(V, k=10):
    if V.shape[0] == 0:
        raise ValueError("Feature matrix 'V' is empty. Ensure images are loaded and features are extracted.")
    n = V.shape[0]
    S = cosine_similarity(V)
    print("Cosine Similarity Matrix:")
    print(S)
    S_normalized = rank_normalization(S)
    print("Rank Normalized Matrix:")
    print(S_normalized)
    E = {}
    for i in range(n):
        neighbors = np.argsort(-S_normalized[i])[:k]  # Top-k neighbors
        E[i] = neighbors
    return E


def compute_hyperedge_similarities(E, S):
    n = len(E)
    Sh = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                shared = set(E[i]).intersection(E[j])
                similarity_sum = sum(S[i][k] + S[j][k] for k in shared)  # Sum of shared similarities
                Sh[i, j] = similarity_sum / len(shared) if shared else 0  # Average shared similarity
    return Sh


def compute_cartesian_product(E):
    C = {}
    for ei, vertices in E.items():
        for vi in vertices:
            for vj in vertices:
                if vi != vj:
                    C[(vi, vj)] = C.get((vi, vj), 0) + 1 / len(vertices)  # Normalize by the size of the hyperedge
    return C


def compute_hypergraph_similarity(E, V, S):
    Sh = compute_hyperedge_similarities(E, S)
    C = compute_cartesian_product(E)
    n = len(V)
    W = np.zeros((n, n))
    for (vi, vj), weight in C.items():
        W[vi, vj] += weight
    W += Sh  # Combine with weighted hyperedge similarities
    return W


def load_images_from_directory(directory_path):
    import os
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(directory_path, filename))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images


if __name__ == "__main__":
    directory = "C:/Users/matin/Downloads/test"
    C = load_images_from_directory(directory)  # local folder with pictures
    if len(C) < 2:
        raise ValueError("At least two images are required to construct a meaningful hypergraph.")
    V = extract_features(C)
    print("Extracted Features Shape:", V.shape)
    print("Feature Vectors:")
    import matplotlib.pyplot as plt

    # Visualize the feature vectors
    plt.figure(figsize=(12, 6))
    for idx, vector in enumerate(V):
        plt.plot(vector[:50], label=f"Image {idx}")  # Plot first 50 features for clarity
    plt.title("Feature Vectors (First 50 Dimensions)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.show()
    for idx, vector in enumerate(V):
        print(f"Image {idx}: {vector[:10]} ... (truncated)")  # Display the first 10 elements of each vector
    S = cosine_similarity(V)
    E = construct_hypergraph(V, k=min(10, len(C)))
    print("Hypergraph Edges:", E)
    W = compute_hypergraph_similarity(E, V, S)
    print("Hypergraph-Based Similarity Matrix:")
    import matplotlib.pyplot as plt

    # Visualize the similarity matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Similarity Score")
    plt.title("Hypergraph-Based Similarity Matrix")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.xticks(range(len(W)))
    plt.yticks(range(len(W)))
    plt.show()

    # Select a target image (e.g., the first image in the dataset)
    target_idx = 0
    print(f"Target Image: {target_idx}")

    # Sort the images based on their similarity to the target image
    similar_indices = np.argsort(-W[target_idx])  # Descending order of similarity

    print("Most Similar Images:")
    for rank, idx in enumerate(similar_indices, 1):
        if idx != target_idx:  # Exclude the target image itself
            print(f"Rank {rank}: Image {idx} (Similarity: {W[target_idx, idx]:.4f})")
    print(W)
