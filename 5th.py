import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # Or "Agg", "QtAgg", or any compatible backend


def extract_features(images, model_name="resnet101"):
    from torchvision.models import ResNet101_Weights
    phi = getattr(models, model_name)(weights=ResNet101_Weights.DEFAULT)
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


def construct_hypergraph(V, k=10, threshold=0.5):
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
        neighbors = [idx for idx in np.argsort(-S_normalized[i]) if S[i, idx] > threshold][:k]  # Filter by threshold
        E[i] = neighbors
    return E


def compute_hyperedge_similarities(E, S):
    n = len(E)
    Sh = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                shared = set(E[i]).intersection(E[j])
                similarity_sum = sum(S[i][k] * S[j][k] for k in shared)  # Weighted similarity
                Sh[i, j] = similarity_sum / len(shared) if shared else 0  # Average weighted similarity
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
    # Normalize W
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
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

    directory = os.path.dirname(os.path.abspath(__file__))

    C = load_images_from_directory(directory)  # local folder with pictures
    if len(C) < 2:
        raise ValueError("At least two images are required to construct a meaningful hypergraph.")
    V = extract_features(C)
    print("Extracted Features Shape:", V.shape)
    S = cosine_similarity(V)
    E = construct_hypergraph(V, k=min(10, len(C)))
    print("Hypergraph Edges:", E)
    W = compute_hypergraph_similarity(E, V, S)
    print("Hypergraph-Based Similarity Matrix:")
    print(W)

    # Example labels for Precision@K (adjust these based on the dataset)
    labels = ["daisy", "daisy", "rose", "daisy", "rose", "rose", "sunflower", "sunflower", "sunflower", "sunflower", "violet"]

    # Compute Precision@K
    def precision_at_k(W, labels, target_idx, k=5):
        similar_indices = np.argsort(-W[target_idx])  # Descending order of similarity
        relevant_count = 0

        for idx in similar_indices:
            if idx == target_idx:
                continue  # Skip the target itself
            if labels[idx] == labels[target_idx]:
                relevant_count += 1
            if relevant_count == k:  # Stop once top-K are processed
                break

        return relevant_count / k

    # Compute Precision@K for all images
    k = 5
    precision_scores = []
    for target_idx in range(len(C)):
        precision = precision_at_k(W, labels, target_idx, k)
        precision_scores.append(precision)
        print(f"Target Image {target_idx}: Precision@{k} = {precision:.2f}")

    # Calculate and display the average Precision@K
    average_precision = np.mean(precision_scores)
    print(f"Average Precision@{k}: {average_precision:.2f}")

    # Select a target image (e.g., the first image in the dataset)
    target_idx =3

    # You can change this index or loop through multiple targets
    print(f" Target Image: {target_idx}")

    # Sort the images based on their similarity to the target image
    similar_indices = np.argsort(-W[target_idx])  # Descending order of similarity

    # Display the target image and its most similar images
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))  # Show target + top 5 similar images

    # Show the target image
    axes[0].imshow(C[target_idx])
    axes[0].set_title("Target Image")
    axes[0].axis("off")

    # Show the top 5 similar images excluding the target
    rank = 1
    for idx in similar_indices:
        if idx == target_idx:
            continue  # Skip the target image
        axes[rank].imshow(C[idx])
        axes[rank].set_title(f"Rank {rank} Sim: {W[target_idx, idx]:.2f}")
        axes[rank].axis("off")
        rank += 1
        if rank > 5:  # Limit to top 5 similar images
            break

    plt.tight_layout()
    plt.show()

    print("Most Similar Images:")
    for rank, idx in enumerate(similar_indices[1:], 1):  # Skip self-similarity
        if idx == target_idx:
            continue
        print(f"Rank {rank}: Image {idx} (Similarity: {W[target_idx, idx]:.4f})")
