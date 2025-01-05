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
    tau = np.zeros_like(S)
    for i in range(n):
        tau[i] = np.argsort(np.argsort(-S[i]))
    S_normalized = 2 * n - (tau + tau.T)
    return S_normalized

def construct_hypergraph(V, k=10):
    if V.shape[0] == 0:
        raise ValueError("Feature matrix 'V' is empty. Ensure images are loaded and features are extracted.")
    n = V.shape[0]
    S = cosine_similarity(V)
    E = {}
    for i in range(n):
        neighbors = np.argsort(-S[i])[:k]
        E[i] = neighbors
    return E

def compute_hyperedge_similarities(E, V):
    n = len(E)
    Sh = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                shared = set(E[i]).intersection(E[j])
                Sh[i, j] = len(shared)
    return Sh

def compute_cartesian_product(E):
    C = {}
    for ei, vertices in E.items():
        for vi in vertices:
            for vj in vertices:
                if vi != vj:
                    C[(vi, vj)] = C.get((vi, vj), 0) + 1
    return C

def compute_hypergraph_similarity(E, V):
    Sh = compute_hyperedge_similarities(E, V)
    C = compute_cartesian_product(E)
    n = len(V)
    W = np.zeros((n, n))
    for (vi, vj), weight in C.items():
        W[vi, vj] += weight
    W += Sh
    return W

def load_images_from_urls(urls):
    images = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            images.append(img)
        else:
            print(f"Failed to fetch image from URL: {url}")
    return images

if __name__ == "__main__":
    urls = [
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",  # Lena
    ]
    C = load_images_from_urls(urls)
    if len(C) == 0:
        raise ValueError("No images were successfully loaded from URLs.")
    V = extract_features(C)
    print("Extracted Features Shape:", V.shape)
    E = construct_hypergraph(V, k=10)
    W = compute_hypergraph_similarity(E, V)
    print(W)
