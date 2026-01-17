import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy import linalg
from typing import Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SimpleMNISTClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x


def train_classifier(
    classifier: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 5,
    device: str = "cpu",
    lr: float = 1e-3
) -> nn.Module:
    classifier.to(device)
    classifier.train()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("training mnist classifier")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        logger.info(f"epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, "
              f"acc={100.*correct/total:.2f}%")
    
    return classifier


@torch.no_grad()
def evaluate_classifier_score(
    generated_samples: torch.Tensor,
    classifier: nn.Module,
    device: str = "cpu"
) -> Tuple[float, np.ndarray]:
    classifier.eval()
    
    samples = generated_samples.to(device)
    outputs = classifier(samples)
    probs = F.softmax(outputs, dim=1)
    
    confidences = probs.max(dim=1)[0].cpu().numpy()
    avg_confidence = confidences.mean()
    
    predicted_classes = outputs.argmax(dim=1).cpu().numpy()
    class_distribution = np.bincount(predicted_classes, minlength=10) / len(predicted_classes)
    
    return avg_confidence, class_distribution


@torch.no_grad()
def calculate_activation_statistics(
    images: torch.Tensor,
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    
    activations = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        feat = model.get_features(batch)
        activations.append(feat.cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Partie imaginaire {m} trop grande")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


@torch.no_grad()
def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    classifier: nn.Module,
    device: str = "cpu",
    batch_size: int = 64
) -> float:
    logger.info("computing fid")
    
    logger.info("extracting features from real images")
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, classifier, device, batch_size
    )
    
    logger.info("extracting features from generated images")
    mu_gen, sigma_gen = calculate_activation_statistics(
        generated_images, classifier, device, batch_size
    )
    
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_score


def get_real_samples(
    num_samples: int = 1000,
    data_dir: str = "./data"
) -> torch.Tensor:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    
    return images


def evaluate_generated_samples(
    generated_samples: torch.Tensor,
    real_samples: Optional[torch.Tensor] = None,
    classifier_path: Optional[str] = None,
    device: str = "cpu"
) -> dict:
    results = {}
    
    classifier = SimpleMNISTClassifier()
    
    if classifier_path and torch.cuda.is_available():
        classifier.load_state_dict(torch.load(classifier_path))
        logger.info("classifier loaded")
    else:
        logger.info("training new classifier")
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        classifier = train_classifier(classifier, dataloader, num_epochs=3, device=device)
    
    classifier.to(device)
    
    avg_conf, class_dist = evaluate_classifier_score(generated_samples, classifier, device)
    results['avg_confidence'] = avg_conf
    results['class_distribution'] = class_dist
    
    logger.info(f"average confidence: {avg_conf:.4f}")
    logger.info("class distribution:")
    for i, prob in enumerate(class_dist):
        logger.info(f"class {i}: {prob*100:.2f}%")
    
    if real_samples is not None:
        fid_score = compute_fid(real_samples, generated_samples, classifier, device)
        results['fid'] = fid_score
        logger.info(f"fid score: {fid_score:.2f}")
    
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    fake_samples = torch.rand(100, 1, 28, 28)
    real_samples = get_real_samples(100)
    
    results = evaluate_generated_samples(fake_samples, real_samples, device=device)
    logger.info(f"results: {results}")
