
# Import libraries (Kaggle có sẵn hầu hết)
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Thiết lập device (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import glob

DATA_PATH = "/kaggle/input/skin-cancer-mnist-ham10000"
METADATA_PATH = os.path.join(DATA_PATH, "HAM10000_metadata.csv")

# Load metadata
df = pd.read_csv(METADATA_PATH)
print(f"Dataset shape: {df.shape}")  # ~10k rows

# FIXED: Build a dict of image_id -> actual full path by globbing both parts
image_paths = {}
for part_folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
    part_dir = os.path.join(DATA_PATH, part_folder)
    if os.path.exists(part_dir):
        for img_file in glob.glob(os.path.join(part_dir, '*.jpg')):
            img_id = os.path.splitext(os.path.basename(img_file))[0]
            image_paths[img_id] = img_file
    else:
        print(f"Warning: Folder {part_dir} not found. Check dataset mounting.")

# Assign paths to DF (will be NaN for any missing images)
df['image_path'] = df['image_id'].map(image_paths.get)

# Map labels (7 classes)
label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
df['label'] = df['dx'].map(label_map)

# Drop rows with missing paths or labels
df = df.dropna(subset=['image_path', 'label'])
print(f"Cleaned dataset shape: {df.shape}")
print(f"Classes: {df['dx'].value_counts()}")

# ============================================
# CRITICAL FIX: Split by lesion_id to prevent data leakage (Review 2)
# Same lesion should NOT appear in both train and test sets
# ============================================

# Get unique lesion_ids
unique_lesions = df['lesion_id'].unique()
print(f"Total unique lesions: {len(unique_lesions)}")

# Split lesion_ids into train/test (80/20)
train_lesions, test_lesions = train_test_split(
    unique_lesions, test_size=0.2, random_state=42
)

# Create train/test DataFrames based on lesion_id split
train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
test_df = df[df['lesion_id'].isin(test_lesions)].reset_index(drop=True)

print(f"Train shape: {train_df.shape} ({len(train_lesions)} lesions)")
print(f"Test shape: {test_df.shape} ({len(test_lesions)} lesions)")

# Further split train into train_sub and val (also by lesion_id)
train_lesions_array = train_df['lesion_id'].unique()
train_sub_lesions, val_lesions = train_test_split(
    train_lesions_array, test_size=0.1, random_state=42
)

train_sub_df = train_df[train_df['lesion_id'].isin(train_sub_lesions)].reset_index(drop=True)
val_df = train_df[train_df['lesion_id'].isin(val_lesions)].reset_index(drop=True)

print(f"Train_sub shape: {train_sub_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
print("✓ No data leakage: Train/Val/Test split by unique lesion_id")

# ============================================
# FIXED: Handle Imbalance - Compute class weights from train_df
# ============================================

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

# Transforms cho ResNet (resize 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx]['label']
        return image, label


# Partition Non-IID theo 'localization' (simulate patients, 10 clients)
num_clients = 10
client_data = []
locations = train_df['localization'].unique()[:num_clients]  # Use first 10 locations as "clients"
for loc in locations:
    client_df = train_df[train_df['localization'] == loc].sample(frac=1)  # Shuffle
    client_data.append(client_df)

print(f"Partitioned into {num_clients} clients (Non-IID by location)")
print(f"Client sizes: {[len(cd) for cd in client_data]}")  # Check balance

# ============================================
# ADDED: Detailed Non-IID Statistics (Review 1)
# ============================================
print("\n=== Non-IID Data Distribution Analysis ===")
client_stats = []
for i, (loc, client_df) in enumerate(zip(locations, client_data)):
    class_dist = client_df['dx'].value_counts().to_dict()
    stats = {
        'Client': f'Client {i+1}',
        'Location': loc,
        'Total Samples': len(client_df),
        'Unique Lesions': client_df['lesion_id'].nunique(),
        **{cls: class_dist.get(cls, 0) for cls in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']}
    }
    client_stats.append(stats)

client_stats_df = pd.DataFrame(client_stats)
print(client_stats_df.to_string(index=False))
client_stats_df.to_csv('/kaggle/working/client_non_iid_distribution.csv', index=False)

# Visualize Non-IID distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Sample distribution per client
axes[0].bar(range(num_clients), [len(cd) for cd in client_data], color='steelblue')
axes[0].set_xlabel('Client ID')
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Data Heterogeneity: Sample Count per Client')
axes[0].set_xticks(range(num_clients))
axes[0].set_xticklabels([f'C{i+1}' for i in range(num_clients)])
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Class distribution heatmap
class_matrix = client_stats_df[['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']].values
sns.heatmap(class_matrix.T, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            xticklabels=[f'C{i+1}' for i in range(num_clients)],
            yticklabels=['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'])
axes[1].set_title('Class Distribution Heatmap (Non-IID)')
axes[1].set_xlabel('Client ID')
axes[1].set_ylabel('Disease Class')

plt.tight_layout()
plt.savefig('/kaggle/working/non_iid_analysis.png', dpi=600, bbox_inches='tight')
plt.show()
print("✓ Non-IID statistics saved to 'client_non_iid_distribution.csv'")

from sklearn.utils.class_weight import compute_class_weight

# Model: ResNet-18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # 7 classes
model = model.to(device)


class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer giữ nguyên
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Model initialized.")

# ============================================
# SUBSECTION 3.1: Overall Architecture
# ============================================
# Simulate Figure 1: Vẽ flowchart đơn giản (save as architecture.png)

import matplotlib.pyplot as plt

def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # COLORS
    c_client = "#b3d9ff"
    c_local = "#b3ffcc"
    c_update = "#ffff99"
    c_server = "#ffcc80"
    c_global = "#ff9999"
    c_infer = "#ffb3e6"

    # Y positions
    Y = [0.75, 0.50, 0.25]

    # ---- CLIENTS ----
    for i, y in enumerate(Y, start=1):
        ax.text(0.05, y, f"Client {i}\nHospital {chr(64 + i)}\nLocal Data (Non-IID)",
                ha='center', bbox=dict(boxstyle="round,pad=0.4", facecolor=c_client))
        ax.annotate("", xy=(0.18, y), xytext=(0.10, y),
                    arrowprops=dict(arrowstyle="->"))

    # ---- LOCAL TRAIN ----
    for y in Y:
        ax.text(0.25, y, "Local Training\n(E Local Epochs)", ha='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=c_local))
        ax.annotate("", xy=(0.38, y), xytext=(0.30, y),
                    arrowprops=dict(arrowstyle="->"))

    # ---- SEND UPDATES ----
    for y in Y:
        ax.text(0.45, y, "Send Updates\n(Weights/Gradients)", ha='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=c_update))
        ax.annotate("", xy=(0.60, 0.50), xytext=(0.52, y),
                    arrowprops=dict(arrowstyle="->"))

    # ---- CENTRAL SERVER ----
    ax.text(0.65, 0.50, "Central Server\nFedAvg Aggregation",
            ha='center', bbox=dict(boxstyle="round,pad=0.4", facecolor=c_server))
    ax.annotate("", xy=(0.78, 0.50), xytext=(0.70, 0.50),
                arrowprops=dict(arrowstyle="->"))

    # ---- GLOBAL MODEL ----
    ax.text(0.85, 0.50, "Global Model\n(Updated Each Round)", ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor=c_global))

    # ------------------------------------------------------------------
    # >>> FIXED BROADCAST ARROWS (vòng xuống dưới, không chồng lên gì) <<<
    # ------------------------------------------------------------------
    for i, y in enumerate(Y):
        curve_strength = -0.6 - i * 0.15   # cong mạnh xuống
        ax.annotate("",
                    xy=(0.25, y),
                    xytext=(0.85, 0.50),
                    arrowprops=dict(
                        arrowstyle="->",
                        linestyle="--",
                        connectionstyle=f"arc3,rad={curve_strength}",
                        color="black"
                    )
        )

    # ---- INFERENCE + GRAD-CAM ----
    ax.text(0.85, 0.20,
            "Inference on New Images\n↓\nGrad-CAM / Grad-CAM++\n(Explainability Heatmap)",
            ha='center', bbox=dict(boxstyle="round,pad=0.4", facecolor=c_infer))

    ax.annotate("", xy=(0.85, 0.28), xytext=(0.85, 0.45),
                arrowprops=dict(arrowstyle="->"))

    ax.set_title("Figure 1: Proposed Federated Learning Architecture with Explainability\n", fontsize=15)

    plt.savefig("/kaggle/working/architecture.png", dpi=600, bbox_inches='tight')
    plt.show()

draw_architecture()

import copy

# ============================================
# SUBSECTION 3.2: Privacy-Preserving Training with FL (FedAvg + FedProx)
# ============================================
def local_train(model, dataloader, epochs, optimizer, use_fedprox=False, global_model=None, mu=0.01):
    """
    Local training with optional FedProx regularization (Review 2, 3)
    Args:
        use_fedprox: If True, add proximal term to handle Non-IID data
        global_model: Reference global model for proximal term
        mu: Proximal term coefficient (default: 0.01)
    """
    model.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # FedProx: Add proximal term ||w - w_global||^2
            if use_fedprox and global_model is not None:
                proximal_term = 0.0
                for w, w_g in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_g).norm(2)
                loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    return model

def fedavg_aggregate(client_models, client_sizes, global_model):
    new_global = copy.deepcopy(global_model)
    total = sum(client_sizes)
    with torch.no_grad():
        for key in new_global.state_dict().keys():
            new_global.state_dict()[key].data.zero_()
            # FIXED: Skip non-float tensors (e.g., num_batches_tracked which is int64)
            if new_global.state_dict()[key].dtype != torch.float32:
                continue
            for m, size in zip(client_models, client_sizes):
                new_global.state_dict()[key].data += (m.state_dict()[key] * (size / total))
    return new_global

print("Functions defined.")

# ---------- INSERT: logging lists & helper functions ----------
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

train_losses_per_round = []
val_losses_per_round = []
train_accs_per_round = []
val_accs_per_round = []

# If you prefer different ordering in plots, reorder class_names accordingly:
class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']  # <-- user asked rows in order: nv, mel, bkl, bcc, akiec, vasc, df

def evaluate_model_get_metrics(model, df_eval, batch_size=64, device=device, return_probs=False):
    model.eval()
    dataset = HAM10000Dataset(df_eval, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    preds = []
    trues = []
    probs_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item() * images.size(0))
            prob = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())
            probs_list.append(prob.cpu().numpy())
    avg_loss = sum(losses) / len(df_eval)
    acc = accuracy_score(trues, preds)
    probs_all = np.vstack(probs_list) if len(probs_list) > 0 else None
    if return_probs:
        return avg_loss, acc, preds, trues, probs_all
    else:
        return avg_loss, acc, preds, trues
# ---------------------------------------------------------------------------------------

# FL Training Loop
num_rounds = 30
local_epochs = 5
batch_size = 32
USE_FEDPROX = True  # Set to True for FedProx, False for FedAvg (Review 2, 3)
MU = 0.01  # FedProx proximal term coefficient

print(f"\n=== Starting FL Training with {'FedProx' if USE_FEDPROX else 'FedAvg'} ===")
if USE_FEDPROX:
    print(f"FedProx μ (mu) = {MU}")

# Copy initial model cho clients
client_models = []
for _ in range(num_clients):
    client_model = models.resnet18(pretrained=True)
    num_ftrs = client_model.fc.in_features
    client_model.fc = nn.Linear(num_ftrs, 7)
    client_model.load_state_dict(model.state_dict())
    client_model = client_model.to(device)
    client_models.append(client_model)

for round_num in range(num_rounds):
    print(f"Round {round_num+1}/{num_rounds}")
    selected_clients = list(range(num_clients))
    updated_models = []
    selected_sizes = []
    
    # Store global model for FedProx
    global_model_round = copy.deepcopy(model) if USE_FEDPROX else None
    
    for client_idx in selected_clients:
        client_df = client_data[client_idx]
        if len(client_df) < batch_size:
            continue
        train_df_client = client_df.sample(frac=1).reset_index(drop=True)
        train_dataset = HAM10000Dataset(train_df_client, transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        local_optimizer = optim.SGD(client_models[client_idx].parameters(), lr=0.001, momentum=0.9)
        local_model = client_models[client_idx]
        # Local train with FedProx option
        local_model = local_train(local_model, train_loader, local_epochs, local_optimizer, 
                                 use_fedprox=USE_FEDPROX, global_model=global_model_round, mu=MU)
        updated_models.append(local_model)
        selected_sizes.append(len(client_df))
    if updated_models:
        model = fedavg_aggregate(updated_models, selected_sizes, model)
        for i in range(num_clients):
            client_models[i].load_state_dict(model.state_dict())

    # ---- EVALUATE global model on train_sub and val after aggregation ----
    train_loss, train_acc, _, _, _ = evaluate_model_get_metrics(model, train_sub_df, batch_size=64, return_probs=True)
    val_loss, val_acc, _, _, _ = evaluate_model_get_metrics(model, val_df, batch_size=64, return_probs=True)
    train_losses_per_round.append(train_loss)
    val_losses_per_round.append(val_loss)
    train_accs_per_round.append(train_acc)
    val_accs_per_round.append(val_acc)
    print(f"After Round {round_num+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

print("FL Training completed. Global model ready.")

# Test trên test set (KHÔNG dùng trong training)
test_dataset = HAM10000Dataset(test_df, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
preds = []
trues = []
probs_list = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        prob = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        probs_list.append(prob.cpu().numpy())

probs_all = np.vstack(probs_list)
acc = accuracy_score(trues, preds)
print(f"Client sizes: {[len(cd) for cd in client_data]}")
print(f"Global Model Test Accuracy: {acc:.4f}")

# ============================================
# SUBSECTION 3.3: Generating Visual Explanations with Grad-CAM++
# ============================================
# Import additional libraries for Grad-CAM

from torch.autograd import Variable
import cv2 # For heatmap overlay (Kaggle has OpenCV)
# Grad-CAM++ Implementation
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
       
        # Hook for gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
       
        # Hook for activations
        def forward_hook(module, input, output):
            self.activations = output
       
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
   
    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
       
        if target_class is None:
            target_class = output.argmax().item()
       
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class].sum()
        class_score.backward(retain_graph=True)
       
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        b, c, h, w = gradients.shape
       
        # Grad-CAM++ weighting: alpha_{k}^{c} = \frac{ \sum_i \sum_j \frac{\partial^2 y^c}{\partial A_i^k \partial A_j^k} A_i^k A_j^k }{ \sum_i \sum_j \frac{\partial y^c}{\partial A_i^k} \frac{\partial y^c}{\partial A_j^k} A_i^k A_j^k }
        # Simplified computation per channel
        cam = torch.zeros(h, w, device=input_tensor.device)
        for k in range(c):
            grad_k = gradients[0, k]  # [H, W]
            act_k = activations[0, k]  # [H, W]
           
            # Second order gradients approximation via element-wise
            grad2 = grad_k.pow(2)
            alpha_k = (grad2 * act_k).sum() / ((grad_k ** 2 * act_k ** 2).sum() + 1e-8)
           
            cam += alpha_k * act_k
       
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
       
        return cam.cpu().detach().numpy(), target_class
   
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
# Target layer for ResNet18: last conv layer in layer4
target_layer = model.layer4[-1].conv2 # ResNet18 structure
# Select 3 random samples from test set
np.random.seed(42) # For reproducibility
random_indices = np.random.choice(len(test_df), 3, replace=False)
samples = test_df.iloc[random_indices].reset_index(drop=True)
# Inverse label map for display
inv_label_map = {v: k for k, v in label_map.items()}
# Generate and visualize Grad-CAM++ for each sample
gradcam = GradCAMpp(model, target_layer)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Grad-CAM++ Visual Explanations for 3 Random Test Samples', fontsize=16)
for i in range(3):
    sample_idx = samples.iloc[i]
    img_path = sample_idx['image_path']  # Fixed: removed redundant sample_path
    true_label = int(sample_idx['label'])
    true_class = inv_label_map[true_label]
   
    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
   
    # Get prediction and CAM
    cam, pred_class_idx = gradcam(input_tensor)
    pred_class = inv_label_map[pred_class_idx]
    pred_prob = F.softmax(model(input_tensor), dim=1)[0, pred_class_idx].item()
   
    # Original image (denormalize for display)
    orig_image = np.array(image.resize((224, 224)))
    orig_image = (orig_image / 255.0) # Normalize to [0,1]
   
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (224, 224))
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
   
    # Overlay
    overlay = heatmap * 0.4 + orig_image * 0.6 # Blend
    overlay = np.clip(overlay, 0, 1)
   
    # Plot
    axes[i, 0].imshow(orig_image)
    axes[i, 0].set_title(f'Original\nTrue: {true_class}')
    axes[i, 0].axis('off')
   
    axes[i, 1].imshow(heatmap)
    axes[i, 1].set_title('Grad-CAM++ Heatmap')
    axes[i, 1].axis('off')
   
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f'Overlay')
    axes[i, 2].axis('off')
   
    axes[i, 3].bar(['True', 'Pred'], [1, pred_prob], color=['blue', 'red'])
    axes[i, 3].set_title(f'Prediction: {pred_class} ({pred_prob:.2f})')
    axes[i, 3].set_ylim(0, 1)
   
plt.tight_layout()
plt.savefig('/kaggle/working/gradcam_samples.png', dpi=600, bbox_inches='tight')
plt.show()
# Clean up hooks
gradcam.remove_hooks()
print("Grad-CAM++ visualizations generated and saved as 'gradcam_samples.png'.")
print("Use in LaTeX: \\includegraphics[width=\\textwidth]{gradcam_samples.png}")

# ============================================
# ADDED: Quantitative Grad-CAM++ Evaluation (Review 2)
# Insertion & Deletion Metrics
# ============================================
print("\n=== Quantitative Grad-CAM++ Evaluation ===")

def compute_insertion_deletion_scores(model, image_tensor, cam, target_class, steps=50):
    """
    Insertion: Start with blurred image, progressively insert pixels by importance
    Deletion: Start with original image, progressively delete pixels by importance
    """
    model.eval()
    # Get original prediction score
    with torch.no_grad():
        output_orig = model(image_tensor)
        prob_orig = F.softmax(output_orig, dim=1)[0, target_class].item()
    
    # Prepare blurred version
    blurred = transforms.GaussianBlur(kernel_size=21, sigma=10)(image_tensor)
    
    # Sort pixels by CAM importance
    cam_flat = cam.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1]  # Descending order
    
    insertion_scores = []
    deletion_scores = []
    
    step_size = max(1, len(sorted_indices) // steps)  # Ensure step size >= 1
    for i in range(0, len(sorted_indices), step_size):
        # Insertion: add top-i pixels from original to blurred
        inserted = blurred.clone()
        mask = np.zeros_like(cam_flat, dtype=bool)
        mask[sorted_indices[:i+1]] = True
        mask_2d = mask.reshape(cam.shape)
        mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0).to(device)
        mask_tensor = F.interpolate(mask_tensor, size=(224, 224), mode='nearest').squeeze()
        for c in range(3):
            inserted[0, c] = inserted[0, c] * (1 - mask_tensor) + image_tensor[0, c] * mask_tensor
        
        with torch.no_grad():
            out_ins = model(inserted)
            insertion_scores.append(F.softmax(out_ins, dim=1)[0, target_class].item())
        
        # Deletion: remove top-i pixels from original
        deleted = image_tensor.clone()
        for c in range(3):
            deleted[0, c] = deleted[0, c] * (1 - mask_tensor) + blurred[0, c] * mask_tensor
        
        with torch.no_grad():
            out_del = model(deleted)
            deletion_scores.append(F.softmax(out_del, dim=1)[0, target_class].item())
    
    # Compute AUC
    insertion_auc = np.trapz(insertion_scores, dx=1.0/len(insertion_scores))
    deletion_auc = np.trapz(deletion_scores, dx=1.0/len(deletion_scores))
    
    return insertion_auc, deletion_auc, insertion_scores, deletion_scores

# Evaluate on the 3 samples
gradcam_metrics = []
for i in range(3):
    sample_idx = samples.iloc[i]
    img_path = sample_idx['image_path']
    true_label = int(sample_idx['label'])
    
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Re-generate CAM
    gradcam_eval = GradCAMpp(model, target_layer)
    cam, pred_class_idx = gradcam_eval(input_tensor)
    
    ins_auc, del_auc, ins_scores, del_scores = compute_insertion_deletion_scores(
        model, input_tensor, cam, pred_class_idx, steps=50
    )
    
    gradcam_metrics.append({
        'Sample': i+1,
        'True Class': inv_label_map[true_label],
        'Pred Class': inv_label_map[pred_class_idx],
        'Insertion AUC': ins_auc,
        'Deletion AUC': del_auc
    })
    
    gradcam_eval.remove_hooks()

metrics_df = pd.DataFrame(gradcam_metrics)
print(metrics_df.to_string(index=False))
metrics_df.to_csv('/kaggle/working/gradcam_quantitative_metrics.csv', index=False)

print(f"\nAverage Insertion AUC: {metrics_df['Insertion AUC'].mean():.4f}")
print(f"Average Deletion AUC: {metrics_df['Deletion AUC'].mean():.4f}")
print("✓ Higher Insertion AUC = better (faster to recover confidence by adding important pixels)")
print("✓ Lower Deletion AUC = better (faster to lose confidence by removing important pixels)")
print("✓ Quantitative metrics saved to 'gradcam_quantitative_metrics.csv'")

# ---------- PLOTTING / SAVING FIGURES ----------
import matplotlib.ticker as mtick
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

os.makedirs('/kaggle/working/figs', exist_ok=True)

# 1) Training & Validation loss per epoch (round)
plt.figure(figsize=(8,6))
epochs = np.arange(1, len(train_losses_per_round)+1)
plt.plot(epochs, train_losses_per_round, marker='o', label='Train Loss')
plt.plot(epochs, val_losses_per_round, marker='o', label='Val Loss')
plt.xlabel('Round / Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Round')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.savefig('/kaggle/working/figs/loss_curve.png', dpi=600, bbox_inches='tight')
plt.show()

# 2) Training & Validation accuracy per epoch (round)
plt.figure(figsize=(8,6))
plt.plot(epochs, train_accs_per_round, marker='o', label='Train Acc')
plt.plot(epochs, val_accs_per_round, marker='o', label='Val Acc')
plt.xlabel('Round / Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy per Round')
plt.xticks(epochs)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.savefig('/kaggle/working/figs/acc_curve.png', dpi=600, bbox_inches='tight')
plt.show()

# 3) Classification report -> heatmap
report_dict = classification_report(trues, preds, target_names=class_names, output_dict=True)
# Convert to DataFrame
report_df = pd.DataFrame(report_dict).T
# Reorder rows to required order and ensure accuracy, macro avg, weighted avg included
# sklearn includes 'accuracy','macro avg','weighted avg' keys already
# But ensure row order: classes (nv, mel, bkl, bcc, akiec, vasc, df), accuracy, macro avg, weighted avg
row_order = class_names + ['accuracy', 'macro avg', 'weighted avg']
report_df = report_df.reindex(row_order)
# For heatmap, we take precision/recall/f1-score columns; accuracy row has only 'precision' etc maybe NaN for support — fix:
heatmap_df = report_df[['precision', 'recall', 'f1-score']].fillna(0)
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='viridis', cbar=True)
plt.title('Classification Report (precision / recall / f1-score)')
plt.savefig('/kaggle/working/figs/class_report_heatmap.png', dpi=600, bbox_inches='tight')
plt.show()

# 4) Confusion Matrix (not normalized and normalized)
# Instead build mapping from your label numbers to the ordering you chose (nv, mel, bkl, bcc, akiec, vasc, df)
# According to your label_map: {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
# Desired display order (nv, mel, bkl, bcc, akiec, vasc, df) corresponds to label indices: [5,4,2,1,0,6,3]
labels_order = [5,4,2,1,0,6,3]
cm = confusion_matrix(trues, preds, labels=labels_order)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=['nv','mel','bkl','bcc','akiec','vasc','df'], yticklabels=['nv','mel','bkl','bcc','akiec','vasc','df'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Test set)')
plt.savefig('/kaggle/working/figs/confusion_matrix.png', dpi=600, bbox_inches='tight')
plt.show()

# 5) Multi-class ROC curves + AUC
n_classes = 7
# Binarize true labels with classes in index order [0..6]
y_test_bin = label_binarize(trues, classes=[0,1,2,3,4,5,6])  # shape (n_samples, n_classes)
# If probs_all has columns in order [0..6], ok. If your model outputs correspond to that label mapping, fine.
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro-average
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro-average (compute all fpr points)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10,8))
# Plot per-class
for i, cls_label in enumerate(['nv','mel','bkl','bcc','akiec','vasc','df']):
    plt.plot(fpr[i], tpr[i], lw=1.5, label=f'{cls_label} (AUC = {roc_auc[i]:.2f})')
# plot micro & macro
plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle=':', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle='--', linewidth=2)
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Multi-class ROC curves')
plt.legend(loc='lower right', fontsize='small')
plt.savefig('/kaggle/working/figs/multiclass_roc_auc.png', dpi=600, bbox_inches='tight')
plt.show()

print("Saved figures in /kaggle/working/figs/")

# Log training results to CSV
logs_df = pd.DataFrame({
    'round': np.arange(1, len(train_losses_per_round) + 1),
    'train_loss': train_losses_per_round,
    'val_loss': val_losses_per_round,
    'train_acc': train_accs_per_round,
    'val_acc': val_accs_per_round
})
logs_df.to_csv('/kaggle/working/logs_fl_training.csv', index=False)
print("Training logs saved to '/kaggle/working/logs_fl_training.csv'")

# Tạo figure với twin axes
fig, ax1 = plt.subplots(figsize=(8, 6))  # Kích thước vuông vức cho single plot

# Plot Loss trên ax1 (trục y trái)
line1 = ax1.plot(epochs, train_losses_per_round, marker='o', label='Train Loss', linewidth=2, color='blue')
line2 = ax1.plot(epochs, val_losses_per_round, marker='o', label='Val Loss', linewidth=2, color='lightblue')
ax1.set_xlabel('Communication Round', fontsize=11)
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Global Model Convergence: Loss and Accuracy', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Tạo ax2 (twinx) cho Accuracy (trục y phải)
ax2 = ax1.twinx()
line3 = ax2.plot(epochs, train_accs_per_round, marker='s', label='Train Acc', linewidth=2, color='red')
line4 = ax2.plot(epochs, val_accs_per_round, marker='s', label='Val Acc', linewidth=2, color='orange')
ax2.set_ylabel('Accuracy (%)', fontsize=11, color='red')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Hiển thị % cho accuracy
ax2.tick_params(axis='y', labelcolor='red')
ax2.tick_params(axis='both', which='major', labelsize=10)

# Kết hợp legend từ cả hai axes
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)

# Tight layout và save as PDF
plt.tight_layout(pad=1.5)
plt.savefig('/kaggle/working/figs/acc_loss_curve.pdf', dpi=600, bbox_inches='tight', format='pdf')
plt.show()

print("Combined acc_loss_curve.pdf saved in /kaggle/working/figs/. Ready for LaTeX insertion!")
# ---------------------------------------------------------------------------------------