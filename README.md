# Deep Learning Project: Facial Emotion Recognition

**Authors:**  
Samikha Srinivasan  
Sanskriti Malakar  
Krisha Jhala  
Sanjana Nuti  

This project focuses on **facial emotion recognition** using the FER-2013 dataset, classifying grayscale facial images into one of seven emotion categories:

`angry` | `disgust` | `fear` | `happy` | `neutral` | `sad` | `surprise`

The system is designed as a **decision-support tool** rather than an authoritative predictor, with explicit ethical safeguards including uncertainty flags, confidence estimation, and Grad-CAM interpretability.

---

## Repository Structure

| File | Description |
|------|-------------|
| `data_preprocessing.ipynb` | Dataset loading, augmentation, sanity checks, class imbalance analysis, and dataloader setup |
| `modeling_cnn.ipynb` | Baseline CNN and improved CNN experiments (custom architectures) |
| `high_accuracy_modeling_v2.py` | Transfer learning pipeline using pretrained ResNet18 |
| `emotion_modeling_utils.py` | Shared utilities for training, evaluation, and dataloaders |

---

## Dataset Summary

**FER-2013** is a widely adopted benchmark for facial expression recognition. Images were collected via automated web queries and labeled through crowdsourcing, introducing realistic imperfections such as label noise, low image quality, and class imbalance.

| Split | Number of Images |
|-------|------------------|
| Training | 28,709 |
| Validation | 20% of training set |
| Test | 7,178 |
| Image Resolution | 48 × 48 pixels (grayscale) |
| Number of Classes | 7 |

### Class Imbalance (Training Set)

| Emotion | Training Samples | % of Training Set |
|---------|----------------|-------------------|
| Happy | 7,215 | 28.5% |
| Neutral | 4,965 | 19.6% |
| Sad | 4,830 | 19.1% |
| Fear | 4,097 | 16.2% |
| Angry | 3,995 | 15.8% |
| Surprise | 3,171 | 12.5% |
| Disgust | 436 | 1.7% |

The **happy** class has ~16× more samples than **disgust**, creating a severe class imbalance that we address using **inverse-frequency class weights** and **data augmentation**.

---

## Methodology Summary

### Data Preprocessing Pipeline

| Stage | Operation |
|-------|-----------|
| 1 | Load & verify CSV/image files, check for corrupt images |
| 2 | Normalize pixel values [0, 255] → [0, 1] |
| 3 | Reshape to (48, 48, 1) for grayscale CNN input |
| 4 | **Augmentation** (training only): horizontal flips, random crops, slight rotations |
| 5 | **Class weights**: inverse-frequency weighting to compensate for imbalance |

### Baseline CNN (Custom)
- Two convolutional blocks with ReLU and 2×2 max pooling
- Fully connected layers → softmax over 7 classes
- CrossEntropyLoss with class weights
- Adam optimizer (lr = 0.001)

**Result:** Best validation accuracy = **46.90%**

### Improved CNN (Custom)
- Added batch normalization, dropout, and an extra convolutional layer
- Same loss weighting and optimizer

**Result:** Test accuracy = **38.31%** (regularization hindered learning on small dataset)

### Transfer Learning (Final Model – ResNet18)
Pretrained on ImageNet, adapted to FER-2013 via:

- **3-channel conversion** (grayscale → RGB repeat) for compatibility
- **Weighted cross-entropy loss** for class imbalance
- **Three-stage training**:
  1. Train head only (5 epochs) → 39.08% val acc
  2. Fine-tune top blocks (10 epochs) → 67.29% val acc (+28 pp)
  3. Full fine-tuning (15 epochs) → **69.92% val acc**

- **Cosine learning rate scheduling** per stage
- **Test-time augmentation (TTA)** at inference
- **Uncertainty flag** when max softmax probability < 0.5

---

## Final Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Standard Test Accuracy | **70.87%** |
| TTA Test Accuracy | **71.23%** |
| Macro F1-Score | 0.69 |
| Weighted Precision | 0.71 |
| Weighted Recall | 0.71 |
| Weighted F1 | 0.71 |

**Benchmark context:** Human accuracy on FER-2013 ≈ 65% | Published CNN baselines: 65–72%  
→ Our model is **competitive with state-of-the-art**.

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Test Support |
|---------|-----------|--------|----------|---------------|
| Angry | 0.64 | 0.64 | 0.64 | 958 |
| Disgust | 0.76 | 0.58 | 0.66 | 111 |
| Fear | 0.60 | 0.51 | 0.55 | 1024 |
| Happy | 0.89 | 0.90 | 0.89 | 1774 |
| Neutral | 0.63 | 0.71 | 0.67 | 1233 |
| Sad | 0.59 | 0.59 | 0.59 | 1247 |
| Surprise | 0.81 | 0.82 | 0.81 | 831 |

**Key findings:**
- **Happy (F1=0.89)** : Best performance – large training support + distinctive expression
- **Fear (F1=0.55)** : Worst performance – often misclassified as sad or neutral (visual ambiguity)
- **Disgust** : High precision (0.76) but low recall (0.58) – model is cautious due to small training set
- **Neutral ↔ Sad confusion** : 289 sad images predicted as neutral – low-intensity sadness defaults to neutral

### Confusion Matrix Highlights
- Happy rarely confused with other classes
- Surprise sometimes mislabeled as fear (both involve raised eyebrows/wide eyes)
- Neutral acts as a "default" class, receiving false positives from sad, fear, and angry

---

## Comparison to Baseline

| Model | Best Validation Acc | Test Acc |
|-------|---------------------|----------|
| Baseline CNN (custom) | 46.90% | ~45-50% (estimated) |
| Improved CNN (custom) | 37.63% | 38.31% |
| **Transfer Learning (ResNet18)** | **69.92%** | **70.87%** |

The transfer learning model **substantially outperforms** both custom CNNs, confirming that a pretrained backbone is essential for this dataset given its small image size and noisy labels.

---

## Ethical Considerations

This project explicitly addresses ethical risks in facial emotion recognition:

| Concern | Mitigation |
|---------|-------------|
| **Ambiguity of emotional expression** | Model framed as decision-support, not ground truth |
| **Dataset bias (no demographic metadata)** | Acknowledged limitation; no fairness analysis possible |
| **Class imbalance → performance disparities** | Weighted loss + augmentation; still uneven across classes |
| **Risk of misuse (hiring, medical diagnosis)** | System not designed for diagnostic or evaluative purposes |
| **Lack of transparency** | Grad-CAM visualizations + confidence estimation + uncertainty flag (<0.5) |

---

## Expected Outcomes (from Proposal) vs. Achieved

| Expected | Achieved |
|----------|----------|
| Test accuracy: 70–72% | **70.87%** ✅ |
| Macro F1: 0.68–0.70 | **0.69** ✅ |
| Weighted avg metrics: 0.70–0.72 | **0.71** ✅ |
| Higher accuracy for happy/surprise | Yes (0.89, 0.81) ✅ |
| Lower accuracy for fear/disgust | Yes (0.55, 0.66) ✅ |
| Significant improvement over baseline | Yes (+24 pp) ✅ |

---

## Recommended Run Order

1. `data_preprocessing.ipynb` - verify loading, augmentation, and class distribution
2. `high_accuracy_modeling_v2.py` - main transfer learning pipeline (best results)
3. `modeling_cnn.ipynb` - baseline and improved CNN comparison (for ablation study)

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn
