# Deep Learning Project

This project focuses on facial emotion recognition using grayscale image data with 7 classes:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

The repository currently contains:
- [data_preprocessing.ipynb] for dataset loading, augmentation, sanity checks, class imbalance analysis, and dataloader setup
- [modeling_cnn.ipynb]for the baseline CNN and improved CNN experiments
- [high_accuracy_modeling_v2.py]for the stronger transfer-learning pipeline
- [emotion_modeling_utils.py]for shared training, evaluation, and dataloader utilities

## Dataset Summary

- Training samples: `28,709`
- Test samples: `7,178`
- Validation split: `20%` of the training set
- Image size for CNN experiments: `48 x 48`
- Number of classes: `7`

Observed class imbalance from the training set:
- `happy`: 7215
- `neutral`: 4965
- `sad`: 4830
- `fear`: 4097
- `angry`: 3995
- `surprise`: 3171
- `disgust`: 436

## Modeling Results from `modeling_cnn.ipynb`

### Baseline CNN

Architecture:
- Conv -> ReLU -> MaxPool
- Conv -> ReLU -> MaxPool
- Fully connected layers
- `CrossEntropyLoss(weight=class_weights)`

Training outcome:
- Final training accuracy: `58.24%`
- Best validation accuracy: `46.90%` at epoch 9
- Final validation accuracy: `46.55%`

Epoch trend:
- The baseline CNN improved steadily across training.
- Validation performance rose from `9.61%` in epoch 1 to `46.90%` by epoch 9.
- This was the strongest-performing custom CNN in the current experiments.

### Improved CNN

Architecture additions:
- Batch normalization
- Extra convolution layer
- Dropout regularization
- `CrossEntropyLoss(weight=class_weights)`

Training outcome:
- Final training accuracy: `41.05%`
- Best validation accuracy: `37.63%` at epoch 12
- Test accuracy: `38.31%`
- Test loss: `1.6917`

Epoch trend:
- The improved CNN learned more slowly than the baseline.
- It started at `7.02%` validation accuracy and ended at `37.63%`.
- In this configuration, the added regularization and model depth did not improve performance.

## Confidence Output Example

The notebook converts model logits to probabilities using:

```python
probs = torch.softmax(outputs, dim=1)
```

Example output from the current model:

```text
Probability tensor shape: torch.Size([64, 7])
First sample probabilities: tensor([0.2066, 0.2147, 0.1531, 0.0800, 0.1152, 0.1810, 0.0494])
True: angry | Predicted: disgust | Confidence: 0.2147
```

This shows that the model can output class probabilities for confidence analysis, but current confidence values are still fairly low.

## Interpretation

- The baseline CNN performed better than the regularized CNN.
- The deeper CNN did not improve accuracy under the current settings.
- The original CNN setup was not strong enough to approach very high accuracy.
- The current results suggest that a stronger model family is needed rather than only adding more CNN layers.

## Next Step: `high_accuracy_modeling.ipynb`

The next step is to move from small custom CNNs to transfer learning.

The high-accuracy notebook now uses:
- a pretrained `ResNet18`
- 3-channel image conversion for compatibility with pretrained ImageNet weights
- weighted cross-entropy loss
- staged training

Training plan in that notebook:
1. Train a new classification head while freezing the pretrained backbone.
2. Unfreeze the final backbone block (`layer4`) and fine-tune with a smaller learning rate.
3. Evaluate on the validation and test sets.
4. Produce softmax probabilities for confidence analysis.

Why this is the next step:
- Transfer learning usually performs much better than a small CNN on emotion datasets.
- The pretrained backbone starts from stronger visual features.
- Fine-tuning gives a better chance of improving validation and test accuracy than continuing to tune the original CNNs.

## Run Order

Recommended order:
1. Run [data_preprocessing.ipynb] to verify loading, augmentation, and class distribution.
2. Run [high_accuracy_modeling.ipynb] for the strongest current training pipeline.
3. Use [modeling_cnn.ipynb] when you want the baseline and improved CNN comparison for reporting.

