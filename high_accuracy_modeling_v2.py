import os
import sys
import gc
import torch
import torch.nn as nn
import requests
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2

from emotion_modeling_utils import (
    build_efficientnet_model,
    build_weighted_loss,
    evaluate_model,
    evaluate_with_tta,
    get_device,
    predict_with_confidence,
    prepare_efficientnet_dataloaders,
    seed_everything,
    train_model_v2,
    unfreeze_all,
    unfreeze_top_blocks,
)


os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
PROJECT_ROOT = '/Users/samikhasrinivasan/Downloads/deep-learning-project-main'
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)

def download_weights():
    """Ensures pretrained weights are available locally."""
    os.makedirs('model_weights', exist_ok=True)
    url = 'https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth'
    dest = 'model_weights/efficientnet_b2.pth'
    if not os.path.exists(dest):
        print(f'Downloading weights to {dest}...')
        r = requests.get(url, verify=False, stream=True)
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Download complete.')

def main():
    # Setup
    seed_everything(42)
    device = get_device()
    print(f'Using device: {device}')
    
    IMAGE_SIZE = 260
    BATCH_SIZE = 32
    num_classes = 7

    # Data Prep
    print("Loading datasets...")
    data = prepare_efficientnet_dataloaders(
        train_dir='archive/train',
        test_dir='archive/test',
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
        image_size=IMAGE_SIZE,
        seed=42,
    )
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    class_weights = data['class_weights']
    idx_to_class = data['idx_to_class']

    # Model Initialization
    download_weights()
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()

    model = build_efficientnet_model(num_classes=num_classes)
    model = model.to(device)
    print(f'Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Stage 1: Training head
    print("\n--- STAGE 1: TRAINING HEAD ---")
    STAGE1_EPOCHS = 5
    criterion = build_weighted_loss(class_weights, device, label_smoothing=0.1)
    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=STAGE1_EPOCHS, eta_min=1e-5)

    history1 = train_model_v2(
        model, train_loader, val_loader, criterion, optimizer1, device,
        epochs=STAGE1_EPOCHS, num_classes=num_classes, scheduler=scheduler1,
        use_mixup=True, mixup_alpha=0.4, grad_clip=1.0
    )

    # STAGE 2: Unfreeze Top 3 Blocks
    print("\n--- STAGE 2: FINE-TUNING TOP BLOCKS ---")
    STAGE2_EPOCHS = 10
    unfreeze_top_blocks(model, num_blocks=3)
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=STAGE2_EPOCHS, eta_min=1e-6)

    history2 = train_model_v2(
        model, train_loader, val_loader, criterion, optimizer2, device,
        epochs=STAGE2_EPOCHS, num_classes=num_classes, scheduler=scheduler2,
        use_mixup=True, mixup_alpha=0.3, grad_clip=1.0
    )

    # STAGE 3: Full Fine-Tuning 
    print("\n--- STAGE 3: FULL NETWORK FINE-TUNING ---")
    # Reduce batch size for full unfreeze if memory is tight
    data_low_mem = prepare_efficientnet_dataloaders(
        train_dir='archive/train', test_dir='archive/test',
        batch_size=16, val_ratio=0.2, image_size=IMAGE_SIZE, seed=42
    )
    
    STAGE3_EPOCHS = 15
    unfreeze_all(model)
    optimizer3 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=STAGE3_EPOCHS, eta_min=1e-7)

    history3 = train_model_v2(
        model, data_low_mem['train_loader'], data_low_mem['val_loader'], criterion, optimizer3, device,
        epochs=STAGE3_EPOCHS, num_classes=num_classes, scheduler=scheduler3,
        use_mixup=True, mixup_alpha=0.2, grad_clip=0.5
    )

    # Final Evaluation
    print("\n--- FINAL EVALUATION ---")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f'Standard Test Accuracy: {test_acc*100:.2f}%')

    tta_loss, tta_acc = evaluate_with_tta(model, test_loader, criterion, device, tta_n=5, image_size=IMAGE_SIZE)
    print(f'TTA Test Accuracy:      {tta_acc*100:.2f}%')

    # Save Results & Model
    os.makedirs('model_weights', exist_ok=True)
    torch.save(model.state_dict(), 'model_weights/efficientnet_b2_emotion_v2.pth')
    print('Model saved to model_weights/efficientnet_b2_emotion_v2.pth')

    # training curves plot
    all_val_acc = history1['val_acc'] + history2['val_acc'] + history3['val_acc']
    plt.figure(figsize=(10, 6))
    plt.plot(all_val_acc, label='Validation Accuracy')
    plt.title('Accuracy through all 3 Stages')
    plt.savefig('training_curves_v2.png')
    print("Training plot saved as training_curves_v2.png")

if __name__ == "__main__":
    main()
