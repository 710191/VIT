import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import OverfitDataset
from models.model import FFTPredictor, compute_fft
from torchvision.utils import save_image

def train(config):
    """Train the model to overfit on a single image."""
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_path = config['data']['image_path']
    dataset = OverfitDataset(
        image_path,
        size=config['data']['size']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    print(f"Loaded image from: {image_path}")
    
    # Initialize model
    model = FFTPredictor(
        checkpoint=config['model']['checkpoint'],
        hidden_dim=config['model']['hidden_dim'],
        output_channels=config['model']['output_channels']
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Setup scheduler
    scheduler = None
    if config['training']['scheduler']['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['gamma']
        )
    elif config['training']['scheduler']['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    # Setup loss function
    loss_type = config['training']['loss_type']
    criterion = nn.MSELoss()
    print(f"Using loss type: {loss_type}")
    
    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, rgb_image in enumerate(dataloader):
            rgb_image = rgb_image.to(device)
            
            # Forward pass
            fft_pred = model(rgb_image)
            
            # Compute loss based on loss_type
            if loss_type == 'fft':
                # L2 loss on FFT coefficients
                fft_target = compute_fft(rgb_image)
                loss = criterion(fft_pred, fft_target)
            elif loss_type == 'rgb':
                # L2 loss on reconstructed RGB image
                rgb_recon = model.predict_rgb(fft_pred)
                loss = criterion(rgb_recon, rgb_image)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Save some example images for visualization
            if batch_idx == 0 and epoch % config['training']['log_interval'] == 0:
                target_rgb = rgb_image.detach().cpu()
                recon_rgb = model.predict_rgb(fft_pred).detach().cpu()
                comparison = torch.cat([target_rgb, recon_rgb], dim=0)
                save_image(
                    comparison,
                    os.path.join(checkpoint_dir, f'epoch_{epoch:04d}_comparison.png'),
                    nrow=rgb_image.size(0)
                )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        if epoch % config['training']['log_interval'] == 0:
            avg_loss = epoch_loss / len(dataloader)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch:4d}/{config['training']['epochs']:4d}] | "
                  f"Loss: {avg_loss:.6f} | LR: {lr:.6f}")
            
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'overfit_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("=" * 80)
    print("Training completed!")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'overfit_final.pth')
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss / len(dataloader),
        'config': config
    }, final_path)
    print(f"Saved final model to {final_path}")


def main():
    """Main entry point for the training script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train INR-FFT model to overfit on a single image')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("INR-FFT Overfitting Training")
    print("=" * 80)
    print(f"Configuration loaded from: {args.config}\n")
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()
