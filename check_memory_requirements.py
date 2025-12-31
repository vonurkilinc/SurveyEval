#!/usr/bin/env python3
"""Estimate memory requirements for GeoAdapterStep2 training."""
import torch

def estimate_memory_requirements(resolution=256, batch_size=1, use_fp16=True, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
    """Estimate GPU memory requirements for SDXL training."""
    
    # Model sizes (approximate, in GB)
    if use_fp16:
        unet_size = 2.5  # UNet in FP16
        vae_size = 0.3   # VAE encoder + decoder in FP16
        text_encoders_size = 0.7  # Two text encoders in FP16
        model_total = unet_size + vae_size + text_encoders_size
    else:
        unet_size = 5.0  # UNet in FP32
        vae_size = 0.6   # VAE encoder + decoder in FP32
        text_encoders_size = 1.4  # Two text encoders in FP32
        model_total = unet_size + vae_size + text_encoders_size
    
    # Training overhead
    # Latents: batch_size * 4 * (resolution/8)^2 * 2 bytes (FP16) or 4 bytes (FP32)
    latent_size = batch_size * 4 * (resolution // 8) ** 2 * (2 if use_fp16 else 4) / (1024**3)
    
    # Gradients (only for mapper, which is small ~50MB)
    mapper_grad_size = 0.05
    
    # Optimizer states (AdamW: 2x mapper size)
    optimizer_size = mapper_grad_size * 2
    
    # Other overhead (activations, temporary buffers)
    overhead = 0.3
    
    training_total = latent_size + mapper_grad_size + optimizer_size + overhead
    
    total_estimated = model_total + training_total
    
    return {
        "model_total": model_total,
        "training_overhead": training_total,
        "total_estimated": total_estimated,
        "unet_size": unet_size,
        "vae_size": vae_size,
        "text_encoders_size": text_encoders_size,
        "latent_size": latent_size,
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Memory Requirements Estimation for GeoAdapterStep2")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("\n⚠ CUDA not available - cannot check GPU memory")
        gpu_memory = 0
    
    # Check FP16 configuration
    print("\n" + "=" * 70)
    print("Configuration: resolution=256, batch_size=1, FP16=True")
    print("=" * 70)
    
    est_fp16 = estimate_memory_requirements(resolution=256, batch_size=1, use_fp16=True)
    
    print(f"\nModel Components (FP16):")
    print(f"  UNet:           {est_fp16['unet_size']:.2f} GB")
    print(f"  VAE:            {est_fp16['vae_size']:.2f} GB")
    print(f"  Text Encoders:  {est_fp16['text_encoders_size']:.2f} GB")
    print(f"  Model Total:    {est_fp16['model_total']:.2f} GB")
    
    print(f"\nTraining Overhead:")
    print(f"  Latents:        {est_fp16['latent_size']:.3f} GB")
    print(f"  Mapper Grads:  {0.05:.3f} GB")
    print(f"  Optimizer:     {0.1:.3f} GB")
    print(f"  Other:          {0.3:.3f} GB")
    print(f"  Training Total: {est_fp16['training_overhead']:.3f} GB")
    
    print(f"\n{'='*70}")
    print(f"Total Estimated: {est_fp16['total_estimated']:.2f} GB")
    print(f"{'='*70}")
    
    if gpu_memory > 0:
        if est_fp16['total_estimated'] <= gpu_memory * 0.9:  # Leave 10% buffer
            print(f"\n✓ Should fit in {gpu_memory:.2f} GB GPU (with ~{gpu_memory - est_fp16['total_estimated']:.2f} GB buffer)")
        elif est_fp16['total_estimated'] <= gpu_memory:
            print(f"\n⚠ Very tight fit! Estimated {est_fp16['total_estimated']:.2f} GB vs {gpu_memory:.2f} GB GPU")
            print(f"  May work but could fail if memory fragmentation occurs")
        else:
            print(f"\n✗ Will likely fail! Estimated {est_fp16['total_estimated']:.2f} GB > {gpu_memory:.2f} GB GPU")
            print(f"  Consider:")
            print(f"    - Using CPU: --device cpu")
            print(f"    - Using smaller model: --model hf-internal-testing/tiny-stable-diffusion-xl-pipe")
            print(f"    - Using gradient checkpointing (if implemented)")
    
    # Compare with FP32
    print("\n" + "=" * 70)
    print("For comparison: FP32 would require:")
    print("=" * 70)
    est_fp32 = estimate_memory_requirements(resolution=256, batch_size=1, use_fp16=False)
    print(f"Total Estimated: {est_fp32['total_estimated']:.2f} GB")
    print(f"FP16 saves: {est_fp32['total_estimated'] - est_fp16['total_estimated']:.2f} GB ({((est_fp32['total_estimated'] - est_fp16['total_estimated']) / est_fp32['total_estimated'] * 100):.1f}%)")

