"""
Ensemble Inference Wrapper
Provides unified interface for ensemble predictions with uncertainty quantification
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

def ensemble_segment_with_confidence(
    ensemble_model,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    use_tta: bool = True
) -> Dict:
    """
    Perform ensemble segmentation with uncertainty quantification
    
    Args:
        ensemble_model: EnsembleSegmentation instance
        input_tensor: Input MRI tensor (B, C, D, H, W)
        device: Device to run on
        use_tta: Use Test-Time Augmentation
        
    Returns:
        Dictionary with prediction, confidence, and uncertainty maps
    """
    if use_tta:
        # Test-Time Augmentation for robust predictions
        ensemble_pred = ensemble_model.predict_with_tta(input_tensor, device=device)
    else:
        # Regular ensemble prediction with uncertainty
        ensemble_pred, uncertainty = ensemble_model.predict_multi_model(input_tensor, device=device)
    
    # Get confidence metrics
    confidence_results = ensemble_model.predict_with_confidence(input_tensor, device=device)
    
    # Extract final prediction
    prediction = confidence_results['prediction'].cpu().numpy()
    
    # Confidence scores (max probability per voxel)
    confidence = confidence_results['confidence'].cpu().numpy()
    
    # Entropy (uncertainty measure)
    entropy = confidence_results['entropy'].cpu().numpy()
    
    # Variance across models
    variance = confidence_results['variance'].cpu().numpy()
    
    # Model agreement
    agreement = confidence_results['agreement'].cpu().numpy()
    
    return {
        'prediction': prediction,
        'probabilities': confidence_results['probabilities'].cpu().numpy(),
        'confidence': confidence,
        'entropy': entropy,
        'variance': variance,
        'agreement': agreement,
        'uncertainty_flags': {
            'high_uncertainty_voxels': int(np.sum(entropy > 1.0)),  # High entropy threshold
            'low_confidence_voxels': int(np.sum(confidence < 0.7)),  # Low confidence threshold
            'low_agreement_voxels': int(np.sum(agreement < 0.8))  # Low agreement threshold
        }
    }


def ensemble_classify_with_confidence(
    ensemble_model,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    num_mc_samples: int = 10
) -> Dict:
    """
    Perform ensemble classification with uncertainty quantification
    
    Args:
        ensemble_model: EnsembleClassification instance
        input_tensor: Input tensor (B, C, H, W)
        device: Device to run on
        num_mc_samples: Number of Monte Carlo dropout samples
        
    Returns:
        Dictionary with prediction, confidence, and uncertainty
    """
    # Soft voting ensemble prediction
    results = ensemble_model.predict(input_tensor, device=device)
    
    # Monte Carlo Dropout for epistemic uncertainty
    mc_results = ensemble_model.predict_with_uncertainty(
        input_tensor, 
        device=device, 
        num_samples=num_mc_samples
    )
    
    prediction_class = results['prediction'].item()
    confidence_score = results['confidence'].item()
    probabilities = results['probabilities'].cpu().numpy()[0]
    
    # Epistemic uncertainty
    epistemic_uncertainty = mc_results['epistemic_uncertainty'].item()
    
    # Classification quality flags
    high_confidence = confidence_score > 0.85
    low_uncertainty = epistemic_uncertainty < 0.1
    
    return {
        'prediction_class': prediction_class,
        'confidence': confidence_score,
        'probabilities': probabilities.tolist(),
        'epistemic_uncertainty': epistemic_uncertainty,
        'quality_flags': {
            'high_confidence': high_confidence,
            'low_uncertainty': low_uncertainty,
            'recommended_for_clinical_use': high_confidence and low_uncertainty,
            'requires_expert_review': not (high_confidence and low_uncertainty)
        },
        'uncertainty_details': {
            'mean_probabilities': mc_results['mean_probabilities'].cpu().numpy()[0].tolist(),
            'std_probabilities': mc_results['std_probabilities'].cpu().numpy()[0].tolist()
        }
    }


def format_uncertainty_summary(uncertainty_data: Dict) -> str:
    """Format uncertainty data into human-readable summary"""
    lines = []
    lines.append("Uncertainty Analysis:")
    
    if 'uncertainty_flags' in uncertainty_data:
        flags = uncertainty_data['uncertainty_flags']
        lines.append(f"  - High uncertainty voxels: {flags['high_uncertainty_voxels']}")
        lines.append(f"  - Low confidence voxels: {flags['low_confidence_voxels']}")
        lines.append(f"  - Low model agreement: {flags['low_agreement_voxels']}")
    
    if 'quality_flags' in uncertainty_data:
        flags = uncertainty_data['quality_flags']
        if flags['recommended_for_clinical_use']:
            lines.append("  ✅ High quality prediction - Suitable for clinical use")
        elif flags['requires_expert_review']:
            lines.append("  ⚠️ Moderate confidence - Expert review recommended")
    
    return "\n".join(lines)
