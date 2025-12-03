import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


def quantize_model(
    model: nn.Module,
    quantization_type: torch.dtype = torch.qint8,
    modules_to_quantize: Optional[set] = None
) -> nn.Module:
    if modules_to_quantize is None:
        modules_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}
    
    model.eval()
    
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec=modules_to_quantize,
        dtype=quantization_type
    )
    
    return quantized_model


def save_model(model: nn.Module, filepath: str) -> float:
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / 1e6
    return size_mb


def evaluate_model_quality(
    model: nn.Module,
    tokenizer,
    validation_dataset,
    max_samples: int = 1000,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    seed: int = 42
) -> Dict:
    """
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        validation_dataset: Dataset with 'text' and 'label' fields
        max_samples: Maximum number of samples to evaluate
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the dataset before evaluation
        seed: Random seed for shuffling
    
    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Shuffle the dataset if requested
    if shuffle:
        validation_dataset = validation_dataset.shuffle(seed=seed)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    eval_samples = min(max_samples, len(validation_dataset))
    
    with torch.no_grad():
        for i in tqdm(range(0, eval_samples, batch_size), desc="Evaluating"):
            batch_end = min(i + batch_size, eval_samples)
            batch = validation_dataset[i:batch_end]
            
            inputs = tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch['label']).to(device)
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    
    # Calculate additional metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    print(np.unique(all_labels))
    
    # For binary classification
    if len(np.unique(all_labels)) == 2:
        true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
        false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
        false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1_score = None
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }
    
    if precision is not None:
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
    
    return metrics


def measure_inference_time(
    model: nn.Module,
    tokenizer,
    test_texts: List[str],
    num_runs: int = 100
) -> Dict:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            inputs = tokenizer(test_texts[0], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
        
        # Measurement
        for _ in tqdm(range(num_runs), desc="Measuring inference time"):
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                start_time = time.time()
                _ = model(**inputs)
                end_time = time.time()
                
                times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_runs': len(times)
    }


def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    tokenizer,
    validation_dataset,
    test_texts: List[str],
    max_samples: int = 1000
) -> Dict[str, Dict]:
    """
    Compare original and quantized models.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized model
        tokenizer: Tokenizer
        validation_dataset: Validation dataset
        test_texts: Test texts for inference timing
        max_samples: Maximum samples for evaluation
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*60)
    print("Evaluating Original Model...")
    print("="*60)
    original_metrics = evaluate_model_quality(
        original_model, tokenizer, validation_dataset, max_samples
    )
    
    print("\n" + "="*60)
    print("Evaluating Quantized Model...")
    print("="*60)
    quantized_metrics = evaluate_model_quality(
        quantized_model, tokenizer, validation_dataset, max_samples
    )
    
    print("\n" + "="*60)
    print("Measuring Inference Time...")
    print("="*60)
    original_time = measure_inference_time(original_model, tokenizer, test_texts)
    quantized_time = measure_inference_time(quantized_model, tokenizer, test_texts)
    
    # Save models and get sizes
    original_size = save_model(original_model, "model_fp32.pt")
    quantized_size = save_model(quantized_model, "model_quantized.pt")
    
    comparison = {
        'original': {
            'metrics': original_metrics,
            'inference_time': original_time,
            'model_size_mb': original_size
        },
        'quantized': {
            'metrics': quantized_metrics,
            'inference_time': quantized_time,
            'model_size_mb': quantized_size
        },
        'improvements': {
            'size_reduction': (1 - quantized_size / original_size) * 100,
            'speedup': original_time['mean_time'] / quantized_time['mean_time'],
            'accuracy_drop': (original_metrics['accuracy'] - quantized_metrics['accuracy']) * 100
        }
    }
    
    return comparison


def print_comparison_results(comparison: Dict):
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\n📊 Model Size:")
    print(f"  Original:  {comparison['original']['model_size_mb']:.2f} MB")
    print(f"  Quantized: {comparison['quantized']['model_size_mb']:.2f} MB")
    print(f"  Reduction: {comparison['improvements']['size_reduction']:.2f}%")
    
    print("\n🎯 Accuracy:")
    print(f"  Original:  {comparison['original']['metrics']['accuracy']:.4f}")
    print(f"  Quantized: {comparison['quantized']['metrics']['accuracy']:.4f}")
    print(f"  Drop:      {comparison['improvements']['accuracy_drop']:.2f}%")
    
    if 'f1_score' in comparison['original']['metrics']:
        print("\n📈 F1 Score:")
        print(f"  Original:  {comparison['original']['metrics']['f1_score']:.4f}")
        print(f"  Quantized: {comparison['quantized']['metrics']['f1_score']:.4f}")
    
    print("\n⚡ Inference Time:")
    print(f"  Original:  {comparison['original']['inference_time']['mean_time']*1000:.2f} ms")
    print(f"  Quantized: {comparison['quantized']['inference_time']['mean_time']*1000:.2f} ms")
    print(f"  Speedup:   {comparison['improvements']['speedup']:.2f}x")
    
    print("\n" + "="*60)
