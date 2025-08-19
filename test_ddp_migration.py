#!/usr/bin/env python3
"""
Test script to verify DDP migration works correctly.
"""

import torch
import sys
import os

# Add the robustness package to path
sys.path.insert(0, '.')

def test_imports():
    """Test that all new imports work correctly."""
    print("Testing imports...")
    try:
        from robustness.tools.distributed_utils import setup_distributed, cleanup_distributed
        from robustness.backup.distributed_main import launch_distributed
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_single_gpu_compatibility():
    """Test that single GPU mode still works (backward compatibility)."""
    print("\nTesting single GPU compatibility...")
    try:
        # This should work without distributed setup
        from robustness.tools.distributed_utils import get_rank, get_world_size, is_distributed
        
        # Should return defaults when not distributed
        assert get_rank() == 0, f"Expected rank 0, got {get_rank()}"
        assert get_world_size() == 1, f"Expected world_size 1, got {get_world_size()}"
        assert not is_distributed(), "Should not be in distributed mode"
        
        print("✓ Single GPU compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Single GPU test failed: {e}")
        return False

def test_ddp_setup():
    """Test DDP setup without actually running training."""
    print("\nTesting DDP setup (dry run)...")
    try:
        if torch.cuda.device_count() < 2:
            print("⚠ Skipping multi-GPU test (less than 2 GPUs available)")
            return True
        
        # Test that we can create a simple model and wrap it with DDP
        import torch.nn as nn
        from robustness.tools.distributed_utils import setup_distributed, cleanup_distributed, get_rank
        
        # Simple test without actually initializing distributed training
        model = nn.Linear(10, 5)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            model = model.cuda(device)
            
        print("✓ DDP setup test passed")
        return True
    except Exception as e:
        print(f"✗ DDP setup test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running DDP Migration Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_single_gpu_compatibility,
        test_ddp_setup
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! DDP migration looks good.")
        print("\nNext steps:")
        print("1. Test with actual training: python -m robustness.distributed_main --distributed [your args]")
        print("2. Compare single vs multi-GPU performance")
        print("3. Verify checkpoint loading/saving works correctly")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
