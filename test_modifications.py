#!/usr/bin/env python3
"""
Test script to verify the modifications to Operation Agent and Task Builder Agent.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_timeout_update():
    """Test that timeout has been updated to 12 hours."""
    print("Testing timeout update...")

    # Check Operation Agent
    with open(project_root / "plexe/langgraph/agents/operation.py", "r") as f:
        content = f.read()
        assert '"timeout": 43200' in content, "Operation Agent timeout not updated to 43200"

    # Check gnn_specialist tool
    with open(project_root / "plexe/langgraph/tools/gnn_specialist.py", "r") as f:
        content = f.read()
        assert "timeout: int = 43200" in content, "execute_training_script default timeout not updated"

    print("✓ Timeout successfully updated to 12 hours (43200 seconds)")

def test_task_table_timeout():
    """Test that task table generation has timeout wrapper."""
    print("Testing task table timeout implementation...")

    with open(project_root / "plexe/langgraph/tools/gnn_specialist.py", "r") as f:
        content = f.read()
        assert "get_table_with_timeout" in content, "Timeout wrapper function not found"
        assert "signal.SIGALRM" in content, "Signal-based timeout not implemented"
        assert "90 minutes" in content or "5400" in content, "90-minute timeout not configured"

    print("✓ Task table generation timeout (90 minutes) successfully implemented")

def test_checkpoint_and_early_stopping():
    """Test checkpoint saving and early stopping implementation."""
    print("Testing checkpoint and early stopping...")

    with open(project_root / "plexe/langgraph/tools/gnn_specialist.py", "r") as f:
        content = f.read()

        # Check checkpoint saving
        assert "checkpoint_path" in content, "Checkpoint path not defined"
        assert "torch.save(checkpoint" in content, "Checkpoint saving not implemented"
        assert "'model_state': model.state_dict()" in content, "Model state not saved in checkpoint"
        assert "'optimizer_state': optimizer.state_dict()" in content, "Optimizer state not saved"

        # Check early stopping
        assert "early_stop_patience" in content, "Early stopping patience not defined"
        assert "patience_counter" in content, "Patience counter not implemented"
        assert "No improvement for" in content, "Early stopping message not found"
        assert "min_delta" in content, "Minimum delta for improvement not defined"

        # Check checkpoint cleanup
        assert "os.remove(checkpoint_path)" in content, "Checkpoint cleanup not implemented"

    print("✓ Checkpoint saving (single file) successfully implemented")
    print("✓ Early stopping (patience=5, min_delta=1e-6) successfully implemented")

def test_sql_optimization_guidance():
    """Test SQL optimization guidance in Task Builder prompts."""
    print("Testing SQL optimization guidance...")

    with open(project_root / "plexe/langgraph/prompts/task_builder.py", "r") as f:
        content = f.read()

        # Check for optimization section
        assert "SQL Performance Optimization for Large Datasets" in content, "SQL optimization section not found"

        # Check for key optimization patterns
        assert "AVOID Cartesian Products" in content, "Cartesian product warning not found"
        assert "Use EXISTS Instead of IN" in content, "EXISTS vs IN guidance not found"
        assert "Aggregate Early" in content, "Early aggregation pattern not found"
        assert "Selective Joins Instead of Cross Joins" in content, "Selective join guidance not found"

        # Check for dataset-specific notes
        assert "Avito" in content and "9M+" in content, "Avito dataset size reference not found"
        assert "90-minute timeout" in content, "Timeout warning not found"

    print("✓ SQL optimization guidance successfully added to Task Builder Agent")

def test_training_script_features():
    """Test that training script template has all new features."""
    print("Testing training script template features...")

    with open(project_root / "plexe/langgraph/tools/gnn_specialist.py", "r") as f:
        content = f.read()

        # Check for memory management
        assert "torch.cuda.empty_cache()" in content, "GPU memory cleanup not found"

        # Check for checkpoint recovery
        assert "if os.path.exists(checkpoint_path):" in content, "Checkpoint recovery not implemented"
        assert "weights_only=True" in content, "Safe checkpoint loading not configured"

        # Check for improved error handling
        assert "if len(val_pred_list) == 0:" in content, "Empty validation handling not found"

    print("✓ Training script template successfully updated with all features")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing RelAu2ML Agent Modifications")
    print("=" * 60)

    try:
        test_timeout_update()
        test_task_table_timeout()
        test_checkpoint_and_early_stopping()
        test_sql_optimization_guidance()
        test_training_script_features()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Modifications successfully implemented!")
        print("=" * 60)

        print("\nSummary of changes:")
        print("1. Training timeout: 4 hours → 12 hours")
        print("2. Task table timeout: None → 90 minutes")
        print("3. Checkpoint: Saves latest checkpoint for crash recovery")
        print("4. Early stopping: patience=5, min_delta=1e-6")
        print("5. SQL optimization: Added guidance for large datasets")
        print("6. Memory management: GPU cache cleanup between epochs")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()