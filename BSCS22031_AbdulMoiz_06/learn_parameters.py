# learn_parameters.py
"""
Main script to run parameter learning
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Learn parameters for Objectness cues')
    parser.add_argument('--dataset', required=True, help='Path to VOC dataset')
    parser.add_argument('--output', default='learned_parameters', help='Output directory')
    parser.add_argument('--n-images', type=int, default=10, help='Number of training images')
    parser.add_argument('--test', action='store_true', help='Run test with synthetic data')
    
    args = parser.parse_args()
    
    if args.test:
        # Run test with synthetic data
        from modules.parameter_learning import test_parameter_learning
        test_parameter_learning()
        return
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist")
        print("Please provide path to VOC dataset (should contain JPEGImages/ and Annotations/)")
        print("Or use --test flag to run with synthetic data")
        return
    
    # Import and run learning
    try:
        from modules.parameter_learning import ParameterLearner
        
        print("="*70)
        print("STARTING PARAMETER LEARNING")
        print("="*70)
        
        learner = ParameterLearner(args.dataset, output_dir=args.output)
        results = learner.run_complete_learning(n_images=args.n_images)
        
        print("\n" + "="*70)
        print("LEARNING COMPLETE!")
        print("="*70)
        print(f"Output directory: {args.output}")
        print(f"Learned parameters saved in:")
        print(f"  - {args.output}/learned_parameters.json")
        print(f"  - {args.output}/config_group_a.json")
        print(f"  - {args.output}/learning_summary.txt")
        print(f"\nVisualizations saved as PNG files in {args.output}/")
        print("="*70)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are in place")
        sys.exit(1)

if __name__ == "__main__":
    main()