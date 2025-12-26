# learn_bayesian.py
"""
Main script for Bayesian parameter learning
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Parameter Learning for Objectness Cues'
    )
    parser.add_argument('--dataset', required=True, help='Path to VOC dataset')
    parser.add_argument('--output', default='bayesian_parameters', help='Output directory')
    parser.add_argument('--n-images', type=int, default=8, help='Number of training images')
    parser.add_argument('--n-windows', type=int, default=1000, help='Windows per image')
    parser.add_argument('--test', action='store_true', help='Run test with synthetic data')
    
    args = parser.parse_args()
    
    if args.test:
        from modules.bayesian_learning import test_bayesian_learning
        test_bayesian_learning()
        return
    
    # Check dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist")
        print("Use --test flag for synthetic data")
        return
    
    # Import and run
    try:
        from modules.bayesian_learning import BayesianParameterLearner
        
        print("="*70)
        print("BAYESIAN PARAMETER LEARNING - GROUP A")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Training images: {args.n_images}")
        print(f"Windows per image: {args.n_windows}")
        print("="*70)
        
        learner = BayesianParameterLearner(args.dataset, output_dir=args.output)
        results = learner.run_bayesian_learning(
            n_images=args.n_images,
            n_windows=args.n_windows
        )
        
        print("\n" + "="*70)
        print("LEARNING COMPLETE!")
        print("="*70)
        print(f"Output saved in: {args.output}/")
        print("\nGenerated files:")
        print(f"  {args.output}/bayesian_parameters.json")
        print(f"  {args.output}/config_bayesian.json")
        print(f"  {args.output}/bayesian_summary.txt")
        print(f"  {args.output}/ed_bayesian_learning.png")
        print(f"  {args.output}/ss_bayesian_learning.png")
        print(f"  {args.output}/ed_likelihoods_best.png")
        print(f"  {args.output}/ss_likelihoods_best.png")
        print("="*70)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all modules are in place")
        sys.exit(1)

if __name__ == "__main__":
    main()