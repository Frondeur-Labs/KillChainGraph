#!/usr/bin/env python3
"""
Real Ensemble Evaluation for Kill Chain Prediction
Evaluates multiple ensemble strategies against single-model baseline
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from load_models import load_phase_models, load_ensemble_phase_models
from prediction import run_kill_chain_prediction, run_ensemble_kill_chain_prediction
from ensemble import (create_voting_ensemble, create_weighted_ensemble,
                      create_stacking_ensemble, evaluate_ensemble_performance)


def create_test_dataset() -> List[Tuple[str, Dict[str, str]]]:
    """Create a comprehensive test dataset with attack descriptions and known techniques.

    Returns realistic attack scenarios based on MITRE ATT&CK mappings.
    """
    test_cases = [
        # Case 1: Phishing with macro delivery
        ("Phishing email containing malicious Office macro that downloads and executes PowerShell payload",
         {"recon": "T1595", "weapon": "T1221", "delivery": "T1071", "exploit": "T1059", "install": "T1070", "c2": "T1071", "objectives": "T1001"}),

        # Case 2: RDP brute force attack
        ("Brute force attack against RDP service followed by ransomware deployment via PsExec",
         {"recon": "T1590", "weapon": "T1210", "delivery": "T1071", "exploit": "T1059", "install": "T1569", "c2": "T1021", "objectives": "T1486"}),

        # Case 3: Web application attack
        ("SQL injection vulnerability exploited in web application to gain database access and establish command shell",
         {"recon": "T1596", "weapon": "T1190", "delivery": "T1071", "exploit": "T1505", "install": "T1055", "c2": "T1059", "objectives": "T1003"}),

        # Case 4: USB-based attack
        ("Malicious USB device left in parking lot with autorun script that installs keylogger and beacon malware",
         {"recon": "T1583", "weapon": "T1091", "delivery": "T1092", "exploit": "T1055", "install": "T1547", "c2": "T1001", "objectives": "T1056"}),

        # Case 5: Supply chain compromise
        ("Compromised third-party software update installs backdoor with persistent access and data exfiltration capabilities",
         {"recon": "T1595", "weapon": "T1195", "delivery": "T1052", "exploit": "T1068", "install": "T1505", "c2": "T1573", "objectives": "T1005"}),
    ]

    return test_cases


def run_comprehensive_evaluation(model_root: str = "../transformer_model_killchain", output_file: str = "ensemble_results.json"):
    """
    Run comprehensive evaluation of ensemble methods vs single model baseline.

    Args:
        model_root: Path to trained model directory
        output_file: File to save results
    """
    print("🚀 Starting Comprehensive Ensemble Evaluation")
    print("=" * 50)

    try:
        # Load models
        print("📥 Loading models...")
        single_models = load_phase_models(model_root)
        ensemble_models = load_ensemble_phase_models(model_root, models_per_phase=2)

        if not ensemble_models:
            print("❌ Failed to load ensemble models")
            return

        # Create test dataset
        test_data = create_test_dataset()
        print(f"📊 Created test dataset with {len(test_data)} attack scenarios")

        # Test different ensemble strategies
        strategies = {
            "single_model": None,
            "voting_soft": create_voting_ensemble("soft"),
            "voting_hard": create_voting_ensemble("hard"),
            "weighted_avg": create_weighted_ensemble(),
            "stacking": create_stacking_ensemble()
        }

        results = {}

        print("\n🔬 Running evaluations...")

        for strategy_name, ensemble_predictor in strategies.items():
            print(f"  Testing {strategy_name}...")

            if strategy_name == "single_model":
                # Use single models for comparison
                from prediction import run_kill_chain_prediction
                from sentence_transformers import SentenceTransformer
                sentence_model = SentenceTransformer("basel/ATTACK-BERT")
                single_accuracy = {phase: [] for phase in single_models.keys()}

                for attack_text, true_techniques in test_data:
                    preds = run_kill_chain_prediction(attack_text, single_models, sentence_model, k=10)
                    for phase, true_tech in true_techniques.items():
                        if phase in preds:
                            techs = [tech for tech, _ in preds[phase][:3]]
                            single_accuracy[phase].append(1.0 if true_tech in techs else 0.0)

                # Calculate averages
                metrics = {}
                for phase in single_models.keys():
                    acc = np.mean(single_accuracy[phase]) if single_accuracy[phase] else 0.0
                    metrics[phase] = {'single_model_accuracy': acc, 'ensemble_accuracy': acc, 'improvement': 0.0}

                overall_acc = np.mean([m['single_model_accuracy'] for m in metrics.values()])
                metrics['overall'] = {
                    'single_model_accuracy': overall_acc,
                    'ensemble_accuracy': overall_acc,
                    'improvement': 0.0
                }
            else:
                # For ensemble strategies, use only ensemble models for both
                metrics = evaluate_ensemble_performance(
                    test_data, ensemble_models, ensemble_models, ensemble_predictor
                )

            results[strategy_name] = metrics

        # Calculate ensemble improvements
        baseline_accuracy = results["single_model"]["overall"]["ensemble_accuracy"]  # This is actually single model accuracy when called with single models

        for strategy_name in ["voting_soft", "voting_hard", "weighted_avg", "stacking"]:
            if strategy_name in results:
                ensemble_acc = results[strategy_name]["overall"]["ensemble_accuracy"]
                improvement = ensemble_acc - baseline_accuracy
                results[strategy_name]["overall"]["improvement_over_single"] = improvement

        # Save results - already imported at top
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)

        print(f"\n✅ Results saved to {output_file}")

        # Pretty print summary
        print("\n📈 ENSEMBLE EVALUATION SUMMARY")
        print("=" * 50)

        baseline = results["single_model"]["overall"]
        print(".3f")

        for strategy in ["voting_soft", "voting_hard", "weighted_avg", "stacking"]:
            if strategy in results:
                data = results[strategy]["overall"]
                improvement = data.get("improvement_over_single", 0)
                color = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                print(f"{strategy.replace('_', ' ').title()}: {color} Accuracy={data['ensemble_accuracy']:.3f} (Improvement:{improvement:+.3f})")

        # Phase-by-phase improvement analysis
        print("\n📊 PHASE-BY-PHASE IMPROVEMENT ANALYSIS")
        print("-" * 50)

        best_phase_improvements = {}
        for phase in single_models.keys():
            baseline_acc = results["single_model"][phase]["single_model_accuracy"]
            best_improvement = 0
            best_strategy = "single_model"

            for strategy in ["voting_soft", "voting_hard", "weighted_avg", "stacking"]:
                if strategy in results:
                    ensemble_acc = results[strategy][phase]["ensemble_accuracy"]
                    improvement = ensemble_acc - baseline_acc
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_strategy = strategy

            best_phase_improvements[phase] = (best_improvement, best_strategy)

            color = "🟢" if best_improvement > 0 else "🔴" if best_improvement < 0 else "🟡"
            print(f"{phase.capitalize()}: {color} Best +{best_improvement:.3f} ({best_strategy})")

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_ensemble_diversity(ensemble_models: Dict[str, List], test_sample: str):
    """
    Analyze model diversity by comparing predictions across ensemble members.

    Args:
        ensemble_models: {phase: [(model1, le1), (model2, le2), ...]}
        test_sample: Sample attack description
    """
    from sentence_transformers import SentenceTransformer
    from prediction import predict_topk_for_phase

    sentence_model = SentenceTransformer("basel/ATTACK-BERT")
    text_vec = sentence_model.encode([test_sample], show_progress_bar=False)[0]

    print(f"\n🔍 Ensemble Diversity Analysis for: '{test_sample[:50]}...'")
    print("-" * 60)

    diversity_stats = {}

    for phase, models_list in ensemble_models.items():
        predictions_per_model = []
        for i, (model, le) in enumerate(models_list):
            preds = predict_topk_for_phase(phase, model, le, text_vec, k=5)
            predictions_per_model.append(preds)

        # Calculate diversity metrics
        top1_techniques = [preds[0][0] for preds in predictions_per_model]
        unique_top1 = len(set(top1_techniques))

        # Agreement score (fraction of models agreeing on top prediction)
        most_common_top1 = max(set(top1_techniques), key=top1_techniques.count)
        agreement = top1_techniques.count(most_common_top1) / len(top1_techniques)

        diversity_stats[phase] = {
            "unique_top1": unique_top1,
            "agreement_score": agreement,
            "top1_techniques": top1_techniques
        }

        print(f"{phase.capitalize()}: Top-1 unique={unique_top1}/{len(models_list)}, "
              f"Agreement={agreement:.2f}, Techniques={top1_techniques}")

    return diversity_stats


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()

    if results:
        # Run diversity analysis on a sample
        try:
            ensemble_models = load_ensemble_phase_models("../transformer_model_killchain", models_per_phase=2)
            if ensemble_models:
                diversity = analyze_ensemble_diversity(
                    ensemble_models,
                    "Advanced persistent threat using spear phishing and custom malware for intellectual property theft"
                )
        except Exception as e:
            print(f"Diversity analysis failed: {e}")
