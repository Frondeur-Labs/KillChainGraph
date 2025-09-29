"""
Ensemble Methods for Kill Chain Prediction
Combines predictions from multiple models per phase using voting, stacking, or weighted averaging.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
import yaml
import json
from pathlib import Path


class EnsembleStrategy(ABC):
    """Abstract base class for ensemble strategies."""

    @abstractmethod
    def combine_predictions(self, predictions_list: List[List[Tuple[str, float]]], weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """Combine predictions from multiple models."""
        pass


class VotingEnsemble(EnsembleStrategy):
    """Hard or soft voting ensemble."""

    def __init__(self, voting_type: str = "soft"):
        """
        Args:
            voting_type: "hard" for majority voting, "soft" for weighted average
        """
        self.voting_type = voting_type

    def combine_predictions(self, predictions_list: List[List[Tuple[str, float]]], weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        if not predictions_list:
            return []

        if weights is None:
            weights = [1.0] * len(predictions_list)
        else:
            weights = np.array(weights) / np.sum(weights)  # normalize

        # Collect all unique techniques and their weighted scores
        technique_scores = {}

        for i, preds in enumerate(predictions_list):
            weight = weights[i]
            for tech, score in preds:
                if self.voting_type == "soft":
                    # Weighted average of probabilities
                    if tech in technique_scores:
                        technique_scores[tech] += score * weight
                    else:
                        technique_scores[tech] = score * weight
                else:  # hard voting
                    # Each model votes for its top technique
                    if tech in technique_scores:
                        technique_scores[tech] += weight
                    else:
                        technique_scores[tech] = weight

        # Sort by combined score and return top results
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_techniques


class WeightedAveragingEnsemble(EnsembleStrategy):
    """Weighted averaging of technique scores."""

    def combine_predictions(self, predictions_list: List[List[Tuple[str, float]]], weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        if not predictions_list:
            return []

        if weights is None:
            weights = [1.0] * len(predictions_list)
        else:
            weights = np.array(weights) / np.sum(weights)

        # Collect all techniques
        all_techniques = set()
        for preds in predictions_list:
            for tech, _ in preds:
                all_techniques.add(tech)

        # Weighted average for each technique
        technique_scores = {}
        for tech in all_techniques:
            total_weighted_score = 0.0
            total_weight = 0.0

            for i, preds in enumerate(predictions_list):
                weight = weights[i]
                # Find this technique in the predictions
                for pred_tech, score in preds:
                    if pred_tech == tech:
                        total_weighted_score += score * weight
                        total_weight += weight
                        break

            if total_weight > 0:
                technique_scores[tech] = total_weighted_score / total_weight

        # Sort by score
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_techniques


class StackingEnsemble(EnsembleStrategy):
    """Stacking ensemble using a meta-model."""

    def __init__(self, meta_model=None):
        """
        Args:
            meta_model: Meta-model for stacking (e.g., sklearn LinearRegression)
                        If None, falls back to weighted averaging
        """
        self.meta_model = meta_model

    def combine_predictions(self, predictions_list: List[List[Tuple[str, float]]], weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        if not predictions_list or len(predictions_list) < 2:
            return predictions_list[0] if predictions_list else []

        # For simplicity, implement basic stacking as weighted average
        # In a full implementation, this would train a meta-model
        if self.meta_model is None:
            fallback = WeightedAveragingEnsemble()
            return fallback.combine_predictions(predictions_list, weights)

        # TODO: Implement full stacking with meta-model training
        return WeightedAveragingEnsemble().combine_predictions(predictions_list, weights)


class EnsemblePredictor:
    """Ensemble predictor for kill chain phases."""

    def __init__(self, strategy: EnsembleStrategy, config_path: Optional[Union[str, Path]] = None):
        """
        Args:
            strategy: Ensemble strategy to use
            config_path: Path to ensemble configuration file (YAML/JSON)
        """
        self.strategy = strategy
        self.config = self._load_config(config_path) if config_path else {}

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load ensemble configuration."""
        path = Path(config_path)
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON")

    def predict_phase(self, phase_predictions: List[List[Tuple[str, float]]], phase_name: str = None) -> List[Tuple[str, float]]:
        """
        Predict techniques for a single phase using ensemble.

        Args:
            phase_predictions: List of predictions from different models [(tech, score), ...]
            phase_name: Optional phase name for config lookup

        Returns:
            Combined predictions [(tech, score), ...]
        """
        # Get weights from config if available
        weights = None
        if phase_name and 'phase_weights' in self.config:
            weights = self.config['phase_weights'].get(phase_name)

        return self.strategy.combine_predictions(phase_predictions, weights)

    def predict_kill_chain(self, kill_chain_predictions: Dict[str, List[List[Tuple[str, float]]]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict techniques for entire kill chain using ensemble.

        Args:
            kill_chain_predictions: {phase: [model1_preds, model2_preds, ...]}

        Returns:
            {phase: combined_predictions}
        """
        ensemble_predictions = {}
        for phase, predictions_list in kill_chain_predictions.items():
            ensemble_predictions[phase] = self.predict_phase(predictions_list, phase)
        return ensemble_predictions

    def get_uncertainty_estimate(self, phase_predictions: List[List[Tuple[str, float]]], phase_name: str = None) -> float:
        """
        Estimate uncertainty of ensemble predictions.

        Returns:
            Variance-based uncertainty score
        """
        if not phase_predictions or len(phase_predictions) < 2:
            return 1.0  # High uncertainty for single/few models

        all_techniques = set()
        for preds in phase_predictions:
            for tech, _ in preds:
                all_techniques.add(tech)

        # Calculate variance in scores across models for each technique
        technique_variances = []
        for tech in all_techniques:
            scores = []
            for preds in phase_predictions:
                for pred_tech, score in preds:
                    if pred_tech == tech:
                        scores.append(score)
                        break
                else:
                    scores.append(0.0)  # Not predicted

            if len(scores) > 1:
                variance = np.var(scores)
                technique_variances.append(variance)

        if not technique_variances:
            return 1.0

        return np.mean(technique_variances)  # Average variance as uncertainty

    def detect_outliers(self, phase_predictions: List[List[Tuple[str, float]]], threshold: float = 0.5) -> bool:
        """
        Detect if ensemble predictions are inconsistent (potential outliers).

        Returns:
            True if predictions vary significantly across models
        """
        uncertainty = self.get_uncertainty_estimate(phase_predictions)
        return uncertainty > threshold

    def update_weights(self, historical_predictions: Dict[str, List], true_labels: Dict[str, str], alpha: float = 0.1):
        """
        Update ensemble weights based on historical performance.

        Args:
            historical_predictions: {phase: [[pred1, pred2, ...], [pred1, pred2, ...]]}
            true_labels: {phase: correct_technique}
            alpha: Learning rate for weight updates
        """
        if 'phase_weights' not in self.config:
            self.config['phase_weights'] = {}

        # Initialize weights if not present
        num_models = None
        for phase, model_preds_list in historical_predictions.items():
            if num_models is None:
                num_models = len(model_preds_list)
            weights = self.config['phase_weights'].get(phase, [1.0] * num_models)

            if phase in true_labels:
                true_tech = true_labels[phase]

                # Calculate accuracy contribution for each model
                model_accuracies = []
                for model_preds in model_preds_list:
                    # Check if true technique is in top predictions
                    predicted_techs = [tech for tech, _ in model_preds]
                    accuracy = 1.0 if true_tech in predicted_techs[:3] else 0.0  # Top-3 accuracy
                    model_accuracies.append(accuracy)

                # Update weights based on accuracy
                for i, acc in enumerate(model_accuracies):
                    weights[i] = weights[i] * (1 + alpha * (acc - 0.5))  # Reward accuracy

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

            self.config['phase_weights'][phase] = weights


# Convenience functions
def create_voting_ensemble(voting_type: str = "soft") -> EnsemblePredictor:
    """Create voting ensemble predictor."""
    return EnsemblePredictor(VotingEnsemble(voting_type))


def create_weighted_ensemble() -> EnsemblePredictor:
    """Create weighted averaging ensemble predictor."""
    return EnsemblePredictor(WeightedAveragingEnsemble())


def create_stacking_ensemble(meta_model=None) -> EnsemblePredictor:
    """Create stacking ensemble predictor."""
    return EnsemblePredictor(StackingEnsemble(meta_model))


def evaluate_ensemble_performance(test_data: List[Tuple[str, Dict[str, str]]],
                                single_models: Dict[str, Tuple],
                                ensemble_models: Dict[str, List],
                                ensemble_predictor: EnsemblePredictor) -> Dict:
    """
    Evaluate ensemble vs single model performance.

    Args:
        test_data: [(attack_text, {phase: true_technique}), ...]
        single_models: {phase: (model, le)}
        ensemble_models: {phase: [(model1, le1), (model2, le2), ...]}
        ensemble_predictor: EnsemblePredictor instance

    Returns:
        Performance metrics dict
    """
    from prediction import run_kill_chain_prediction, run_ensemble_kill_chain_prediction
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer("basel/ATTACK-BERT")

    single_accuracy = {phase: [] for phase in single_models.keys()}
    ensemble_accuracy = {phase: [] for phase in ensemble_models.keys()}

    for attack_text, true_techniques in test_data:
        # Single model predictions
        single_preds = run_kill_chain_prediction(attack_text, single_models, sentence_model, k=10)

        # Ensemble predictions
        ensemble_preds = run_ensemble_kill_chain_prediction(
            attack_text, ensemble_models, sentence_model, ensemble_predictor, k=10
        )

        # Calculate accuracies
        for phase, true_tech in true_techniques.items():
            if phase in single_preds:
                single_techs = [tech for tech, _ in single_preds[phase][:3]]  # Top-3
                single_accuracy[phase].append(1.0 if true_tech in single_techs else 0.0)

            if phase in ensemble_preds:
                ensemble_techs = [tech for tech, _ in ensemble_preds[phase][:3]]
                ensemble_accuracy[phase].append(1.0 if true_tech in ensemble_techs else 0.0)

    # Calculate averages
    metrics = {}
    for phase in single_models.keys():
        single_acc = np.mean(single_accuracy[phase]) if single_accuracy[phase] else 0.0
        ensemble_acc = np.mean(ensemble_accuracy[phase]) if ensemble_accuracy[phase] else 0.0

        metrics[phase] = {
            'single_model_accuracy': single_acc,
            'ensemble_accuracy': ensemble_acc,
            'improvement': ensemble_acc - single_acc
        }

    overall_single = np.mean([m['single_model_accuracy'] for m in metrics.values()])
    overall_ensemble = np.mean([m['ensemble_accuracy'] for m in metrics.values()])

    metrics['overall'] = {
        'single_model_accuracy': overall_single,
        'ensemble_accuracy': overall_ensemble,
        'improvement': overall_ensemble - overall_single
    }

    return metrics


if __name__ == "__main__":
    # Example usage with real loading
    from load_models import load_phase_models, load_ensemble_phase_models
    from prediction import run_ensemble_kill_chain_prediction
    from sentence_transformers import SentenceTransformer

    # Mock predictions from different models
    model1_preds = [("T1059", 0.9), ("T1071", 0.7)]
    model2_preds = [("T1059", 0.8), ("T1210", 0.6)]
    model3_preds = [("T1071", 0.5), ("T1059", 0.8)]

    ensemble = create_voting_ensemble("soft")
    combined = ensemble.predict_phase([model1_preds, model2_preds, model3_preds])
    print("Mock Ensemble predictions:", combined)

    # Test with real models
    try:
        print("\nTesting real ensemble loading...")
        single_models = load_phase_models("../transformer_model_killchain")
        ensemble_models = load_ensemble_phase_models("../transformer_model_killchain", models_per_phase=2)

        if ensemble_models:
            sentence_model = SentenceTransformer("basel/ATTACK-BERT")
            ensemble_pred = run_ensemble_kill_chain_prediction(
                "Test phishing attack",
                ensemble_models,
                sentence_model,
                ensemble
            )
            print("Real ensemble test successful for phases:", list(ensemble_pred.keys())[:3])

            # Simple evaluation (mock test data)
            test_data = [
                ("phishing email with macro", {"recon": "T1595", "weapon": "T1221"}),
                ("ransomware attack via RDP", {"recon": "T1590", "weapon": "T1210"})
            ]
            # Note: This is mock evaluation since we don't have ground truth labels
            print("Mock evaluation complete")
        else:
            print("No ensemble models loaded")
    except Exception as e:
        print("Error in real ensemble test:", e)
