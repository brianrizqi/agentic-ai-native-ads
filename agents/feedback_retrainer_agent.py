"""
Feedback and ReTrainer Agent Module
Collects feedback and manages model improvement.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackReTrainerAgent:
    """
    Feedback/ReTrainer Agent yang mengumpulkan feedback dan mengelola improvement.
    """
    
    def __init__(self, feedback_dir: str = 'data/feedback'):
        """
        Initialize Feedback/ReTrainer Agent.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_buffer = []
        logger.info("Feedback/ReTrainer Agent initialized")
    
    def collect_feedback(self, classification: Dict[str, Any],
                        explanation: Dict[str, Any],
                        preprocessed_data: Dict[str, Any],
                        user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect and analyze feedback.
        
        Args:
            classification: Classification result
            explanation: Explanation result
            preprocessed_data: Original preprocessed data
            user_feedback: Optional user feedback
            
        Returns:
            Feedback analysis
        """
        logger.info("Collecting feedback")
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'classification': {
                'label': classification.get('label'),
                'confidence': classification.get('confidence'),
                'reasoning': classification.get('reasoning')
            },
            'explanation': {
                'summary': explanation.get('summary'),
                'key_factors': explanation.get('key_factors')
            },
            'input_data': {
                'url': preprocessed_data.get('original_url'),
                'title': preprocessed_data.get('title'),
                'text_length': preprocessed_data.get('metadata', {}).get('cleaned_length')
            },
            'user_feedback': user_feedback or {},
            'quality_metrics': self._calculate_quality_metrics(classification, explanation)
        }
        
        # Add to buffer
        self.feedback_buffer.append(feedback_entry)
        
        # Save to file
        self._save_feedback(feedback_entry)
        
        # Analyze feedback
        analysis = self._analyze_feedback(feedback_entry)
        
        logger.info(f"Feedback collected (quality: {analysis.get('quality_score', 0):.2f})")
        return analysis
    
    def _calculate_quality_metrics(self, classification: Dict[str, Any],
                                   explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the classification."""
        confidence = classification.get('confidence', 0)
        
        # Quality based on confidence
        if confidence > 0.9:
            quality = 'excellent'
            score = 1.0
        elif confidence > 0.7:
            quality = 'good'
            score = 0.8
        elif confidence > 0.5:
            quality = 'fair'
            score = 0.6
        else:
            quality = 'poor'
            score = 0.4
        
        # Check explanation completeness
        has_summary = bool(explanation.get('summary'))
        has_factors = len(explanation.get('key_factors', [])) > 0
        explanation_score = (has_summary + has_factors) / 2
        
        return {
            'quality_level': quality,
            'quality_score': score,
            'confidence_score': confidence,
            'explanation_completeness': explanation_score,
            'overall_score': (score + explanation_score) / 2
        }
    
    def _analyze_feedback(self, feedback_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback entry."""
        metrics = feedback_entry.get('quality_metrics', {})
        user_feedback = feedback_entry.get('user_feedback', {})
        
        # Determine if retraining is needed
        needs_improvement = (
            metrics.get('overall_score', 1.0) < 0.7 or
            user_feedback.get('is_correct') == False
        )
        
        analysis = {
            'quality_score': metrics.get('overall_score', 0),
            'quality_level': metrics.get('quality_level', 'unknown'),
            'needs_improvement': needs_improvement,
            'user_satisfaction': user_feedback.get('satisfaction', 'unknown'),
            'recommendations': self._generate_recommendations(metrics, user_feedback)
        }
        
        return analysis
    
    def _generate_recommendations(self, metrics: Dict[str, Any],
                                  user_feedback: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics.get('confidence_score', 1.0) < 0.6:
            recommendations.append("Low confidence - consider adding more training data")
        
        if metrics.get('explanation_completeness', 1.0) < 0.5:
            recommendations.append("Improve explanation generation")
        
        if user_feedback.get('is_correct') == False:
            recommendations.append("Classification error - review and retrain model")
        
        if not recommendations:
            recommendations.append("Performance is satisfactory")
        
        return recommendations
    
    def _save_feedback(self, feedback_entry: Dict[str, Any]):
        """Save feedback to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.feedback_dir / f'feedback_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(feedback_entry, f, indent=2)
            logger.info(f"Feedback saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        if not self.feedback_buffer:
            return {'message': 'No feedback collected yet'}
        
        # Calculate statistics
        quality_scores = [
            fb.get('quality_metrics', {}).get('overall_score', 0)
            for fb in self.feedback_buffer
        ]
        
        needs_improvement_count = sum(
            1 for fb in self.feedback_buffer
            if self._analyze_feedback(fb).get('needs_improvement', False)
        )
        
        summary = {
            'total_feedback': len(self.feedback_buffer),
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'needs_improvement_count': needs_improvement_count,
            'improvement_rate': needs_improvement_count / len(self.feedback_buffer) if self.feedback_buffer else 0,
            'latest_feedback': self.feedback_buffer[-1] if self.feedback_buffer else None
        }
        
        return summary
    
    def export_training_data(self, output_file: str = 'data/training_data.json'):
        """
        Export feedback as training data.
        
        Args:
            output_file: Output file path
        """
        logger.info(f"Exporting training data to {output_file}")
        
        training_data = []
        for fb in self.feedback_buffer:
            # Only include high-quality feedback
            if fb.get('quality_metrics', {}).get('overall_score', 0) > 0.7:
                training_data.append({
                    'text': fb.get('input_data', {}).get('title', ''),
                    'label': fb.get('classification', {}).get('label'),
                    'confidence': fb.get('classification', {}).get('confidence')
                })
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"Exported {len(training_data)} training samples")
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
