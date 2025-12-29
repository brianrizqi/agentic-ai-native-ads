"""
Web Agent Module
Handles user input and initial processing.
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class WebAgent:
    """
    Web Agent that receives and processes user input.
    Acts as the entry point for user interactions.
    """
    
    def __init__(self):
        """Initialize the Web Agent."""
        self.session_id = None
        logger.info("Web Agent initialized")
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and prepare it for the preprocessing stage.
        
        Args:
            user_input: Raw input from the user
            
        Returns:
            Dictionary containing processed input and metadata
        """
        logger.info("Web Agent processing user input")
        
        # Generate session ID if not exists
        if not self.session_id:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Basic input validation
        if not user_input or not isinstance(user_input, str):
            raise ValueError("Invalid user input")
        
        # Clean and normalize input
        cleaned_input = user_input.strip()
        
        # Prepare output
        output = {
            'raw_input': user_input,
            'cleaned_input': cleaned_input,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'input_length': len(cleaned_input),
            'metadata': {
                'source': 'web_interface',
                'language': 'auto-detect'  # Can be enhanced with language detection
            }
        }
        
        logger.info(f"Web Agent processed input (length: {output['input_length']})")
        return output
    
    def reset_session(self):
        """Reset the current session."""
        self.session_id = None
        logger.info("Session reset")
