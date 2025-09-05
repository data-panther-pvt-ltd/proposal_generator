"""
Validation utilities for proposal content
"""

from typing import Dict, List, Any
import re
import logging

logger = logging.getLogger(__name__)

class ProposalValidator:
    """Validate proposal content and quality"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_config = config.get('quality', {})
        
    def validate_section(self, section_name: str, content: Dict) -> Dict:
        """
        Validate a section's content
        
        Returns:
            Dictionary with validation results
        """
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'score': 0,
            'feedback': ''
        }
        
        # Get section requirements
        section_config = self.config['section_routing'].get(section_name, {})
        min_words = section_config.get('min_words', 100)
        max_words = section_config.get('max_words', 10000)
        
        # Extract text content
        text_content = self._extract_text(content)
        word_count = len(text_content.split())
        
        # Check word count
        if word_count < min_words:
            validation_result['errors'].append(
                f"Section too short: {word_count} words (minimum: {min_words})"
            )
            validation_result['is_valid'] = False
        elif word_count > max_words:
            validation_result['warnings'].append(
                f"Section too long: {word_count} words (maximum: {max_words})"
            )
        
        # Check for required elements
        if section_name == "Budget" and "total_cost" not in str(content):
            validation_result['errors'].append("Budget section missing total cost")
            validation_result['is_valid'] = False
        
        if section_name == "Project Plan and Timelines" and "phases" not in str(content):
            validation_result['errors'].append("Timeline section missing project phases")
            validation_result['is_valid'] = False
        
        # Calculate quality score
        score = self._calculate_quality_score(text_content, section_name)
        validation_result['score'] = score
        
        # Check minimum quality threshold
        min_score = self.quality_config.get('min_section_score', 7.0)
        if score < min_score:
            validation_result['warnings'].append(
                f"Quality score below threshold: {score:.1f} (minimum: {min_score})"
            )
            validation_result['feedback'] = self._generate_improvement_feedback(
                score, text_content, section_name
            )
        
        return validation_result
    
    def _extract_text(self, content: Any) -> str:
        """Extract text from various content formats"""
        
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            if 'content' in content:
                return self._extract_text(content['content'])
            else:
                # Concatenate all text values
                text_parts = []
                for value in content.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, list):
                        text_parts.extend([str(item) for item in value])
                return ' '.join(text_parts)
        else:
            return str(content)
    
    def _calculate_quality_score(self, text: str, section_name: str) -> float:
        """Calculate quality score for content - improved algorithm"""
        
        scores = []
        
        # Enhanced relevance score (check for section-specific keywords)
        relevance_keywords = {
            "Executive Summary": ["strategic", "solution", "vision", "objective", "deliver", "comprehensive", "innovation"],
            "Problem or Need Statement": ["challenge", "issue", "problem", "need", "opportunity", "requirement", "solution"],
            "Project Scope": ["scope", "objective", "deliverable", "requirement", "timeline", "resource", "boundary"],
            "Proposed Solution": ["solution", "approach", "methodology", "architecture", "design", "implement", "technology"],
            "List of Deliverables": ["deliverable", "output", "product", "component", "feature", "documentation", "training"],
            "Technical Approach and Methodology": ["architecture", "technology", "methodology", "implementation", "design", "framework", "approach"],
            "Project Plan and Timelines": ["timeline", "phase", "milestone", "deliverable", "schedule", "planning", "duration"],
            "Budget": ["cost", "price", "investment", "ROI", "budget", "financial", "resource"],
            "Risk Analysis and Mitigation": ["risk", "mitigation", "contingency", "challenge", "threat", "vulnerability", "assessment"],
            "Our Team/Company Profile": ["team", "experience", "expertise", "capability", "qualification", "professional", "company"],
            "Success Stories/Case Studies": ["success", "case", "study", "example", "experience", "achievement", "result"],
            "Implementation Strategy": ["implementation", "strategy", "deployment", "approach", "execution", "plan", "methodology"],
            "Support and Maintenance": ["support", "maintenance", "service", "assistance", "ongoing", "continuous", "help"],
            "Terms and Conditions": ["terms", "condition", "agreement", "contract", "policy", "compliance", "legal"],
            "Conclusion": ["conclusion", "summary", "commitment", "partnership", "future", "success", "forward"]
        }
        
        keywords = relevance_keywords.get(section_name, ["deliver", "implement", "solution", "strategic"])
        keyword_count = sum(1 for word in keywords if word.lower() in text.lower())
        # More generous relevance scoring - if content has good keywords, score well
        relevance_score = min(10, max(7, keyword_count * 1.2 + 3))
        scores.append(relevance_score)
        
        # Enhanced completeness score (based on length and content depth)
        word_count = len(text.split())
        if word_count >= 300:
            completeness_score = 10
        elif word_count >= 200:
            completeness_score = 9
        elif word_count >= 150:
            completeness_score = 8
        elif word_count >= 100:
            completeness_score = 7
        else:
            completeness_score = max(5, word_count / 20)
        scores.append(completeness_score)
        
        # Enhanced clarity score (based on sentence structure and formatting)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 0:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Reward well-structured content
            if 8 <= avg_sentence_length <= 30:
                clarity_score = 9
            elif 5 <= avg_sentence_length <= 35:
                clarity_score = 8
            else:
                clarity_score = 7
        else:
            clarity_score = 6
            
        # Bonus for formatting (headers, bullets, structure)
        if any(indicator in text for indicator in ['#', '##', '###', '- ', '* ', '1.', '2.']):
            clarity_score = min(10, clarity_score + 1)
        scores.append(clarity_score)
        
        # Enhanced professional language score
        professional_terms = [
            "deliver", "implement", "optimize", "enhance", "strategic", "comprehensive", 
            "innovative", "robust", "scalable", "efficient", "expertise", "experience",
            "solution", "methodology", "approach", "framework", "architecture", "quality"
        ]
        prof_count = sum(1 for term in professional_terms if term in text.lower())
        # More generous professional scoring
        prof_score = min(10, max(7, prof_count * 0.8 + 4))
        scores.append(prof_score)
        
        # Bonus for comprehensive content (mentions of specific technologies, processes, etc.)
        technical_indicators = [
            "AI", "machine learning", "API", "database", "cloud", "security", "integration",
            "agile", "scrum", "testing", "deployment", "monitoring", "analytics"
        ]
        tech_count = sum(1 for term in technical_indicators if term.lower() in text.lower())
        if tech_count > 0:
            tech_bonus = min(1, tech_count * 0.2)
            for i in range(len(scores)):
                scores[i] = min(10, scores[i] + tech_bonus)
        
        # Calculate weighted average with slight bias toward completeness and relevance
        weighted_score = (
            relevance_score * 0.3 +
            completeness_score * 0.3 +
            clarity_score * 0.2 +
            prof_score * 0.2
        )
        
        return min(10, weighted_score)
    
    def _generate_improvement_feedback(
        self, 
        score: float, 
        text: str, 
        section_name: str
    ) -> str:
        """Generate specific feedback for improvement"""
        
        feedback_parts = []
        
        if score < 5:
            feedback_parts.append("Content needs significant improvement")
        elif score < 7:
            feedback_parts.append("Content is adequate but could be enhanced")
        
        # Specific suggestions
        if len(text.split()) < 200:
            feedback_parts.append("Add more detail and explanation")
        
        if section_name == "Budget" and "breakdown" not in text.lower():
            feedback_parts.append("Include detailed cost breakdown")
        
        if section_name == "Technical Approach" and "architecture" not in text.lower():
            feedback_parts.append("Add technical architecture details")
        
        return ". ".join(feedback_parts)
    
    def calculate_overall_score(self, sections: Dict) -> float:
        """Calculate overall proposal quality score"""
        
        if not sections:
            return 0.0
        
        total_score = 0
        section_count = 0
        
        for section_name, content in sections.items():
            text = self._extract_text(content)
            score = self._calculate_quality_score(text, section_name)
            total_score += score
            section_count += 1
        
        return round(total_score / max(section_count, 1), 1)
    
    def validate_proposal(self, proposal: Dict) -> Dict:
        """Validate entire proposal"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'overall_score': 0,
            'section_scores': {}
        }
        
        # Validate each section
        for section_name, content in proposal.get('sections', {}).items():
            section_validation = self.validate_section(section_name, content)
            validation_result['section_scores'][section_name] = section_validation['score']
            
            if not section_validation['is_valid']:
                validation_result['errors'].extend(section_validation['errors'])
                validation_result['is_valid'] = False
            
            validation_result['warnings'].extend(section_validation.get('warnings', []))
        
        # Calculate overall score
        validation_result['overall_score'] = self.calculate_overall_score(
            proposal.get('sections', {})
        )
        
        # Check overall quality threshold
        min_overall = self.quality_config.get('min_overall_score', 8.0)
        if validation_result['overall_score'] < min_overall:
            validation_result['warnings'].append(
                f"Overall quality below threshold: {validation_result['overall_score']:.1f}"
            )
        
        return validation_result