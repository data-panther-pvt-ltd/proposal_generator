"""
Data loader utilities
"""

import pandas as pd
import json
import os
from typing import Dict, Optional  # Added proper imports
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and manage data files"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def load_skills_data(self) -> pd.DataFrame:
        """Load skills data strictly from internal CSV (skill_company.csv)"""
        try:
            internal_path = self.config['data']['skills_internal']
            if os.path.exists(internal_path):
                df = pd.read_csv(internal_path)
                df['source'] = 'internal'
                logger.info(f"Loaded {len(df)} internal skill records from {internal_path}")
                return df
            logger.warning(f"Internal skills file not found: {internal_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading skills data: {str(e)}")
            return pd.DataFrame()
    
    def load_company_profile(self) -> Dict:
        """Load company profile data"""
        
        profile_path = self.config['data'].get('company_profile')
        
        if profile_path and os.path.exists(profile_path):
            try:
                _, ext = os.path.splitext(profile_path)
                # Support JSON profiles directly
                if ext.lower() == '.json':
                    with open(profile_path, 'r') as f:
                        return json.load(f)
                # Support Markdown or text profiles as raw content
                elif ext.lower() in {'.md', '.markdown', '.txt'}:
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if not content.strip():
                        logger.warning(f"Company profile file is empty: {profile_path}")
                    return {
                        "name": self.config['company']['name'],
                        "location": self.config['company']['location'],
                        "profile_markdown": content,
                    }
                else:
                    # Fallback: try JSON first, then return as text wrapper
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                    try:
                        return json.loads(data)
                    except Exception:
                        logger.info(f"Treating company profile as text (ext: {ext})")
                        return {
                            "name": self.config['company']['name'],
                            "location": self.config['company']['location'],
                            "profile_text": data,
                        }
            except Exception as e:
                logger.error(f"Error loading company profile: {str(e)}")
        
        # No hardcoded defaults; return empty dict if not available
        return {}
    
    def load_case_studies(self) -> list:
        """Load case studies"""
        
        case_studies_path = self.config['data'].get('case_studies')
        
        if case_studies_path and os.path.exists(case_studies_path):
            try:
                _, ext = os.path.splitext(case_studies_path)
                if ext.lower() == '.json':
                    with open(case_studies_path, 'r') as f:
                        return json.load(f)
                elif ext.lower() in {'.md', '.markdown', '.txt'}:
                    with open(case_studies_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Return as a single-item list containing markdown source
                    return [{"case_studies_markdown": content}]
                else:
                    # Try JSON, else wrap as text
                    with open(case_studies_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                    try:
                        return json.loads(data)
                    except Exception:
                        return [{"case_studies_text": data}]
            except Exception as e:
                logger.error(f"Error loading case studies: {str(e)}")
        
        # No hardcoded defaults; return empty list if not available
        return []