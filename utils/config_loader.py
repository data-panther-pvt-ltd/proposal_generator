"""
Enhanced Config Loader
Loads settings from YAML and enriches with company profile from markdown
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from utils.company_profile_parser import CompanyProfileParser

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and merge configuration from multiple sources"""
    
    def __init__(self, 
                 settings_path: str = "config/settings.yml",
                 profile_path: str = "data/company_profile.md",
                 auto_fetch: bool = True):
        """
        Initialize config loader
        
        Args:
            settings_path: Path to YAML settings file
            profile_path: Path to company profile markdown
            auto_fetch: If True, automatically fetch company details from markdown
        """
        self.settings_path = Path(settings_path)
        self.profile_path = Path(profile_path)
        self.auto_fetch = auto_fetch
        self._config = None
    
    def load_yaml_settings(self) -> Dict[str, Any]:
        """Load base settings from YAML file"""
        if not self.settings_path.exists():
            raise FileNotFoundError(f"Settings file not found at {self.settings_path}")
        
        with open(self.settings_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_company_profile(self) -> Dict[str, str]:
        """Load company details from markdown profile"""
        try:
            parser = CompanyProfileParser(str(self.profile_path))
            return parser.extract_company_details()
        except FileNotFoundError:
            logger.warning(f"Company profile not found at {self.profile_path}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing company profile: {e}")
            return {}
    
    def merge_configs(self, base_config: Dict, company_data: Dict) -> Dict:
        """
        Merge company data from markdown into base config
        
        Priority: Markdown data overrides YAML for company section
        """
        merged = base_config.copy()
        
        if self.auto_fetch and company_data:
            # Create or update company section
            if 'company' not in merged:
                merged['company'] = {}
            
            # Map markdown fields to YAML structure
            field_mapping = {
                'name': 'name',
                'location': 'location', 
                'tagline': 'tagline',
                'website': 'website',
                'email': 'email',
                'phone': 'phone',
                'address': 'address',
                'founded': 'founded_year'
            }
            
            for md_field, yaml_field in field_mapping.items():
                if md_field in company_data and company_data[md_field]:
                    merged['company'][yaml_field] = company_data[md_field]
                    logger.debug(f"Updated company.{yaml_field} from markdown: {company_data[md_field]}")
        
        return merged
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load complete configuration
        
        Returns:
            Merged configuration dictionary
        """
        if self._config is not None:
            return self._config
        
        # Load base YAML settings
        base_config = self.load_yaml_settings()
        
        # Load company profile from markdown if auto_fetch is enabled
        if self.auto_fetch:
            company_data = self.load_company_profile()
            self._config = self.merge_configs(base_config, company_data)
            logger.info("Configuration loaded with company profile from markdown")
        else:
            self._config = base_config
            logger.info("Configuration loaded from YAML only")
        
        return self._config
    
    def get_company_info(self) -> Dict[str, str]:
        """Get just the company information section"""
        config = self.load_config()
        return config.get('company', {})
    
    def reload(self) -> Dict[str, Any]:
        """Force reload configuration from sources"""
        self._config = None
        return self.load_config()


# Helper functions for backward compatibility
def load_config(config_path: str = "config/settings.yml", 
                profile_path: str = "data/company_profile.md",
                auto_fetch: bool = True) -> Dict[str, Any]:
    """
    Load configuration with optional markdown profile integration
    
    Args:
        config_path: Path to YAML settings
        profile_path: Path to company profile markdown
        auto_fetch: Whether to automatically fetch from markdown
        
    Returns:
        Merged configuration dictionary
    """
    loader = ConfigLoader(config_path, profile_path, auto_fetch)
    return loader.load_config()


def load_config_with_profile(config_path: str = "config/settings.yml") -> Dict[str, Any]:
    """
    Load configuration with company profile from markdown
    This is the recommended function for loading settings
    """
    return load_config(config_path, auto_fetch=True)


def load_config_yaml_only(config_path: str = "config/settings.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML only (backward compatibility)
    """
    return load_config(config_path, auto_fetch=False)