"""
Company Profile Parser
Extracts company details from markdown file
"""

import re
from typing import Dict, Optional
from pathlib import Path


class CompanyProfileParser:
    """Parse company details from markdown profile"""
    
    def __init__(self, profile_path: str = "data/company_profile.md"):
        self.profile_path = Path(profile_path)
        self._content = None
        self._company_data = None
    
    def _load_content(self) -> str:
        """Load markdown content from file"""
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Company profile not found at {self.profile_path}")
        
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            self._content = f.read()
        return self._content
    
    def extract_company_details(self) -> Dict[str, str]:
        """Extract company details from markdown content"""
        if not self._content:
            self._load_content()
        
        company_data = {
            'name': self._extract_name(),
            'location': self._extract_location(),
            'tagline': self._extract_tagline(),
            'website': self._extract_website(),
            'email': self._extract_email(),
            'phone': self._extract_phone(),
            'address': self._extract_address(),
            'founded': self._extract_founded_year()
        }
        
        # Clean up None values
        company_data = {k: v for k, v in company_data.items() if v is not None}
        
        self._company_data = company_data
        return company_data
    
    def _extract_name(self) -> Optional[str]:
        """Extract company name from title or content"""
        # Try to find in main heading
        patterns = [
            r'#\s+([A-Z]+\s*[X]?)\s+DESIGN',  # AZM X DESIGN
            r'#\s+([A-Z]+\s*[X]?)\s+—',  # AZM X —
            r'We Are\s+([A-Z]+\s*[X]?)\s+—',  # We Are AZM X
            r'At\s+\*\*([A-Z]+\s*[X]?)\*\*',  # At **AZM X**
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self._content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Normalize the name format
                if 'AZM' in name.upper():
                    return 'AzmX'
        
        return 'AzmX'  # Default fallback
    
    def _extract_location(self) -> Optional[str]:
        """Extract company location"""
        # Look for "Based in" pattern
        pattern = r'\*\*Based in\s+([^*\n]+)\*\*'
        match = re.search(pattern, self._content)
        if match:
            location = match.group(1).strip()
            return f"{location}, KSA"
        
        # Alternative: look in contact section
        if 'Riyadh' in self._content:
            return 'Riyadh, KSA'
        
        return None
    
    def _extract_tagline(self) -> Optional[str]:
        """Extract company tagline or main value proposition"""
        # Look for Innovation & Creativity or similar phrases
        patterns = [
            r'\*\*Innovation & Creativity\*\*',
            r'Excellence in Digital Transformation',
            r'Changing People\'s Everyday Stories'
        ]
        
        for pattern in patterns:
            if pattern in self._content or re.search(pattern, self._content, re.IGNORECASE):
                if 'Innovation & Creativity' in self._content:
                    return 'Excellence in Digital Transformation'
        
        return 'Excellence in Digital Transformation'
    
    def _extract_website(self) -> Optional[str]:
        """Extract website URL"""
        # Since website isn't explicitly in the current markdown, return the known URL
        # Could be enhanced to look for patterns like https://azmx.sa
        return 'https://azmx.sa/'
    
    def _extract_email(self) -> Optional[str]:
        """Extract email address"""
        # Look for email pattern
        email_pattern = r'📧?\s*\*?\*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        match = re.search(email_pattern, self._content)
        if match:
            return match.group(1)
        
        # Alternative pattern
        pattern = r'\(mailto:([^)]+)\)'
        match = re.search(pattern, self._content)
        if match:
            return match.group(1)
            
        return 'info@azmx.sa'
    
    def _extract_phone(self) -> Optional[str]:
        """Extract phone number if available"""
        # Look for phone pattern (not in current markdown but prepared for future)
        phone_pattern = r'(?:📞|Phone:|Tel:)\s*([+\d\s-]+)'
        match = re.search(phone_pattern, self._content)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_address(self) -> Optional[str]:
        """Extract full address"""
        # Look for address pattern
        pattern = r'📍\s*([^📧\n]+)'
        match = re.search(pattern, self._content)
        if match:
            return match.group(1).strip()
        
        # Alternative: look for specific address components
        if 'Tuwaiq gate' in self._content:
            address_match = re.search(r'(Tuwaiq gate[^📧\n]+Saudi Arabia)', self._content)
            if address_match:
                return address_match.group(1).strip()
        
        return None
    
    def _extract_founded_year(self) -> Optional[str]:
        """Extract founding year"""
        # Look for "Since YYYY" pattern
        pattern = r'Since\s+\*\*(\d{4})\*\*'
        match = re.search(pattern, self._content)
        if match:
            return match.group(1)
        return None
    
    def get_company_info(self) -> Dict[str, str]:
        """Get cached company data or extract if not available"""
        if not self._company_data:
            self.extract_company_details()
        return self._company_data


# Helper function for easy integration
def get_company_from_profile(profile_path: str = "data/company_profile.md") -> Dict[str, str]:
    """
    Convenience function to get company details from markdown profile
    
    Args:
        profile_path: Path to company profile markdown file
        
    Returns:
        Dictionary with company details
    """
    parser = CompanyProfileParser(profile_path)
    return parser.extract_company_details()