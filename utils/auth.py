"""
Authentication utility module for Streamlit app
"""
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from typing import Tuple, Optional


def load_auth_config(yaml_path: str = 'credentials.yaml') -> Optional[dict]:
    """
    Load authentication configuration from YAML file
    
    Args:
        yaml_path: Path to credentials YAML file
        
    Returns:
        Dictionary with authentication config or None if error
    """
    try:
        with open(yaml_path) as file:
            config = yaml.load(file, Loader=SafeLoader)
        return config
    except FileNotFoundError:
        st.error(f"❌ Credentials file not found: {yaml_path}")
        return None
    except yaml.YAMLError as e:
        st.error(f"❌ Invalid YAML format: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load credentials: {e}")
        return None


def create_authenticator(config: dict) -> Optional[stauth.Authenticate]:
    """
    Create and return authenticator object
    
    Args:
        config: Authentication configuration dictionary
        
    Returns:
        Authenticator object or None if error
    """
    try:
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config.get('preauthorized')
        )
        return authenticator
    except KeyError as e:
        st.error(f"❌ Missing required config key: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to initialize authenticator: {e}")
        return None


def authenticate_user(yaml_path: str = 'credentials.yaml') -> Tuple[Optional[bool], Optional[stauth.Authenticate]]:
    """
    Authenticate user with streamlit-authenticator
    
    Args:
        yaml_path: Path to credentials YAML file
        
    Returns:
        Tuple of (authentication_status, authenticator_object)
        - authentication_status: True if authenticated, False if wrong credentials, None if no input
        - authenticator_object: The authenticator instance for logout functionality
    """
    # Load configuration
    config = load_auth_config(yaml_path)
    if not config:
        return None, None
    
    # Create authenticator
    authenticator = create_authenticator(config)
    if not authenticator:
        return None, None
    
    # Render login widget
    try:
        name, auth_status, username = authenticator.login("Login", "main")
        
        # Store in session state
        st.session_state['authentication_status'] = auth_status
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        # Handle authentication results
        if auth_status is False:
            st.error("❌ Username/password is incorrect")
        elif auth_status is None:
            st.warning("🔐 Please enter your username and password")
            
        return auth_status, authenticator
        
    except Exception as e:
        st.error(f"❌ Authentication error: {e}")
        return None, None


def show_user_info(authenticator: stauth.Authenticate, location: str = "sidebar") -> None:
    """
    Display user info and logout button
    
    Args:
        authenticator: Authenticator object
        location: Where to show the logout button ("sidebar" or "main")
    """
    if st.session_state.get('authentication_status'):
        name = st.session_state.get('name', 'User')
        
        if location == "sidebar":
            with st.sidebar:
                st.write(f"Welcome *{name}*")
                authenticator.logout("Logout", "sidebar")
                st.divider()
        else:
            st.write(f"Welcome *{name}*")
            authenticator.logout("Logout", "main")


def require_authentication(yaml_path: str = 'credentials.yaml') -> Tuple[bool, Optional[stauth.Authenticate]]:
    """
    Wrapper function to require authentication before proceeding
    Stops the app if not authenticated
    
    Args:
        yaml_path: Path to credentials YAML file
        
    Returns:
        Tuple of (is_authenticated, authenticator)
    """
    auth_status, authenticator = authenticate_user(yaml_path)
    
    if not auth_status:
        st.stop()
        
    return auth_status, authenticator