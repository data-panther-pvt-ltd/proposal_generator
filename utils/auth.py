"""
Authentication utility module for Streamlit app
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from typing import Tuple, Optional


def _load_auth_config(yaml_path: str = "credentials.yaml") -> Optional[dict]:
    """Load authentication configuration from YAML file"""
    try:
        with open(yaml_path) as file:
            return yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error(f"❌ Credentials file not found: {yaml_path}")
    except yaml.YAMLError as e:
        st.error(f"❌ Invalid YAML format: {e}")
    except Exception as e:
        st.error(f"❌ Failed to load credentials: {e}")
    return None


def _create_authenticator(config: dict) -> Optional[stauth.Authenticate]:
    """Create and return authenticator object"""
    try:
        return stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config.get("preauthorized"),
        )
    except KeyError as e:
        st.error(f"❌ Missing required config key: {e}")
    except Exception as e:
        st.error(f"❌ Failed to initialize authenticator: {e}")
    return None


def authenticate(yaml_path: str = "credentials.yaml") -> Tuple[bool, Optional[stauth.Authenticate]]:
    """
    Main entry: Authenticate user and return authenticator.

    Usage in main app:
        auth_status, authenticator = authenticate()
    """
    config = _load_auth_config(yaml_path)
    if not config:
        st.stop()

    authenticator = _create_authenticator(config)
    if not authenticator:
        st.stop()

    try:
        name, auth_status, username = authenticator.login("Login", "main")

        st.session_state["authentication_status"] = auth_status
        st.session_state["name"] = name
        st.session_state["username"] = username

        if auth_status is False:
            st.error("❌ Username/password is incorrect")
        elif auth_status is None:
            st.warning("🔐 Please enter your username and password")

        return bool(auth_status), authenticator

    except Exception as e:
        st.error(f"❌ Authentication error: {e}")
        st.stop()


def show_user_info(authenticator: stauth.Authenticate, location: str = "sidebar") -> None:
    """
    Display user info + logout button.

    Usage in main app:
        show_user_info(authenticator)
    """
    if st.session_state.get("authentication_status"):
        name = st.session_state.get("name", "User")

        if location == "sidebar":
            with st.sidebar:
                st.success(f"👋 Welcome *{name}*")
                authenticator.logout("🚪 Logout", "sidebar")
                st.divider()
        else:
            st.success(f"👋 Welcome *{name}*")
            authenticator.logout("🚪 Logout", "main")
