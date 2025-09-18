import getpass
import os
import secrets
import sys
from typing import Dict, Any

try:
    import bcrypt
except Exception:
    print("ERROR: Missing 'bcrypt' library. Install with: pip install bcrypt")
    sys.exit(1)

try:
    import yaml
except Exception:
    print("ERROR: Missing 'pyyaml' library. Install with: pip install pyyaml")
    sys.exit(1)


def hash_password(plain_password: str) -> str:
    """
    Hash a password using bcrypt and return the string (utf-8) with $2b$ prefix.
    """
    pw_bytes = plain_password.encode("utf-8")
    hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def ask_user_entry() -> Dict[str, Any]:
    """
    Ask for a single user's details and return a dict entry suitable for YAML.
    """
    while True:
        username = input("Enter username (slug, e.g. 'admin'): ").strip()
        if not username:
            print("Username cannot be empty. Try again.")
            continue
        # basic validation: no spaces
        if " " in username:
            print("Please avoid spaces in username. Use underscore or dash.")
            continue
        break

    name = input("Enter display name (e.g. 'Admin User'): ").strip()
    email = input("Enter email (optional, press Enter to skip): ").strip()

    while True:
        pw = getpass.getpass("Enter password (input hidden): ")
        if not pw:
            print("Password cannot be empty. Try again.")
            continue
        pw2 = getpass.getpass("Confirm password: ")
        if pw != pw2:
            print("Passwords do not match. Try again.")
            continue
        break

    hashed = hash_password(pw)
    entry = {
        "email": email if email else "",
        "name": name if name else username,
        "password": hashed,
    }
    return username, entry


def main():
    print("=== credentials.yaml generator for streamlit-authenticator ===")
    print("You will be prompted to add one or more users.\n")

    users = {}
    while True:
        username, entry = ask_user_entry()
        if username in users:
            print(f"Username '{username}' already added. Overwriting the entry.")
        users[username] = entry

        more = input("Add another user? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            break
        print("")  # space between entries

    # Cookie / file options
    print("\n--- Cookie / file configuration ---")
    default_cookie_name = "some_cookie_name"
    default_expiry = "1"  # days

    cookie_name = input(f"Cookie name [{default_cookie_name}]: ").strip() or default_cookie_name
    expiry_days_raw = input(f"Cookie expiry days [{default_expiry}]: ").strip() or default_expiry
    try:
        expiry_days = int(expiry_days_raw)
    except ValueError:
        print("Invalid expiry days, defaulting to 1")
        expiry_days = 1

    # generate a secure random key if user doesn't provide
    default_key = secrets.token_urlsafe(24)
    cookie_key = input(f"Cookie key (leave blank to auto-generate): ").strip() or default_key

    # output file path
    default_path = "credentials.yaml"
    out_path = input(f"Output YAML path [{default_path}]: ").strip() or default_path
    out_path = os.path.abspath(out_path)

    # Construct YAML structure
    yaml_obj = {
        "credentials": {
            "usernames": users
        },
        "cookie": {
            "expiry_days": expiry_days,
            "key": cookie_key,
            "name": cookie_name
        }
    }

    # Write YAML
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            # Use safe_dump to keep output clean
            yaml.safe_dump(yaml_obj, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        print(f"\n✅ Created credentials file at: {out_path}")
    except Exception as e:
        print(f"\n❌ Failed to write YAML file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
