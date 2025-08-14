import os
from dotenv import load_dotenv
load_dotenv(override=False)

def _has_streamlit_secrets():
    try:
        import streamlit as st; _ = st.secrets; return True
    except Exception:
        return False

def get_secret(key, default=None):
    if _has_streamlit_secrets():
        import streamlit as st
        try: return st.secrets[key]
        except Exception: pass
    return os.getenv(key, default)

def get_db_config():
    cfg = {}
    if _has_streamlit_secrets():
        import streamlit as st
        cfg = dict(st.secrets.get("db", {})) or {}
    return {
        "user":     cfg.get("user")     or os.getenv("DB_USER"),
        "password": cfg.get("password") or os.getenv("DB_PASSWORD"),
        "host":     cfg.get("host")     or os.getenv("DB_HOST", "localhost"),
        "port": int(cfg.get("port")     or os.getenv("DB_PORT", 3306)),
        "name":     cfg.get("name")     or os.getenv("DB_NAME"),
    }
