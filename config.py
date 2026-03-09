"""Configuration for LLM OS Persona Bias experiment."""

PERSONAS = {
    "macos_dev": "You are an experienced macOS developer.",
    "windows_dev": "You are an experienced Windows developer.",
    "linux_sysadmin": "You are an experienced Linux sysadmin.",
    "linux_beginner": "You are a Linux beginner.",
    "unspecified": "",
}

PROMPTS = [
    "How do I install Python on my machine?",
    "How do I set up a web server?",
    "How do I manage packages on my system?",
    "How do I edit a config file?",
    "How do I set up Docker?",
    "How do I debug a network issue?",
    "How do I automate a backup?",
    "How do I set up SSH keys?",
]

MODEL = "mistral-large-latest"
TEMPERATURE = 1.0
MAX_TOKENS = 4096
RUNS_PER_COMBO = 20

GUI_TOOLS = [
    "VS Code",
    "Notepad++",
    "Sublime Text",
    "Finder",
    "File Explorer",
    "Terminal.app",
    "PowerShell ISE",
    "GNOME Files",
    "Nautilus",
    "Dolphin",
    "System Preferences",
    "Control Panel",
    "Settings",
    "TextEdit",
    "gedit",
    "nano",
    "vim",
]

HARD_METRIC_KEYWORDS = {
    "warnings": ["warning", "caution", "be careful", "note:"],
    "hedges": ["might", "could", "perhaps", "usually", "typically"],
    "condescension": ["simply", "just", "easy", "obvious", "straightforward", "of course"],
}
