from colorama import Fore, Back, Style
from typing import Tuple

LOG_LEVEL_TO_COLOR = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "magenta",
}
    
# Colored String Representation utilites
FORE_COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "reset": Fore.RESET,
}

LIGHT_FORE_COLOR_MAP = {
    "black": Fore.LIGHTBLACK_EX,
    "red": Fore.LIGHTRED_EX,
    "green": Fore.LIGHTGREEN_EX,
    "yellow": Fore.LIGHTYELLOW_EX,
    "blue": Fore.LIGHTBLUE_EX,
    "magenta": Fore.LIGHTMAGENTA_EX,
    "cyan": Fore.LIGHTCYAN_EX,
    "white": Fore.LIGHTWHITE_EX,
}

BG_COLOR_MAP = {
    "black": Back.BLACK,
    "red": Back.RED,
    "green": Back.GREEN,
    "yellow": Back.YELLOW,
    "blue": Back.BLUE,
    "magenta": Back.MAGENTA,
    "cyan": Back.CYAN,
    "white": Back.WHITE,
    "reset": Back.RESET,
}

LIGHT_BG_COLOR_MAP = {
    "black": Back.LIGHTBLACK_EX,
    "red": Back.LIGHTRED_EX,
    "green": Back.LIGHTGREEN_EX,
    "yellow": Back.LIGHTYELLOW_EX,
    "blue": Back.LIGHTBLUE_EX,
    "magenta": Back.LIGHTMAGENTA_EX,
    "cyan": Back.LIGHTCYAN_EX,
    "white": Back.LIGHTWHITE_EX,
}

def colorize_log_message(message: str, log_level: str, bg_color=None, bold=True, light_fore=False, light_bg=False) -> Tuple[str, str]:

    log_level = log_level.upper()
    if log_level in list(LOG_LEVEL_TO_COLOR.keys()):
        text_color = LOG_LEVEL_TO_COLOR.get(log_level)
    else:
        raise KeyError(f"Given log level is not found in {list(LOG_LEVEL_TO_COLOR.keys())}. Pass a valid log level.")
    
    return (colorize_string(message, text_color, bg_color=bg_color, bold=bold, light_fore=light_fore, light_bg=light_bg),
            colorize_string(log_level, text_color, bg_color=bg_color, bold=bold, light_fore=light_fore, light_bg=light_bg))

def colorize_string(text, text_color, bg_color=None, bold=True, light_fore=True, light_bg=True):
    
    """
    Generate a colored string with optional background color and bold text using Colorama.

    Args:
        text (str): The text to be colored.
        text_color (str): The Colorama text color code.
        color (str, optional): The Colorama background color code. Defaults to None.
        bold (bool, optional): Whether to make the text bold. Defaults to False.

    Returns:
        str: The colored string.
    """

    FORE = LIGHT_FORE_COLOR_MAP if light_fore else FORE_COLOR_MAP
    BG = LIGHT_BG_COLOR_MAP if light_bg else BG_COLOR_MAP
    
    if text_color in list(FORE.keys()):
        text_color = FORE[text_color]
    else:
        raise KeyError(f"Given key does not match with available text color keys {list(FORE.keys())}")

    if bg_color and bg_color in list(BG.keys()):
        bg_color = BG[bg_color]
    else:
        raise KeyError(f"Given key does not match with available background color keys {list(BG.keys())}")

    style = Style.BRIGHT if bold else ''
    
    if bg_color:
        return f"{style}{text_color}{bg_color}{text.upper()}{Style.RESET_ALL}"
    else:
        return f"{style}{text_color}{text.upper()}{Style.RESET_ALL}"