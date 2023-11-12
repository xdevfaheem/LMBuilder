from functools import lru_cache
from importlib.util import find_spec
import pkg_resources
import yaml


def number_to_words(number):
    
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if 0 <= number < 10:
        return units[number]
    elif 10 <= number < 20:
        return teens[number - 10]
    elif 20 <= number < 100:
        return tens[number // 10] + (" " + units[number % 10] if number % 10 != 0 else "")
    elif 100 <= number < 1000:
        return units[number // 100] + " hundred" + (" and " + number_to_words(number % 100) if number % 100 != 0 else "")
    elif 1000 <= number < 1000000:
        return number_to_words(number // 1000) + " thousand" + (" " + number_to_words(number % 1000) if number % 1000 != 0 else "")
    elif 1000000 <= number < 1000000000:
        return number_to_words(number // 1000000) + " million" + (" " + number_to_words(number % 1000000) if number % 1000000 != 0 else "")
    elif 1000000000 <= number < 1000000000000:
        return number_to_words(number // 1000000000) + " billion" + (" " + number_to_words(number % 1000000000) if number % 1000000000 != 0 else "")
    elif 1000000000000 <= number < 1000000000000000:
        return number_to_words(number // 1000000000000) + " trillion" + (" " + number_to_words(number % 1000000000000) if number % 1000000000000 != 0 else "")
    else:
        raise ValueError("Number out of range")
    
def yaml_to_dict(yaml_path):
    # read yaml and return contents
    if not yaml_path.endswith(".yaml"):
        raise ValueError("Inappropriate file Yaml file needed.")
    with open(yaml_path, 'r') as file:
        try:
            yaml_dict =  yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error Occured when reading the file\n", exc)
    return yaml_dict

@lru_cache()
def is_package_available(package_name: str) -> bool:
    
    """Check if a package is available in your environment.
    >>> package_available('os')
    True
    >>> package_available('bla')
    False
    """
    
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False

def is_pkg_req_met(req, msg=False):

    """
    check if the package requirement with extras and version specifiers on available in your env.

    >>> is_pkg_req_met("torch>=0.1", msg=True)
    Package with specified version critiria 'torch>=0.1' met!
    >>> is_pkg_req_met("torch>=0.1")
    True
    >>> is_pkg_req_met("torch>100.0")
    False
    """

    try:
        pkg_resources.require(req)
        met = True
        message = f"Package with specified version critiria {req!r} met!"
    except Exception as ex:
        met = False
        message = f"{ex.__class__.__name__}: {ex}. Package with specified version critiria not met! \n Try running `pip install -U {req!r}`"
    
    if msg:
        print(message)
    
    return met

class PkgRequirement:
    """Boolean-like class for check of requirement with extras and version specifiers.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    """

    def __init__(self, requirement: str) -> None:
        self.requirement = requirement

    def _check_requirement(self) -> None:
        if not hasattr(self, "available"):
            try:
                pkg_resources.require(self.requirement)
                self.available = True
                self.message = f"Requirement {self.requirement!r} met"
            except Exception as ex:
                self.available = False
                self.message = f"{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`"

    def __bool__(self) -> bool:
        self._check_requirement()
        return self.available

    def __str__(self) -> str:
        self._check_requirement()
        return self.message

    def __repr__(self) -> str:
        return self.__str__()
