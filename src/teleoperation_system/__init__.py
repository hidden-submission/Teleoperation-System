"""
Project - main package.
"""

# Делаем внутренние пакеты доступными напрямую
from . import cameras
from . import data_utils
from . import robots
from . import teleoperators
from . import utils

__version__ = "2.0"
__all__ = ['cameras', 'data_utils', 'robots', 'teleoperators', 'utils']