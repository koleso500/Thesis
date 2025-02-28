import os

import safeaipackage
print(os.listdir(os.path.dirname(safeaipackage.__file__))) #['check_explainability.py', # 'check_fairness.py', 'check_robustness.py', 'core.py']
import safeaipackage.core
print(dir(safeaipackage.core))

import safeaipackage.check_fairness
print(dir(safeaipackage.check_fairness))

import safeaipackage.check_robustness
print(dir(safeaipackage.check_robustness))

import safeaipackage.check_explainability
print(dir(safeaipackage.check_explainability))

