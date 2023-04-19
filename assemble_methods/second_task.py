import pandas as pd

from first_task import test_accuracy_dependence_on_estimators_amount

def vehicle_map(element):
    if element == 'opel':
        return 1
    elif element == 'bus':
        return 0
    elif element == 'van':
        return -1
    elif element == 'saab':
        return 5
    else:
        return element

def second_task():
    source = pd.read_csv(r'files\vehicle.csv')
    test_accuracy_dependence_on_estimators_amount(source,  vehicle_map,False)