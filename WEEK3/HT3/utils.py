
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple

def calculate_psi(
    expected: Union[pd.Series, np.ndarray],
    actual: Union[pd.Series, np.ndarray],
    buckets: int = 10,
    buckettype: str = 'bins',
    handle_na: bool = False
) -> float:

    # Конвертируем np.array в pd.Series если необходимо
    if isinstance(expected, np.ndarray):
        expected = pd.Series(expected)
    if isinstance(actual, np.ndarray):
        actual = pd.Series(actual)

    def scale_range(input_series, n_bins):
        """Создает бакеты для числовых данных"""
        min_val = input_series.min()
        max_val = input_series.max()
        return np.linspace(min_val, max_val, n_bins + 1)
    
    # Обработка NA значений
    if handle_na:
        # Подсчитываем долю NA в каждой выборке
        expected_na_count = expected.isna().sum()
        actual_na_count = actual.isna().sum()
        
        expected_total = len(expected)
        actual_total = len(actual)
        
        expected_na_percent = expected_na_count / expected_total if expected_total > 0 else 0
        actual_na_percent = actual_na_count / actual_total if actual_total > 0 else 0

    else:
        expected_na_percent = 0
        actual_na_percent = 0

    expected_clean = expected.dropna()
    actual_clean = actual.dropna()

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        if handle_na and (expected_na_percent > 0 or actual_na_percent > 0):
            # Если есть только NA значения, рассчитываем PSI только для NA бина
            expected_na_percent = max(expected_na_percent, 0.0001)
            actual_na_percent = max(actual_na_percent, 0.0001)
            return (actual_na_percent - expected_na_percent) * np.log(actual_na_percent / expected_na_percent)
        else:
            raise ValueError("Недостаточно данных для расчета PSI после удаления NA")
    
    # Определяем границы бакетов на основе expected
    if buckettype == 'bins':
        breakpoints = scale_range(expected_clean, buckets)
    elif buckettype == 'quantiles':
        breakpoints = np.percentile(expected_clean, np.linspace(0, 100, buckets + 1))
    else:
        raise ValueError("buckettype должен быть 'bins' или 'quantiles'")
    
    # Убираем дубликаты границ
    breakpoints = np.unique(breakpoints)
    
    # Создаем бакеты для не-NA значений
    expected_percents = pd.cut(expected_clean, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts(normalize=True).sort_index() # type: ignore
    actual_percents = pd.cut(actual_clean, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts(normalize=True).sort_index() # type: ignore
    
    # Выравниваем индексы
    expected_percents, actual_percents = expected_percents.align(actual_percents, fill_value=0.0001)
    
    # Если обрабатываем NA, добавляем их как отдельный бин
    if handle_na:
        # Корректируем проценты с учетом NA
        expected_percents = expected_percents * (1 - expected_na_percent)
        actual_percents = actual_percents * (1 - actual_na_percent)
        
        # Добавляем NA бин
        expected_percents['NA'] = max(expected_na_percent, 0.0001)
        actual_percents['NA'] = max(actual_na_percent, 0.0001)
    
    # Заменяем нули на малое значение для избежания деления на ноль
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)
    
    # Рассчитываем PSI
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value




