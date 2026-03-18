import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





print("\n\n\n--- Импорт временного ряда ---")

time   = []
values = []

with open('weather.csv', mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    for row in dict_reader:
        r = row["# Метеостанция Стерлитамак"]

        if "14:00" in r:
            time.append(r[0:10])

            end = r.index(";", 18)
            values.append(float(r[18:end - 1]))

time.reverse()
values.reverse()





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 1 / Определение типа процесса =====")



print("\n\n\n--- Задание 1 / Анализ графика исходного ряда ---")

# plt.plot(time, values)
# plt.title("Погода в Стерлитамаке за 2021-2025 годы", fontweight='bold')
# plt.xlabel("Дата")
# plt.ylabel("Температура")
# plt.show()



print("\n\n\n--- Задание 2 / Анализ коррелограмм ---")

values_first_difference  = np.diff(values)
values_second_difference = np.diff(values, 2)
values_third_difference  = np.diff(values, 3)



# plot_acf(values,                    lags=12)
# plt.show()
#
# plot_acf(values_first_difference,   lags=12)
# plt.show()
#
# plot_acf(values_second_difference,  lags=12)
# plt.show()
#
# plot_pacf(values,                   lags=12)
# plt.show()
#
# plot_pacf(values_first_difference,  lags=12)
# plt.show()
#
# plot_pacf(values_second_difference, lags=12)
# plt.show()



print("\n\n\n--- Задание 3 / Проведение тестов Дики-Фуллера ---")

test_1          = adfuller(values_first_difference,  regression='ct', regresults=True)
test_2          = adfuller(values,                   regression='ct', regresults=True)
test_3          = adfuller(values_second_difference, regression='n',  regresults=True)
extended_test_3 = adfuller(values_second_difference, regression='c',  regresults=True)
test_4          = adfuller(values_first_difference,  regression='n',  regresults=True)
extended_test_4 = adfuller(values_first_difference,  regression='c',  regresults=True)
test_5          = adfuller(values,                   regression='n',  regresults=True)
extended_test_5 = adfuller(values,                   regression='c',  regresults=True)

tests       = [test_1, test_2, test_3, extended_test_3, test_4, extended_test_4, test_5, extended_test_5]
regressions = ["ct", "ct", "n", "c", "n", "c", "n", "c"]
test_names  = ["Тест 1              ", "Тест 2              ",
               "Тест 3              ", "Тест 3 (расширенный)",
               "Тест 4              ", "Тест 4 (расширенный)",
               "Тест 5              ", "Тест 5 (расширенный)"]



print("\n\n\n--- Задание 4 / Определение типа процесса ---")

result = []

for test, regression in zip(tests, regressions):
    # ADF-статистика
    if test[0] > test[2]["1%"]:
        result.append(False)
        continue

    # F-statistic
    if test[3].resols.f_pvalue >= 0.05:
        result.append(False)
        continue

    # p-value
    if regression == "ct":
        if max(test[3].resols.pvalues[0], test[3].resols.pvalues[-2], test[3].resols.pvalues[-1]) >= 0.05:
            result.append(False)
            continue
    elif regression == "c":
        if max(test[3].resols.pvalues[0], test[3].resols.pvalues[-1]) >= 0.05:
            result.append(False)
            continue
    else:
        if test[3].resols.pvalues[0] >= 0.05:
            result.append(False)
            continue

    # Durbin-Watson statistic
    if not(1.6 < durbin_watson(test[3].resols.resid) < 2.4):
        result.append(False)

    result.append(True)


print("Результаты тестов:")
for i in range(len(result)):
    if result[i]:
        print(test_names[i], "- выполнен")
    else:
        print(test_names[i], "- не выполнен")



if result[2] and result[4] and (result[6] or result[7]):
    res = "DS I(0)"
elif result[2] and (result[4] or result[5]):
    res = "DS I(1)"
elif result[2] or result[3]:
    res = "DS I(2)"
elif result[0] and result[1] and result[4]:
    res = "TS + DS"
elif result[0] and result[1]:
    res = "TS"
else:
    raise NameError("ТИП ПРОЦЕССА НЕ ОПРЕДЕЛЕН")

print("\nТип процесса: " + res)





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 2 / Структурные изменения =====")



print("\n\n\n--- Задание 1 / Проведение теста Кванда-Эндрюса ---")
t = [i for i in range(1, len(values) + 1)]
X = sm.add_constant(t)

start = int(len(values) * 0.15)
end   = int(len(values) * 0.85)
k     = X.shape[1]

f_statistics = []
for i in range(start, end):
    x1, y1 = X[:i], values[:i]
    x2, y2 = X[i:], values[i:]

    S  = sm.OLS(values, X).fit().ssr
    S1 = sm.OLS(y1, x1).fit().ssr
    S2 = sm.OLS(y2, x2).fit().ssr

    f_statistics.append(((S - (S1 + S2)) / k) / ((S1 + S2) / (len(values) - 2 * k)))

breakpoint_index = start + f_statistics.index(max(f_statistics))
p_value = 1 - stats.f.cdf(max(f_statistics), k, len(values) - 2 * k)
print(f"Точка излома - {time[breakpoint_index]} (p-value: {p_value})")



print("\n\n\n--- Задание 2 / Ввод фиктивных переменных ---")

ds  = [0] * breakpoint_index + [1] + [0] * (len(values) - breakpoint_index - 1)
ds1 = [0] * breakpoint_index + [1] * (len(values) - breakpoint_index)
dt  = [0] * breakpoint_index + [i for i in range(1, (len(values) - breakpoint_index) + 1)]



print("\n\n\n--- Задание 3 / Построение моделей ---")

def building_models(values, *variables: list):
    X     = sm.add_constant(np.column_stack(variables))
    model = sm.OLS(values, X).fit()
    return model



ds_first_difference   = ds[:-1]
ds_second_difference  = ds[:-2]
ds_third_difference   = ds[:-3]

ds1_first_difference  = ds1[:-1]
ds1_second_difference = ds1[:-2]
ds1_third_difference  = ds1[:-3]

values_for_first_difference = values[:-1]

values_first_difference_for_second_difference = values_first_difference[:-1]
values_second_difference_for_third_difference = values_second_difference[:-1]



if res == "TS + DS" or res == "TS":
    model_1 = building_models(values, t, dt)
    model_2 = building_models(values, t, ds1)
    model_3 = building_models(values, t, dt, ds1)

if res == "DS I(0)":
    model_1 = building_models(values_first_difference, values_for_first_difference, ds_first_difference)
    model_2 = building_models(values_first_difference, values_for_first_difference, ds1_first_difference)
    model_3 = building_models(values_first_difference, values_for_first_difference, ds_first_difference, ds1_first_difference)

if res == "DS I(1)":
    model_1 = building_models(values_second_difference, values_first_difference_for_second_difference, ds_second_difference)
    model_2 = building_models(values_second_difference, values_first_difference_for_second_difference, ds1_second_difference)
    model_3 = building_models(values_second_difference, values_first_difference_for_second_difference, ds_second_difference, ds1_second_difference)

if res == "DS I(2)":
    model_1 = building_models(values_third_difference, values_second_difference_for_third_difference, ds_third_difference)
    model_2 = building_models(values_third_difference, values_second_difference_for_third_difference, ds1_third_difference)
    model_3 = building_models(values_third_difference, values_second_difference_for_third_difference, ds_third_difference, ds1_third_difference)



print("\n\n\n--- Задание 4 / Проверка статистической значимости ---")

def significance_test(model):
    # F-statistic
    if model.f_pvalue >= 0.05:
        return False

    # p-value
    for i in model.pvalues[1:]:
        if i >= 0.05:
            return False

    # Durbin-Watson statistic
    if not(1.6 < durbin_watson(model.resid) < 2.4):
        return False

    return True



print("\n\n\n--- Задание 5 / Оценка качества ---")

model_1 = significance_test(model_1)
model_2 = significance_test(model_2)
model_3 = significance_test(model_3)



print("\n\n\n--- Задание 6 / Определение типа процесса c учетом структурных изменений ---")

if model_1 or model_2 or model_3:
    print(res + " со структурным скачком")
else:
    print(res + " без структурного скачка")





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 3 / Выделение детерминированных компонент из структуры ряда =====")



print("\n\n\n--- Задание 1 / Анализ графика исходного ряда ---")

# plt.plot(time, values)
# plt.title("Погода в Стерлитамаке за 2021-2025 годы", fontweight='bold')
# plt.xlabel("Дата")
# plt.ylabel("Температура")
# plt.show()



print("\n\n\n--- Задание 2 / Анализ коррелограмм ---")

def acf_pacf(values):
    # plot_acf(values,  lags=(len(values) // 2))
    # plt.show()
    #
    # plot_pacf(values, lags=(len(values) // 2))
    # plt.show()
    pass



if res == "DS I(0)":
    current_values = values.copy()

if res == "DS I(1)":
    current_values = values_first_difference.copy()

if res == "DS I(2)":
    current_values = values_second_difference.copy()

if res == "TS + DS" or res == "TS":
    model_type = 0
    while model_type != "1" and model_type != "2":
        print("\nУкажите тип модели"
              "\nВведите '1' - если аддитивная"
              "\nВведите '2' - если мультипликативная")
        model_type = input("Ввод        - ")

    if model_type == "1":
        leftovers = sm.OLS(values, X).fit().resid
    else:
        leftovers = sm.OLS(np.log([(i + 1 + min(values) * (-1)) for i in values]), X).fit().resid

    if res == "TS":
        current_values = leftovers
    else:
        current_values = np.diff(leftovers)

acf_pacf(current_values)



print("\n\n\n--- Задание 3 / Выделение сезонности ---")

# Первый способ: оценка сезонности с помощью тригонометрических функций

def first_method(values):
    best   = None
    best_n = 0

    for n in range(2, 10, 2):
        t = np.arange(1, len(values) + 1)

        sin_t = np.sin(n * np.pi * t / T)
        cos_t = np.cos(n * np.pi * t / T)

        model = sm.OLS(values, sm.add_constant(np.column_stack((sin_t, cos_t)))).fit()

        if best is None or (1.0 - model.rsquared) < best.rsquared:
            best    = model
            best_n = n

    print(f"\nИндекс детерминации: {best.rsquared} (достигается при {best_n} * pi)")
    acf_pacf(best.resid)



    plt.plot(time[:len(values)], values)
    plt.plot(time[:len(values)], np.subtract(values, best.resid))
    plt.plot(time[:len(values)], best.resid)
    plt.show()



T = int(input("Введите период сезонности: "))

first_method(current_values)



# Второй способ: оценка методом сезонных поправок (индексов)



# Третий способ: оценка методом введения фиктивных переменных



# Четвертый способ:



print("\n\n\n--- Задание 4 / Удаление детерминированного тренда ---")
print("\n\n\n--- Задание 5 / x ---")





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 4 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 5 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 6 =====")
