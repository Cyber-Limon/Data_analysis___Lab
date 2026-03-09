import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



print("\n\n\n--- Импорт временного ряда ---")

weather_time = []
weather_value = []

with open('weather.csv', mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    for row in dict_reader:
        r = row["# Метеостанция Стерлитамак"]

        if "14:00" in r:
            weather_time.append(r[0:10])

            end = r.index(";", 18)
            weather_value.append(float(r[18:end - 1]))

weather_time.reverse()
weather_value.reverse()





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 1 =====")



print("\n\n\n--- Задание 1 / Анализ графика исходного ряда ---")

plt.plot(weather_time, weather_value)
plt.title("Погода в Стерлитамаке за 2021-2025 годы", fontweight='bold')
plt.xlabel("Дата")
plt.ylabel("Температура")
plt.show()



print("\n\n\n--- Задание 2 / Анализ коррелограмм ---")

first_difference = []
for i in range(len(weather_value) - 1):
    first_difference.append(weather_value[i + 1] - weather_value[i])

second_difference = []
for i in range(len(first_difference) - 1):
    second_difference.append(first_difference[i + 1] - first_difference[i])

plot_acf(weather_value,      lags=12)
plt.show()

plot_acf(first_difference,   lags=12)
plt.show()

plot_acf(second_difference,  lags=12)
plt.show()

plot_pacf(weather_value,     lags=12)
plt.show()

plot_pacf(first_difference,  lags=12)
plt.show()

plot_pacf(second_difference, lags=12)
plt.show()



print("\n\n\n--- Задание 3 / Проведение тестов Дики-Фуллера ---")

test_1          = adfuller(first_difference,  regression='ct', regresults=True)
test_2          = adfuller(weather_value,     regression='ct', regresults=True)
test_3          = adfuller(second_difference, regression='n',  regresults=True)
extended_test_3 = adfuller(second_difference, regression='c',  regresults=True)
test_4          = adfuller(first_difference,  regression='n',  regresults=True)
extended_test_4 = adfuller(first_difference,  regression='c',  regresults=True)
test_5          = adfuller(weather_value,     regression='n',  regresults=True)
extended_test_5 = adfuller(weather_value,     regression='c',  regresults=True)

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





print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 2 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 3 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 4 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 5 =====")
print("\n\n\n\n\n===== ЛАБОРАТОРНАЯ РАБОТА 6 =====")
