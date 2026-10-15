"""Проверка: примеры допустимых и недопустимых решений (с двумя ограничениями)."""
import numpy as np
from ga_lab2 import (generate_data, is_feasible, total_cost, total_volume,
                     deficit, repair_constraints)

data = generate_data(seed=42)

print(f"Бюджет: {data['budget']}, полная стоимость: "
      f"{total_cost(data['demand'], data['cost'])}")
print(f"Объём: {data['volume_cap']}, полный объём: "
      f"{total_volume(data['demand'], data['volume'])}")
print(f"Стоимость минимальных запасов: "
      f"{total_cost(data['min_stock'], data['cost'])}")
print(f"Объём минимальных запасов: "
      f"{total_volume(data['min_stock'], data['volume'])}")
print()

# ---------- 2 допустимых решения ----------
x1 = data["min_stock"].copy()
print("Допустимое №1 (только минимальные запасы):")
print(f"  стоимость = {total_cost(x1, data['cost'])}, "
      f"объём = {total_volume(x1, data['volume'])}, "
      f"дефицит = {deficit(x1, data['demand'], data['weight']):.2f}, "
      f"допустимо = {is_feasible(x1, data)}")

x2 = repair_constraints(data["demand"], data)
print("Допустимое №2 (обрезанный спрос под оба ограничения):")
print(f"  стоимость = {total_cost(x2, data['cost'])}, "
      f"объём = {total_volume(x2, data['volume'])}, "
      f"дефицит = {deficit(x2, data['demand'], data['weight']):.2f}, "
      f"допустимо = {is_feasible(x2, data)}")

# ---------- 2 недопустимых ----------
x3 = data["demand"].copy()
print("Недопустимое №1 (полный спрос — превышает и бюджет, и объём):")
print(f"  стоимость = {total_cost(x3, data['cost'])} > "
      f"{data['budget']}, объём = {total_volume(x3, data['volume'])} > "
      f"{data['volume_cap']}, допустимо = {is_feasible(x3, data)}")

x4 = data["min_stock"].copy()
x4[0] -= 5
print("Недопустимое №2 (ниже минимума по материалу 0):")
print(f"  x[0] = {x4[0]}, min_stock[0] = {data['min_stock'][0]}, "
      f"допустимо = {is_feasible(x4, data)}")