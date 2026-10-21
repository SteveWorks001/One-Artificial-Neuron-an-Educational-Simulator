# -*- coding: utf-8 -*-
"""
Учебная программа: моделирование работы ОДНОГО искусственного нейрона.

y = f(w1*x1 + w2*x2 + ... + w10*x10 + b)

Только стандартная библиотека Python (tkinter, math). Без сторонних зависимостей.
Язык интерфейса переключается на ходу: русский (по умолчанию) или английский.
Никакого обучения нейросети — только прямой прозрачный расчёт по шагам.

Single Artificial Neuron Simulator

Author: SteveWorks001

Где что искать:
  0.   Все надписи (ru/en) .......... LANG / TEXTS / T(), справки GUIDE_RU и GUIDE_EN
  1-2. Входы (x) и веса (w) ......... read_numbers() / parse_all()
  3.   Смещение (bias) ............. STEP 4 функции compute_steps()
  4.   Умножение x * w ............. STEP 2 функции compute_steps()
       Взвешенная сумма Σ .......... STEP 3 функции compute_steps()
  5.   Функция активации ........... relu / sigmoid / tanh_fn / binary_step
  6.   Выход нейрона y ............. STEP 6 функции compute_steps()
  7.   Визуализация ................ class Diagram (вкладка «Схема нейрона»)
  8.   Пошаговый режим ............. next_step() и STEPS
"""

import math
import textwrap
import tkinter as tk
from tkinter import ttk, font as tkfont

N_INPUTS = 10
TOTAL_STEPS = 6

# ---------------------------------------------------------------
# Локализация: весь текст интерфейса живёт здесь (русский по умолчанию)
# ---------------------------------------------------------------
LANG = "ru"          # текущий язык: "ru" или "en"

# Полная справка: 14 разделов на каждом языке.
GUIDE_RU = """1. ОБЩАЯ ИДЕЯ
Что это: программа моделирует работу ОДНОГО искусственного нейрона.
Зачем: чтобы увидеть, как из десяти чисел получается одно число.
Как пользоваться: заполняйте поля слева и нажимайте кнопки.
Формула: y = f(w₁x₁ + w₂x₂ + ... + w₁₀x₁₀ + b)
Порядок: Входы → умножение на веса → сумма → bias → функция активации → выход.
Это не обучение нейросети: веса меняете только вы, вручную.

2. ВХОДЫ (x₁ ... x₁₀)
Что это: числа, которые нейрон получает «на вход».
Зачем: это его информация о мире — десять признаков.
Как пользоваться: введите число в столбец «Значение x» (можно отрицательное и дробное:
-1.5, 0.25; запятая тоже подходит). Пустое поле считается нулём.
Что увидите: значения x₁ ... x₁₀ в шапке решения и в схеме нейрона.

3. ИМЕНА ВХОДОВ (столбец «Имя входа»)
Что это: необязательная человеческая подпись для входа, например «Рост».
Зачем: с именами примеры понятнее: «Рост (x₁) = 180» вместо просто «x₁ = 180».
Как пользоваться: введите название в поле «Имя входа» напротив нужного входа.
Математическое обозначение x₁ сохраняется — имя только добавляется в скобках.
Что увидите: в окне решения и на вкладке «Вклады входов» вход подписан как
«Рост (x₁)»; если поле пустое — остаётся «x₁». На схеме вход остаётся коротким (x₁),
число стоит слева от кружка. На результат вычисления имена не влияют.

4. ВЕСА (w₁ ... w₁₀)
Что это: коэффициент «важности» каждого входа.
Зачем: вес показывает, насколько сильно данный вход влияет на нейрон.
Как пользоваться: введите вес в столбец «Вес w» напротив нужного входа.
Что увидите: произведение x × w на шаге 2. Вес 0 «отключает» вход, вес 2 удваивает его
вклад, отрицательный вес разворачивает вклад в обратную сторону.

5. СМЕЩЕНИЕ (BIAS)
Что это: число b, которое прибавляется к взвешенной сумме.
Зачем: оно сдвигает значение z перед применением функции активации — как порог
возбуждения нейрона.
Как пользоваться: введите значение в поле «Смещение b» под списком входов.
Что увидите на шаге 4:
  Взвешенная сумма: Σ = 1.6
  Смещение:         b = 0.4
  Значение до функции активации: z = 2.0
Положительный bias увеличивает z, отрицательный уменьшает, нулевой не меняет.

6. ФУНКЦИЯ АКТИВАЦИИ
Что это: способ превратить сумму z в выход нейрона y.
Зачем: без неё нейрон был бы просто сложением чисел.
Как пользоваться: выберите один из четырёх вариантов кнопками под полями.
Что увидите: описание выбранной функции и результат y = f(z).
  ReLU — возвращает 0 для отрицательных значений и само значение для положительных.
  Sigmoid — преобразует значение в диапазон от 0 до 1.
  Tanh — преобразует значение примерно в диапазон от -1 до 1.
  Binary Step — возвращает 0 или 1 в зависимости от значения.

7. КНОПКА «РАССЧИТАТЬ»
Что это: полный расчёт нейрона за один клик.
Зачем: быстро увидеть всю цепочку вычислений от входов до ответа.
Как пользоваться: заполните поля и нажмите «Рассчитать».
Что увидите: справа шесть шагов: входы, вклад каждого входа, Σ, bias, активация, y.
Значения также появятся на схеме нейрона.

8. ПОШАГОВЫЙ РЕЖИМ: «СЛЕДУЮЩИЙ ШАГ» И «СБРОСИТЬ ШАГИ»
Что это: ручной проход расчёта по одному этапу.
Зачем: лучше всего подходит для понимания — вы идёте в том темпе, в каком удобно.
Как пользоваться: нажмите «Следующий шаг» — откроется шаг 1 (входы); каждое
следующее нажатие добавляет новый блок: 2 — умножение, 3 — сумма, 4 — bias,
5 — активация, 6 — результат. «Сбросить шаги» возвращает к началу, к шагу 1.
Что увидите: в окне решения — один блок с пояснением «что происходит на этом шаге»,
а на схеме — подсветку пройденных этапов. Поля можно менять в любой момент: следующий
шаг уже использует новые значения.

9. СХЕМА НЕЙРОНА (вкладка «Схема нейрона»)
Что это: графическая схема пути данных.
Зачем: одним взглядом увидеть, как данные текут от входов к выходу.
Как пользоваться: откройте вкладку «Схема нейрона» и читайте слева направо:
кружки x₁...x₁₀ (входы) → линии с «× w» (умножение на вес) → большой кружок Σ
(сложение всех произведений) → подпись «+ b» (прибавление смещения, получается z) →
зелёный блок f(z) (функция активации) → кружок y (выход нейрона).
Что увидите: после «Рассчитать» или «Следующий шаг» пройденные этапы подсвечиваются
оранжевым, рядом с x показываются числа, у Σ — z, у y — результат.

10. ВКЛАД ВХОДОВ (шаг 2 решения)
Что это: произведение каждого входа на его вес.
Зачем: видно, кто из входов реально влияет на ответ нейрона.
Как пользоваться: задайте разные веса и нажмите «Рассчитать», затем откройте вкладку
«Вклады входов» справа.
Что увидите: строки вида «Рост (x₁)  180 × 0.5 = +90.0» и полоски-диаграммы:
вправо — положительный вклад (увеличивает z), влево — отрицательный (уменьшает z),
короткая полоска — почти не влияет, её нет — вклад нулевой.
Поэкспериментируйте: поменяйте один вес с 0.5 на -0.5 и посмотрите на z.

11. «СРАВНИТЬ ФУНКЦИИ АКТИВАЦИИ»
Что это: таблица из четырёх ответов нейрона при одном и том же z.
Зачем: увидеть, что выбор функции активации меняет результат.
Как пользоваться: сначала задайте входы, веса и bias, затем нажмите кнопку.
Что увидите: z = ... и выход y для ReLU, Sigmoid, Tanh и Binary Step с короткими
пояснениями. Все функции получают одно и то же число z, поэтому сравнение корректно.
Меняйте x, w, b и нажимайте кнопку снова — например при отрицательном z различия
заметны сильнее всего.

12. «АВТОМАТИЧЕСКИЙ ПЕРЕСЧЁТ»
Что это: переключатель мгновенного пересчёта.
Зачем: быстро исследовать, как значения влияют на ответ, без нажатия кнопок.
Как пользоваться: включите флажок — любое изменение x, веса, bias или функции сразу
пересчитывает нейрон и перерисовывает схему. Выключите флажок — расчёт пойдёт только
по кнопке «Рассчитать».
Что увидите: результат обновляется сам; кнопка «Рассчитать» работает при обоих режимах.

13. КНОПКА «ПРИМЕР»
Что это: готовый учебный пример.
Зачем: сразу увидеть работу нейрона на проверенных числах.
Как пользоваться: нажмите «Пример» — программа заполнит x₁=2, w₁=0.5, x₂=3, w₂=0.2,
остальные входы и веса нулями, bias=0.4, функцию ReLU, подпишет входы как «Рост» и
«Вес» и сама произведёт расчёт.
Что увидите: Σ = 1.6, z = 2.0, y = 2.0.

14. КНОПКА «СБРОСИТЬ»
Что это: возврат к пустой исходной форме.
Зачем: начать объяснение или эксперимент заново.
Как пользоваться: нажмите «Сбросить».
Что увидите: все x и w становятся 0, bias = 0, имена входов очищаются, функция —
ReLU, пошаговый режим возвращается к шагу 1, схема и окно решения очищаются."""

GUIDE_EN = """1. THE BIG IDEA
What is this: the program models the work of ONE artificial neuron.
Why: to see how ten numbers turn into a single number.
How to use it: fill in the fields on the left and press the buttons.
Formula: y = f(w₁x₁ + w₂x₂ + ... + w₁₀x₁₀ + b)
Order: inputs → multiplying by the weights → sum → bias → activation function → output.
This is not neural-network training: only you change the weights, by hand.

2. INPUTS (x₁ ... x₁₀)
What is this: the numbers the neuron receives at its input.
Why: this is its information about the world — ten features.
How to use it: type a number into the "Value x" column (negative and fractional values
are fine: -1.5, 0.25; a comma works too). An empty field counts as zero.
What you will see: the values of x₁ ... x₁₀ in the header of the solution and in the diagram.

3. INPUT NAMES (the "Input name" column)
What is this: an optional human label for an input, for example "Height".
Why: examples read better with names: "Height (x₁) = 180" instead of just "x₁ = 180".
How to use it: type a label into the "Input name" field next to the input you need.
The mathematical symbol x₁ is kept — the name is only added in brackets.
What you will see: in the solution window and on the "Input contributions" tab the input
is labelled "Height (x₁)"; if the field is empty it stays "x₁". On the diagram the input
stays short (x₁) and its number sits to the left of the circle. Names never affect the result.

4. WEIGHTS (w₁ ... w₁₀)
What is this: a "how important" coefficient for each input.
Why: a weight shows how strongly this input affects the neuron.
How to use it: type a weight into the "Weight w" column next to the input you need.
What you will see: the product x × w in step 2. A weight of 0 "disconnects" the input,
a weight of 2 doubles its contribution, a negative weight turns it the other way.

5. BIAS
What is this: the number b that is added to the weighted sum.
Why: it shifts the value z before the activation function is applied — like the firing
threshold of the neuron.
How to use it: type a value into the "Bias b" field below the list of inputs.
What you will see in step 4:
  Weighted sum: Σ = 1.6
  Bias:         b = 0.4
  Value before the activation function: z = 2.0
A positive bias raises z, a negative one lowers it, a zero one changes nothing.

6. ACTIVATION FUNCTION
What is this: the way the sum z is turned into the neuron output y.
Why: without it the neuron would be nothing but adding numbers.
How to use it: choose one of the four options with the buttons under the fields.
What you will see: a description of the chosen function and the result y = f(z).
  ReLU — returns 0 for negative values and the value itself for positive ones.
  Sigmoid — turns the value into the range from 0 to 1.
  Tanh — turns the value into roughly the range from -1 to 1.
  Binary Step — returns 0 or 1 depending on the sign of the value.

7. THE "CALCULATE" BUTTON
What is this: the full calculation of the neuron in one click.
Why: to see the whole chain of computations from the inputs to the answer quickly.
How to use it: fill in the fields and press "Calculate".
What you will see: six steps on the right: inputs, the contribution of each input, Σ,
bias, activation, y. The values also appear on the neuron diagram.

8. STEP-BY-STEP MODE: "NEXT STEP" AND "RESET STEPS"
What is this: walking through the calculation one stage at a time.
Why: it is the best way to understand the process — you go at whatever pace suits you.
How to use it: press "Next step" and step 1 (inputs) appears; every further press adds
a new block: 2 — multiplication, 3 — sum, 4 — bias, 5 — activation, 6 — result.
"Reset steps" returns you to the beginning, to step 1.
What you will see: one block in the solution window with the note "what happens at this
step", and the passed stages highlighted on the diagram. You may change the fields at any
moment: the next step already uses the new values.

9. NEURON DIAGRAM (the "Neuron diagram" tab)
What is this: a graphical picture of the path the data takes.
Why: to see at one glance how data flows from the inputs to the output.
How to use it: open the "Neuron diagram" tab and read it from left to right:
circles x₁...x₁₀ (inputs) → lines with "× w" (multiplying each input by its weight) →
the big circle Σ (adding all the products) → the "+ b" label (adding the bias, giving z) →
the green block f(z) (the activation function) → the circle y (the neuron output).
What you will see: after "Calculate" or "Next step" the passed stages turn orange,
numbers appear next to x, z next to Σ and the result next to y.

10. INPUT CONTRIBUTIONS (step 2 of the solution)
What is this: each input multiplied by its weight.
Why: you can see which inputs really influence the neuron's answer.
How to use it: set different weights and press "Calculate", then open the
"Input contributions" tab on the right.
What you will see: lines like "Height (x₁)  180 × 0.5 = +90.0" and bar diagrams:
to the right — a positive contribution (raises z), to the left — a negative one
(lowers z), a short bar — barely influences, no bar at all — a zero contribution.
Experiment: change one weight from 0.5 to -0.5 and watch z.

11. "COMPARE ACTIVATION FUNCTIONS"
What is this: a table of the four answers of the neuron at the very same z.
Why: to see that the choice of the activation function changes the result.
How to use it: first set the inputs, the weights and the bias, then press the button.
What you will see: z = ... and the output y for ReLU, Sigmoid, Tanh and Binary Step with
short notes. All functions receive the same number z, so the comparison is fair.
Change x, w, b and press the button again — at a negative z the differences are the
most noticeable.

12. "AUTOMATIC RECALCULATION"
What is this: an instant-recalculation switch.
Why: to explore quickly how the values affect the answer without pressing any buttons.
How to use it: tick the box — any change of x, a weight, the bias or the function
recalculates the neuron and redraws the diagram at once. Untick the box — the calculation
runs only when you press "Calculate".
What you will see: the result refreshes by itself; the "Calculate" button works in both modes.

13. THE "EXAMPLE" BUTTON
What is this: a ready-made learning example.
Why: to see the neuron work on verified numbers straight away.
How to use it: press "Example" — the program fills in x₁=2, w₁=0.5, x₂=3, w₂=0.2, all
other inputs and weights with zeros, bias=0.4, the ReLU function, labels the inputs
"Height" and "Weight", and runs the calculation itself.
What you will see: Σ = 1.6, z = 2.0, y = 2.0.

14. THE "RESET" BUTTON
What is this: a return to the empty initial form.
Why: to start the explanation or the experiment over.
How to use it: press "Reset".
What you will see: all x and w become 0, bias = 0, the input names are cleared, the
function is ReLU, the step-by-step mode returns to step 1, the diagram and the solution
window are cleared."""

TEXTS = {
    "ru": {
        # ---- окно, переключатель языка, вкладки ----
        "app_title": "Один искусственный нейрон — учебная программа",
        "lang_label": "Язык интерфейса",
        "lang_ru": "Русский",
        "lang_en": "English",
        "tab_solve": "Пошаговое решение",
        "tab_diagram": "Схема нейрона",
        "tab_contrib": "Вклады входов",
        "tab_compare": "Сравнение",
        # ---- подписи полей ----
        "inputs_box": "Входы (x), имена и веса (w)",
        "col_name": "Имя входа (необязательно)",
        "col_x": "Значение x",
        "col_w": "Вес w",
        "bias_field": "Смещение b:",
        "act_box": "Функция активации",
        # ---- кнопки и флажки ----
        "btn_calc": "Рассчитать",
        "btn_reset": "Сбросить",
        "btn_example": "Пример",
        "btn_compare": "Сравнить функции активации",
        "btn_next_step": "Следующий шаг",
        "btn_reset_steps": "Сбросить шаги",
        "btn_help": "Показать полную справку",
        "chk_auto": "Автоматический пересчёт",
        # ---- шапка справки и подсказка под схемой ----
        "guide_box": "Как пользоваться программой",
        "guide_head": (
            "Кратко: введите 10 значений x, к каждому — свой вес w, задайте смещение b,\n"
            "выберите функцию активации и нажмите «Рассчитать». Программа покажет все\n"
            "шаги: входы → умножение на веса → сумма → bias → активация → выход y."
        ),
        "diagram_help": (
            "Как читать схему (слева направо): кружки x₁...x₁₀ — входы; "
            "линии с «× w» — умножение каждого входа на его вес; "
            "большой кружок Σ — сложение всех произведений; «+ b» — "
            "прибавление смещения, получается z; зелёный блок f(z) — функция "
            "активации; кружок y — выход нейрона. Пройденные этапы "
            "окрашены оранжевым, непройденные — серым и голубым."
        ),
        # ---- пустые состояния и строка состояния ----
        "empty_result": "Заполните поля слева и нажмите «Рассчитать».",
        "empty_contrib": ("Нажмите «Рассчитать», затем откройте эту вкладку: здесь "
                          "показано, сколько каждый вход добавляет к сумме z."),
        "empty_compare": ("Нажмите «Сравнить функции активации», чтобы увидеть выходы "
                          "всех четырёх функций\nпри одном и том же значении z."),
        "status_ready": "Готово. Введите значения и нажмите «Рассчитать».",
        "status_calc": "Расчёт выполнен: z = {z}, y = {y}.",
        "status_step_end": "Пошаговый режим завершён. Нажмите «Сбросить шаги».",
        "status_steps_reset": "Шаги сброшены: готовы пройти расчёт с самого начала.",
        "status_compare": "Сравнение показано: все четыре функции получили одно и то же "
                          "z = {z}.",
        "status_reset": "Сброс выполнен: входы, веса, имена и смещение снова равны 0.",
        "status_lang": "Язык интерфейса переключён на русский.",
        "step_counter": "Шаг {n} из {total}",
        "step_all": "Все 6 шагов показаны",
        # ---- ошибки ввода ----
        "err_title": "ОШИБКА ВВОДА",
        "err_hint": "Введите числа (целые или дробные), например: 2, -1.5, 0.25",
        "err_not_number": "Поле «{label}» содержит нечисловое значение: {raw}",
        "err_bias_label": "смещение b",
        "err_status": "Ошибка: проверьте введённые значения.",
        # ---- названия и описания шести шагов ----
        "step1_title": "Входы",
        "step1_desc": "Нейрон получает десять чисел x₁ ... x₁₀ — это его входные сигналы.",
        "step2_title": "Умножение на веса",
        "step2_desc": "Каждый вход умножается на свой вес: x₁ × w₁, x₂ × w₂, ... Вес показывает силу "
                      "влияния входа.",
        "step3_title": "Суммирование",
        "step3_desc": "Все произведения складываются: Σ = x₁×w₁ + x₂×w₂ + ... + x₁₀×w₁₀.",
        "step4_title": "Смещение (bias)",
        "step4_desc": "К сумме прибавляется смещение: z = Σ + b. Оно сдвигает результат до функции активации.",
        "step5_title": "Функция активации",
        "step5_desc": "Выбранная функция превращает z в выход: y = f(z).",
        "step6_title": "Результат",
        "step6_desc": "y — это ответ нейрона: одно число, полученное из десяти входов.",
        # ---- описания функций активации (имена ReLU/Sigmoid/Tanh/Binary Step не меняются) ----
        "act_desc_ReLU": "возвращает 0 для отрицательных значений и само значение для положительных.",
        "act_desc_Sigmoid": "преобразует значение в диапазон от 0 до 1.",
        "act_desc_Tanh": "преобразует значение примерно в диапазон от -1 до 1.",
        "act_desc_Binary Step": "возвращает 0 или 1 в зависимости от знака значения.",
        # ---- заголовки и пояснения в окне решения ----
        "step_word": "ШАГ",
        "h1": "Входы (x) и веса (w)",
        "h2": "Вклад каждого входа (x × w)",
        "h3": "Взвешенная сумма",
        "h4": "Смещение (bias)",
        "h5": "Функция активации",
        "h6": "Результат",
        "th_input": "Вход",
        "th_x": "значение x",
        "th_w": "вес w",
        "th_product": "произведение",
        "s1_bias": "  Смещение b = {b}",
        "note_pos": "Положительный вклад увеличивает значение z.",
        "note_neg": "Отрицательный вклад уменьшает значение z.",
        "note_zero": "Нулевой вклад не изменяет значение z.",
        "note_weights": "Веса около 0 почти не пропускают сигнал, большие веса усиливают его.",
        "b_sum": "  Взвешенная сумма:     Σ = {s}",
        "b_bias": "  Смещение:             b = {b}",
        "b_before": "  Значение до функции активации:",
        "bias_expl": "  Bias (смещение) изменяет значение z перед применением функции активации:",
        "bias_pos": "  положительный — сдвигает z вправо (нейрон возбуждается легче),",
        "bias_neg": "  отрицательный — сдвигает z влево (нейрон возбуждается труднее),",
        "bias_zero": "  нулевой — ничего не меняет.",
        "f_line": "  f = {name} — {desc}",
        "r_z": "  Значение до функции активации:  z = {z}",
        "r_act": "  Функция активации:              {name}",
        "r_y": "  РЕЗУЛЬТАТ (выход нейрона):      y = {y}",
        "mode_header": "Пошаговый режим. Шаг {n} из {total}: {title}",
        "what_happens": "Что происходит на шаге {n}: {text}",
        "last_step": "Это последний шаг: нейрон посчитан целиком. «Сбросить шаги» — начать заново.",
        "press_next": "Нажмите «Следующий шаг», чтобы перейти дальше.",
        "reset_steps_msg": "Пошаговый режим сброшен. Нажмите «Следующий шаг», чтобы "
                           "увидеть шаг 1 — входы нейрона.",
        "reset_result_msg": "Все поля сброшены на значения по умолчанию (0).",
        "reset_contrib_msg": "Нажмите «Рассчитать», чтобы увидеть вклад каждого входа.",
        # ---- сравнение функций активации ----
        "cmp_title": "Сравнение функций активации при одном и том же z = {z}",
        "cmp_intro1": "Все функции получают одно и то же число z, поэтому сравнивать",
        "cmp_intro2": "результаты корректно: отличается только способ превращения z в выход y.",
        "cmp_th_func": "Функция",
        "cmp_note1": "Положительный z разные функции сжимают или усиливают по-разному,",
        "cmp_note2": "а отрицательный z показывает их различия ярче всего.",
        "cmp_again": "Меняйте x, w и b и нажимайте «Сравнить функции активации» снова.",
        # ---- вкладка «Вклады входов» ----
        "contrib_title": "ВКЛАД КАЖДОГО ВХОДА В СУММУ z",
        "contrib_how1": "  Вклад = x × w. Полоска вправо — вклад положительный (увеличивает z),",
        "contrib_how2": "  влево — отрицательный (уменьшает z), пустая строка — вклад нулевой.",
        "contrib_th_product": "произведение x × w",
        "contrib_th_bar": "диаграмма вклада",
        "contrib_sum": "  Сумма всех вкладов: Σ = {s}",
        "contrib_bias": "  Со смещением b:     z = Σ + b = {z}",
        "contrib_exp1": "Поэкспериментируйте: поменяйте один вес с 0.5 на -0.5 — полоска",
        "contrib_exp2": "развернётся влево, а значение z уменьшится.",
        # ---- схема нейрона ----
        "dg_activation": "активация",
        "caps": [
            "Нажмите «Рассчитать» или «Следующий шаг» — увидите путь данных.",
            "Шаг 1 — входы: нейрон получает числа x₁ ... x₁₀.",
            "Шаг 2 — умножение: каждый вход умножается на свой вес w.",
            "Шаг 3 — суммирование: произведения складываются в Σ.",
            "Шаг 4 — смещение: к Σ прибавляется bias, получается z.",
            "Шаг 5 — активация: функция f превращает z в y.",
            "Шаг 6 — результат: y — ответ нейрона (одно число).",
        ],
        # ---- учебный пример ----
        "example_name1": "Рост",
        "example_name2": "Вес",
        "guide": GUIDE_RU,
    },
    "en": {
        # ---- window, language switch, tabs ----
        "app_title": "One artificial neuron — a learning program",
        "lang_label": "Interface language",
        "lang_ru": "Русский",
        "lang_en": "English",
        "tab_solve": "Step-by-step solution",
        "tab_diagram": "Neuron diagram",
        "tab_contrib": "Input contributions",
        "tab_compare": "Comparison",
        # ---- field labels ----
        "inputs_box": "Inputs (x), names and weights (w)",
        "col_name": "Input name (optional)",
        "col_x": "Value x",
        "col_w": "Weight w",
        "bias_field": "Bias b:",
        "act_box": "Activation function",
        # ---- buttons and the check box ----
        "btn_calc": "Calculate",
        "btn_reset": "Reset",
        "btn_example": "Example",
        "btn_compare": "Compare activation functions",
        "btn_next_step": "Next step",
        "btn_reset_steps": "Reset steps",
        "btn_help": "Show the full guide",
        "chk_auto": "Automatic recalculation",
        # ---- guide header and the hint under the diagram ----
        "guide_box": "How to use this program",
        "guide_head": (
            "In short: enter 10 values of x, give each one its weight w, set the bias b,\n"
            "choose an activation function and press \"Calculate\". The program shows all\n"
            "steps: inputs → multiplying by the weights → sum → bias → activation → output y."
        ),
        "diagram_help": (
            "How to read the diagram (left to right): the circles x₁...x₁₀ are the inputs; "
            "the lines marked \"× w\" multiply each input by its weight; "
            "the big circle Σ adds all the products; \"+ b\" adds the bias, "
            "which gives z; the green block f(z) is the activation function; "
            "the circle y is the neuron output. Stages you have passed are "
            "coloured orange, the ones you have not stay grey and blue."
        ),
        # ---- empty states and the status line ----
        "empty_result": "Fill in the fields on the left and press \"Calculate\".",
        "empty_contrib": ("Press \"Calculate\", then open this tab: it shows how much "
                          "each input adds to the sum z."),
        "empty_compare": ("Press \"Compare activation functions\" to see the outputs of "
                          "all four functions\nfor the very same value of z."),
        "status_ready": "Ready. Enter the values and press \"Calculate\".",
        "status_calc": "Calculation finished: z = {z}, y = {y}.",
        "status_step_end": "The step-by-step mode is complete. Press \"Reset steps\".",
        "status_steps_reset": "Steps reset: ready to walk through the calculation from the start.",
        "status_compare": "Comparison shown: all four functions received the same z = {z}.",
        "status_reset": "Reset done: the inputs, weights, names and bias are 0 again.",
        "status_lang": "The interface language has been switched to English.",
        "step_counter": "Step {n} of {total}",
        "step_all": "All 6 steps shown",
        # ---- input errors ----
        "err_title": "INPUT ERROR",
        "err_hint": "Enter numbers (whole or fractional), for example: 2, -1.5, 0.25",
        "err_not_number": "The field \"{label}\" contains a non-numeric value: {raw}",
        "err_bias_label": "bias b",
        "err_status": "Error: check the values you entered.",
        # ---- titles and notes of the six steps ----
        "step1_title": "Inputs",
        "step1_desc": "The neuron receives ten numbers x₁ ... x₁₀ — these are its input signals.",
        "step2_title": "Multiplying by the weights",
        "step2_desc": "Each input is multiplied by its own weight: x₁ × w₁, x₂ × w₂, ... The weight "
                      "shows how strongly the input matters.",
        "step3_title": "Summing",
        "step3_desc": "All the products are added up: Σ = x₁×w₁ + x₂×w₂ + ... + x₁₀×w₁₀.",
        "step4_title": "Bias",
        "step4_desc": "The bias is added to the sum: z = Σ + b. It shifts the result before the activation function.",
        "step5_title": "Activation function",
        "step5_desc": "The chosen function turns z into the output: y = f(z).",
        "step6_title": "Result",
        "step6_desc": "y is the neuron's answer: one number obtained from ten inputs.",
        # ---- activation function descriptions (the names ReLU/Sigmoid/Tanh/Binary Step stay) ----
        "act_desc_ReLU": "returns 0 for negative values and the value itself for positive ones.",
        "act_desc_Sigmoid": "turns the value into the range from 0 to 1.",
        "act_desc_Tanh": "turns the value into roughly the range from -1 to 1.",
        "act_desc_Binary Step": "returns 0 or 1 depending on the sign of the value.",
        # ---- headers and notes in the solution window ----
        "step_word": "STEP",
        "h1": "Inputs (x) and weights (w)",
        "h2": "Contribution of each input (x × w)",
        "h3": "Weighted sum",
        "h4": "Bias",
        "h5": "Activation function",
        "h6": "Result",
        "th_input": "Input",
        "th_x": "value x",
        "th_w": "weight w",
        "th_product": "product",
        "s1_bias": "  Bias b = {b}",
        "note_pos": "A positive contribution increases the value z.",
        "note_neg": "A negative contribution decreases the value z.",
        "note_zero": "A zero contribution does not change the value z.",
        "note_weights": "Weights near 0 barely pass the signal; large weights amplify it.",
        "b_sum": "  Weighted sum:        Σ = {s}",
        "b_bias": "  Bias:                b = {b}",
        "b_before": "  Value before the activation function:",
        "bias_expl": "  The bias changes z before the activation function is applied:",
        "bias_pos": "  a positive one shifts z to the right (the neuron fires more easily),",
        "bias_neg": "  a negative one shifts z to the left (the neuron fires with difficulty),",
        "bias_zero": "  a zero one changes nothing.",
        "f_line": "  f = {name} — {desc}",
        "r_z": "  Value before the activation function:  z = {z}",
        "r_act": "  Activation function:                   {name}",
        "r_y": "  RESULT (the neuron output):              y = {y}",
        "mode_header": "Step-by-step mode. Step {n} of {total}: {title}",
        "what_happens": "What happens at step {n}: {text}",
        "last_step": "This is the last step: the neuron is fully calculated. \"Reset steps\" starts over.",
        "press_next": "Press \"Next step\" to go further.",
        "reset_steps_msg": "The step-by-step mode is reset. Press \"Next step\" to see "
                           "step 1 — the inputs of the neuron.",
        "reset_result_msg": "All fields are back to their default values (0).",
        "reset_contrib_msg": "Press \"Calculate\" to see the contribution of each input.",
        # ---- comparing the activation functions ----
        "cmp_title": "Comparing the activation functions at the very same z = {z}",
        "cmp_intro1": "Every function receives the same number z, so comparing the",
        "cmp_intro2": "results is fair: only the way z becomes the output y differs.",
        "cmp_th_func": "Function",
        "cmp_note1": "For a positive z the functions shrink or amplify it in different ways,",
        "cmp_note2": "while a negative z shows their differences most clearly.",
        "cmp_again": "Change x, w and b and press \"Compare activation functions\" again.",
        # ---- the \"Input contributions\" tab ----
        "contrib_title": "THE CONTRIBUTION OF EACH INPUT TO THE SUM z",
        "contrib_how1": "  A contribution = x × w. A bar to the right — a positive contribution (raises z),",
        "contrib_how2": "  to the left — a negative one (lowers z), an empty row — a zero contribution.",
        "contrib_th_product": "product x × w",
        "contrib_th_bar": "contribution bar",
        "contrib_sum": "  Sum of all contributions: Σ = {s}",
        "contrib_bias": "  With the bias b:          z = Σ + b = {z}",
        "contrib_exp1": "Experiment: change one weight from 0.5 to -0.5 — the bar turns",
        "contrib_exp2": "to the left and the value z becomes smaller.",
        # ---- the neuron diagram ----
        "dg_activation": "activation",
        "caps": [
            "Press \"Calculate\" or \"Next step\" to see the path of the data.",
            "Step 1 — inputs: the neuron gets the numbers x₁ ... x₁₀.",
            "Step 2 — multiplication: each input is multiplied by its weight w.",
            "Step 3 — summing: the products are added up into Σ.",
            "Step 4 — bias: b is added to Σ, which gives z.",
            "Step 5 — activation: the function f turns z into y.",
            "Step 6 — result: y is the neuron's answer (one number).",
        ],
        # ---- the learning example ----
        "example_name1": "Height",
        "example_name2": "Weight",
        "guide": GUIDE_EN,
    },
}


def T(key, **kw):
    """Текст текущего языка по ключу; подстановки {name} — только если они переданы."""
    text = TEXTS[LANG][key]
    return text.format(**kw) if kw else text


def set_language(code):
    """Сменить язык интерфейса (обновление виджетов делает NeuronApp.retranslate)."""
    global LANG
    LANG = code


def guide_text():
    """Полная справка на текущем языке (14 разделов)."""
    return TEXTS[LANG]["guide"]


# ---------------------------------------------------------------
# Активационные функции (шаг 6: применение функции активации)
# ---------------------------------------------------------------
def relu(z):
    return max(0.0, z)

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def tanh_fn(z):
    return math.tanh(z)

def binary_step(z):
    return 1.0 if z >= 0 else 0.0

ACTIVATIONS = {
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh_fn,
    "Binary Step": binary_step,
}

# Названия шести шагов расчёта: ключи TEXTS для заголовка и пояснения
# (используются и в пошаговом режиме, и в подсветке схемы нейрона).
STEPS = [
    ("step1_title", "step1_desc"),
    ("step2_title", "step2_desc"),
    ("step3_title", "step3_desc"),
    ("step4_title", "step4_desc"),
    ("step5_title", "step5_desc"),
    ("step6_title", "step6_desc"),
]


def act_desc(name):
    """Пояснение функции активации на текущем языке."""
    return T(f"act_desc_{name}")


def step_title(k):
    """Название шага k (1..6) на текущем языке."""
    return T(STEPS[k - 1][0])


def step_note(k):
    """Пояснение «что происходит на шаге k» на текущем языке."""
    return T(STEPS[k - 1][1])


# ---------------------------------------------------------------
# Форматирование чисел
# ---------------------------------------------------------------
def fmt(v):
    """Красивое отображение числа: 2.0 -> 2, 0.6 -> 0.6."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def fmt_result(v):
    """Отображение результатов шагов: всегда с точкой, 2 -> 2.0."""
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    return f"{v:.6g}"


def fmt_sign(v):
    """То же, но со знаком: +2.0 или -3.0 (чтобы вклад был наглядным)."""
    s = fmt_result(abs(v))
    if v < 0:
        return "-" + s
    return "+" + s


def sub(i):
    """Индекс с нижним подписным символом (1 -> ₁)."""
    digits = "₀₁₂₃₄₅₆₇₈₉"
    s = ""
    for ch in str(i):
        s += digits[int(ch)]
    return s


def input_label(i, names):
    """Название входа: «Рост (x₁)», а если имени нет — просто «x₁»."""
    name = names[i - 1].strip() if i - 1 < len(names) else ""
    return f"{name} (x{sub(i)})" if name else f"x{sub(i)}"


# ---------------------------------------------------------------
# Расчёт нейрона по шагам (этапы 1-7)
# ---------------------------------------------------------------
def compute_steps(xs, ws, bias, act_name, step, names=None):
    """Возвращает текст расчёта НЕЙРОНА, начиная с STEP (1..6).

    xs   — 10 входных значений x₁..x₁₀          (этап 1: входы)
    ws   — 10 соответствующих весов w₁..w₁₀     (этап 2: веса)
    bias — смещение b                            (этап 4: сумма + bias)
    act_name — выбранная функция активации       (этап 5: активация)
    step — 1..6: показать только этот шаг; 0 — все шаги сразу
    names — необязательные имена входов
    """
    names = names or [""] * N_INPUTS

    # Этап 3: умножение каждого входа на его вес (x × w)
    products = [x * w for x, w in zip(xs, ws)]
    # Этап 4: взвешенная сумма Σ и прибавление смещения
    weighted_sum = sum(products)
    z = weighted_sum + bias
    # Этап 5-6: функция активации и выход нейрона y
    fn = ACTIVATIONS[act_name]
    y = fn(z)

    sw = T("step_word")          # «ШАГ» / «STEP»

    # Служебные блоки текста (шаг -> строки)
    blocks = {}

    blocks[1] = [f"{sw} 1. {T('h1')}", "",
                 f"  {T('th_input'):<22} {T('th_x'):>11}   {T('th_w'):>9}", ""]
    for i, (x, w) in enumerate(zip(xs, ws), start=1):
        blocks[1].append(f"  {input_label(i, names):<22} {fmt(x):>11}   {fmt(w):>9}")
    blocks[1] += ["", T("s1_bias", b=fmt(bias))]

    blocks[2] = [f"{sw} 2. {T('h2')}", "",
                 f"  {T('th_input'):<22}{T('th_product'):>18}", ""]
    for i, (x, w, p) in enumerate(zip(xs, ws, products), start=1):
        blocks[2].append(f"  {input_label(i, names):<22} {fmt(x):>6} × {fmt(w):>6} = "
                         f"{fmt_result(p):>7}")
    blocks[2] += ["",
                  "  " + T("note_pos"),
                  "  " + T("note_neg"),
                  "  " + T("note_zero"),
                  "  " + T("note_weights")]

    blocks[3] = [f"{sw} 3. {T('h3')}", "",
                 f"  Σ = x₁×w₁ + x₂×w₂ + ... + x₁₀×w₁₀ = {fmt_result(weighted_sum)}"]

    blocks[4] = [f"{sw} 4. {T('h4')}", "",
                 T("b_sum", s=fmt_result(weighted_sum)),
                 T("b_bias", b=fmt_result(bias)),
                 T("b_before"),
                 f"  z = Σ + b = {fmt_result(weighted_sum)} + {fmt(bias)} = {fmt_result(z)}",
                 "",
                 T("bias_expl"),
                 T("bias_pos"),
                 T("bias_neg"),
                 T("bias_zero")]

    blocks[5] = [f"{sw} 5. {T('h5')}", "",
                 T("f_line", name=act_name, desc=act_desc(act_name)),
                 f"  y = f({fmt_result(z)}) = {fmt_result(y)}"]

    blocks[6] = [f"{sw} 6. {T('h6')}", "",
                 T("r_z", z=fmt_result(z)),
                 T("r_act", name=act_name),
                 T("r_y", y=fmt_result(y))]

    if step == 0:
        lines = []
        for s in range(1, TOTAL_STEPS + 1):
            lines += blocks[s] + [""]
        return "\n".join(lines).rstrip()

    # Пошаговый режим: показываем шаги 1..step накопительно, плюс пояснение к последнему
    lines = [T("mode_header", n=step, total=TOTAL_STEPS, title=step_title(step)), ""]
    for s in range(1, step + 1):
        lines += blocks[s] + [""]
    lines += [T("what_happens", n=step, text=step_note(step)), ""]
    lines += [T("last_step") if step == TOTAL_STEPS else T("press_next")]
    return "\n".join(lines).rstrip()


# Сохраняем исходное имя функции: вся математика работает как раньше.
def compute_neuron(xs, ws, bias, act_name, names=None):
    """Полный расчёт одного нейрона за раз (все 6 шагов)."""
    return compute_steps(xs, ws, bias, act_name, 0, names)


def compare_activation_text(z):
    """Сравнение всех четырёх функций активации при ОДНОМ и том же z."""
    lines = [T("cmp_title", z=fmt_result(z)), "",
             T("cmp_intro1"),
             T("cmp_intro2"),
             "", f"  {T('cmp_th_func'):<13}{'y = f(z)':>11}", f"  {'-' * 25}"]
    for name, fn in ACTIVATIONS.items():
        lines.append(f"  {name:<13}{fmt_result(fn(z)):>11}")
        # длинное пояснение переносим, чтобы таблица не «расползалась»
        wrapped = textwrap.wrap(f"{name} — {act_desc(name)}", width=52)
        for k, part in enumerate(wrapped):
            lines.append(f"  {'':<13}{part}" if k == 0 else f"  {'':<14}{part}")
        lines.append("")
    lines += [T("cmp_note1"),
              T("cmp_note2"),
              "", T("cmp_again")]
    return "\n".join(lines)


# ---------------------------------------------------------------
# Схема нейрона (этап 7: визуализация на tkinter Canvas)
# ---------------------------------------------------------------
class Diagram:
    """Простая схема: входы → веса → Σ → +bias → f(z) → y.

    Каждый элемент схемы регистрируется в self.stages под номером шага (1..6),
    поэтому подсветка — это просто перекраска элементов пройденных шагов.
    """

    ACTIVE = "#e8761a"
    ACTIVE_EDGE = "#8a4300"
    DIM = "#a6a6a6"

    def __init__(self, parent, width=620, height=530):
        self.canvas = tk.Canvas(parent, width=width, height=height,
                                highlightthickness=0, bg="white")
        self.stages = {s: [] for s in range(1, TOTAL_STEPS + 1)}
        self.shown_step = 0      # последний подсвеченный шаг (для подписи под схемой)
        self._build()

    def _add(self, stage, item):
        """Запомнить элемент схемы под номером шага для последующей подсветки."""
        self.stages[stage].append(item)
        return item

    def _build(self):
        c = self.canvas
        x_col, sym_col, sum_cx, act_cx, out_cx = 92, 180, 300, 430, 548
        ys = [64 + i * 48 for i in range(N_INPUTS)]
        mid_y = 280

        for i, y in enumerate(ys, start=1):
            # ШАГ 1: вход
            self._add(1, c.create_oval(x_col - 22, y - 15, x_col + 22, y + 15,
                                       fill="#dbeafe", outline="#31597a"))
            self._add(1, c.create_text(x_col, y, text=f"x{sub(i)}",
                                       font=("TkDefaultFont", 16), fill="#31597a"))
            # ШАГ 2: умножение на вес (линия + подпись «× wᵢ»)
            self._add(2, c.create_line(x_col + 22, y, sym_col + 40, y,
                                       fill=self.DIM, arrow="last", width=2))
            self._add(2, c.create_text(sym_col, y - 12, text="×",
                                       font=("TkDefaultFont", 16, "bold"), fill="#333"))
            self._add(2, c.create_text(sym_col + 12, y - 12, text=f"w{sub(i)}",
                                       font=("TkDefaultFont", 16), fill="#333",
                                       anchor="w"))
            self._add(2, c.create_line(sym_col + 40, y, sum_cx - 36, mid_y,
                                       fill=self.DIM, width=2))

        # ШАГ 3: суммирование
        self._add(3, c.create_oval(sum_cx - 36, mid_y - 36, sum_cx + 36, mid_y + 36,
                                   fill="#fdf1c7", outline="#8a6d1b"))
        self._add(3, c.create_text(sum_cx, mid_y - 12, text="Σ",
                                   font=("TkDefaultFont", 26, "bold"), fill="#8a6d1b"))
        # ШАГ 4: смещение (bias)
        self._add(4, c.create_text(sum_cx, mid_y + 20, text="+ b",
                                   font=("TkDefaultFont", 16), fill="#8a6d1b"))
        self._add(4, c.create_line(sum_cx + 36, mid_y, act_cx - 62, mid_y,
                                   fill=self.DIM, arrow="last", width=2))
        self._add(4, c.create_text(sum_cx + 66, mid_y - 40, text="z = Σ + b",
                                   font=("TkDefaultFont", 14), fill="#555"))
        # ШАГ 5: функция активации
        self._add(5, c.create_rectangle(act_cx - 62, mid_y - 30, act_cx + 62, mid_y + 30,
                                        fill="#e4f4e4", outline="#1a5c2a"))
        self._add(5, c.create_text(act_cx, mid_y - 12, text="f(z)",
                                   font=("TkDefaultFont", 22, "bold"), fill="#1a5c2a"))
        self.act_lbl_item = self._add(
            5, c.create_text(act_cx, mid_y + 16, text=T("dg_activation"),
                             font=("TkDefaultFont", 13), fill="#1a5c2a"))
        self._add(5, c.create_line(act_cx + 62, mid_y, out_cx - 30, mid_y,
                                   fill=self.DIM, arrow="last", width=2))
        # ШАГ 6: выход
        self._add(6, c.create_oval(out_cx - 30, mid_y - 30, out_cx + 30, mid_y + 30,
                                   fill="#f6dde3", outline="#8a1b3a"))
        self._add(6, c.create_text(out_cx, mid_y, text="y",
                                   font=("TkDefaultFont", 22, "bold"), fill="#8a1b3a"))

        # Колонка текущих чисел слева от входов — заполняется после расчёта
        self.val_items = [c.create_text(x_col - 30, y, text="", anchor="e",
                                        font=("TkDefaultFont", 14), fill="#31597a")
                          for y in ys]
        self.z_val = c.create_text(sum_cx, mid_y - 66, text="",
                                   font=("TkDefaultFont", 20, "bold"), fill="#8a6d1b")
        self.y_val = c.create_text(out_cx, mid_y + 54, text="",
                                   font=("TkDefaultFont", 20, "bold"), fill="#8a1b3a")
        self.cap = c.create_text(6, 16, anchor="w", text=T("caps")[0],
                                 font=("TkDefaultFont", 16), fill="#333")

    def refresh_language(self):
        """Перевести текстовые подписи схемы на текущий язык."""
        self.canvas.itemconfigure(self.act_lbl_item, text=T("dg_activation"))
        self.canvas.itemconfigure(self.cap, text=T("caps")[self.shown_step])

    def set_values(self, xs, products, z, y):
        """Показать реальные числа после расчёта."""
        for item, x in zip(self.val_items, xs):
            self.canvas.itemconfigure(item, text=f"{fmt(x)}")
        self.canvas.itemconfigure(self.z_val, text=f"z = {fmt_result(z)}")
        self.canvas.itemconfigure(self.y_val, text=f"y = {fmt_result(y)}")

    def clear_values(self):
        for item in self.val_items + [self.z_val, self.y_val]:
            self.canvas.itemconfigure(item, text="")

    # базовые цвета фигур: шаг расчёта -> цвет заливки
    NODE_FILL = {1: "#dbeafe", 3: "#fdf1c7", 5: "#e4f4e4", 6: "#f6dde3"}
    # цвет подсвеченной фигуры (мягкий оранжевый, чтобы схема не «кричала»)
    NODE_ACTIVE = "#ffd8a8"

    def highlight(self, up_to):
        """Подсветить шаги 1..up_to (0 — снять подсветку)."""
        self.shown_step = up_to
        for s, items in self.stages.items():
            active = s <= up_to
            for item in items:
                kind = self.canvas.type(item)
                if kind == "line":
                    self.canvas.itemconfigure(
                        item, fill=self.ACTIVE if active else self.DIM,
                        width=4 if active else 2)
                elif kind in ("oval", "rectangle"):
                    self.canvas.itemconfigure(
                        item, fill=self.NODE_ACTIVE if active else self.NODE_FILL.get(s, "white"),
                        outline=self.ACTIVE_EDGE if active else "#777")
                else:  # text
                    self.canvas.itemconfigure(
                        item, fill=self.ACTIVE_EDGE if active else "#333")
        self.canvas.itemconfigure(self.cap, text=T("caps")[up_to])



# ---------------------------------------------------------------
# Интерфейс (все надписи берутся из TEXTS; русский по умолчанию)
# ---------------------------------------------------------------
class NeuronApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(T("app_title"))
        self.geometry("1320x1000")
        self.minsize(1240, 820)

        # ---- 1-2. Входы (x) и веса (w): значения и имена ----
        self.var_x = [tk.StringVar(value="0") for _ in range(N_INPUTS)]
        self.var_w = [tk.StringVar(value="0") for _ in range(N_INPUTS)]
        self.var_name = [tk.StringVar(value="") for _ in range(N_INPUTS)]
        # ---- 3. Смещение ----
        self.var_bias = tk.StringVar(value="0")
        # ---- 5-6. Функция активации ----
        self.var_act = tk.StringVar(value="ReLU")
        # ---- 8. Пошаговый режим: просто целое число 0..6 ----
        self.var_step = tk.IntVar(value=0)
        # ---- 12. Автоматический пересчёт ----
        self.var_auto = tk.BooleanVar(value=False)
        # ---- Язык интерфейса ----
        self.var_lang = tk.StringVar(value=LANG)
        self._silent = False   # выключает автопересчёт на время программного заполнения

        # что сейчас показано на вкладках, чтобы перевести это без перезапуска
        self._text_refs = []           # (виджет, ключ TEXTS) — статические надписи
        self._pages = {}               # ключ вкладки -> сам фрейм вкладки
        self._snap = None              # данные последнего расчёта (для смены языка)
        self._err = None               # (поле, номер, сырая строка) последней ошибки
        self._result_kind = ("empty",)
        self._contrib_kind = ("empty",)
        self._compare_kind = ("empty",)
        self._step_lbl = ("counter", 0)
        self._example_named = {}       # номер входа -> ключ TEXTS с его именем из «Примера»

        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 8))

        self._build_language(left)
        self._build_inputs(left)
        self._build_activation(left)
        self._build_buttons(left)
        self._build_guide(left)

        self._build_right(root)

        self.var_status = tk.StringVar()
        ttk.Label(root, textvariable=self.var_status, anchor="w",
                  foreground="#555").pack(fill="x", side="bottom")
        # имя автора не переводится: сознательно не заведено в TEXTS/_text_refs
        ttk.Label(root, text="SteveWorks001", foreground="#555").pack(
            side="bottom", anchor="e")
        self.set_status("status_ready")

        # ---- 12. автоматический пересчёт при изменении полей ----
        for var in self.var_x + self.var_w + self.var_name:
            var.trace_add("write", lambda *_: self._maybe_auto())

    # ---------- локализация ----------
    def _reg(self, widget, key):
        """Надпись виджета берётся из TEXTS и будет переводиться при смене языка."""
        widget.configure(text=T(key))
        self._text_refs.append((widget, key))
        return widget

    def _build_language(self, parent):
        """Переключатель языка: интерфейс меняется сразу, без перезапуска."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 6))
        self._reg(ttk.Label(frame, text=""), "lang_label").pack(side="left")
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            ttk.Radiobutton(frame, text=T(key), value=code, variable=self.var_lang,
                            command=self.switch_language).pack(side="left", padx=(8, 0))

    def switch_language(self):
        code = self.var_lang.get()
        if code == LANG:
            return
        old_names = {i: self.var_name[i].get() for i in self._example_named}
        old_texts = {i: TEXTS[LANG][key] for i, key in self._example_named.items()}
        set_language(code)
        # имена входов из учебного примера переводим, чтобы не было смеси языков
        for i, key in self._example_named.items():
            if old_names[i] == old_texts[i]:
                self.var_name[i].set(T(key))
                if self._snap is not None:
                    self._snap[2][i] = T(key)
        self.retranslate()
        self.set_status("status_lang")

    def retranslate(self):
        """Перевести статические надписи и заново напечатать динамические панели."""
        self.title(T("app_title"))
        for widget, key in self._text_refs:
            widget.configure(text=T(key))
        for key, page in self._pages.items():
            self.nb.tab(page, text=T(key))
        self.update_act_desc()
        self.render_step_lbl()
        self.diagram.refresh_language()
        self.render_result()
        self.render_contrib()
        self.render_compare()
        if self.help_shown:
            self.help_lbl.configure(state="normal")
            self.help_lbl.delete("1.0", "end")
            self.help_lbl.insert("1.0", guide_text())
            self.help_lbl.configure(state="disabled")

    # ---------- построение интерфейса ----------
    def _build_inputs(self, parent):
        """Поля: имя входа, значение x, вес w и смещение b."""
        frame = ttk.LabelFrame(parent, padding=8)
        self._reg(frame, "inputs_box")
        frame.pack(fill="x")

        for col, key in ((1, "col_name"), (2, "col_x"), (3, "col_w")):
            self._reg(ttk.Label(frame, font=tkfont.nametofont("TkHeadingFont")),
                      key).grid(row=0, column=col, padx=4, sticky="w")

        for i in range(1, N_INPUTS + 1):
            k = i - 1
            ttk.Label(frame, text=f"x{sub(i)}:").grid(row=i, column=0, sticky="e",
                                                      padx=(0, 6), pady=1)
            ttk.Entry(frame, width=15, textvariable=self.var_name[k]).grid(
                row=i, column=1, pady=1)
            ttk.Entry(frame, width=9, textvariable=self.var_x[k]).grid(
                row=i, column=2, pady=1)
            ttk.Entry(frame, width=9, textvariable=self.var_w[k]).grid(
                row=i, column=3, pady=1)

        row = N_INPUTS + 1
        self._reg(ttk.Label(frame), "bias_field").grid(row=row, column=0, columnspan=3,
                                                      sticky="e", pady=(8, 1))
        ttk.Entry(frame, width=9, textvariable=self.var_bias).grid(row=row, column=3,
                                                                   pady=(8, 1))
        self.var_bias.trace_add("write", lambda *_: self._maybe_auto())

    def _build_activation(self, parent):
        act_frame = ttk.LabelFrame(parent, padding=8)
        self._reg(act_frame, "act_box")
        act_frame.pack(fill="x", pady=(8, 0))
        radios = ttk.Frame(act_frame)
        radios.pack(anchor="w")
        for name in ACTIVATIONS:
            ttk.Radiobutton(radios, text=name, value=name,
                            variable=self.var_act).pack(side="left", padx=(0, 10))
        self.act_desc = ttk.Label(act_frame, text="", wraplength=430, foreground="#444")
        self.act_desc.pack(anchor="w", pady=(4, 0))
        self.var_act.trace_add("write", lambda *_: self._activation_changed())
        self.update_act_desc()

    def _build_buttons(self, parent):
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))
        self._reg(ttk.Button(btns, command=self.calculate), "btn_calc").pack(side="left")
        self._reg(ttk.Button(btns, command=self.reset), "btn_reset").pack(
            side="left", padx=6)
        self._reg(ttk.Button(btns, command=self.load_example), "btn_example").pack(
            side="left")
        self._reg(ttk.Button(btns, command=self.compare), "btn_compare").pack(
            side="left", padx=(14, 0))

        steps = ttk.Frame(parent)
        steps.pack(fill="x", pady=(6, 0))
        self._reg(ttk.Button(steps, command=self.next_step), "btn_next_step").pack(
            side="left")
        self._reg(ttk.Button(steps, command=self.reset_steps), "btn_reset_steps").pack(
            side="left", padx=6)
        self.var_step_lbl = tk.StringVar()
        ttk.Label(steps, textvariable=self.var_step_lbl, foreground="#555").pack(
            side="left", padx=6)
        self._reg(ttk.Checkbutton(steps, variable=self.var_auto), "chk_auto").pack(
            side="left", padx=(14, 0))
        self.render_step_lbl()

    def _build_guide(self, parent):
        """Сворачиваемая справка «Как пользоваться программой» с прокруткой."""
        help_frame = ttk.LabelFrame(parent, padding=6)
        self._reg(help_frame, "guide_box")
        help_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.help_head = ttk.Label(help_frame, wraplength=440,
                                   foreground="#1a5c2a", justify="left")
        self._reg(self.help_head, "guide_head").pack(anchor="w")
        self._reg(ttk.Button(help_frame, command=self.toggle_help),
                  "btn_help").pack(anchor="w", pady=(4, 4))

        box = ttk.Frame(help_frame)
        box.pack(fill="both", expand=True)
        ui_font = tkfont.nametofont("TkDefaultFont").copy()
        ui_font.configure(size=14)
        self.help_lbl = tk.Text(box, width=55, height=8, wrap="word", font=ui_font,
                                bg="#fbfbf7", relief="flat")
        scroll = ttk.Scrollbar(box, command=self.help_lbl.yview)
        self.help_lbl.configure(yscrollcommand=scroll.set, state="disabled")
        self.help_lbl.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.help_shown = False

    def _build_right(self, root):
        """Вкладки: решение, схема нейрона, вклады входов, сравнение функций."""
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.nb = nb

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=15)

        solve_page = ttk.Frame(nb)
        nb.add(solve_page)
        self._pages["tab_solve"] = solve_page
        self.result = tk.Text(solve_page, wrap="word", font=mono, state="disabled",
                              bg="#fbfbf7")
        self.result.pack(side="left", fill="both", expand=True)
        self.result_scroll = ttk.Scrollbar(solve_page, command=self.result.yview)
        self.result_scroll.pack(side="right", fill="y")
        self.result.configure(yscrollcommand=self.result_scroll.set)
        self.render_result()

        diagram_page = ttk.Frame(nb)
        nb.add(diagram_page)
        self._pages["tab_diagram"] = diagram_page
        self.diagram = Diagram(diagram_page)
        self.diagram.canvas.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        self._reg(ttk.Label(diagram_page, wraplength=640, justify="left",
                            foreground="#444"), "diagram_help").pack(
            anchor="w", pady=6)

        contrib_page = ttk.Frame(nb)
        nb.add(contrib_page)
        self._pages["tab_contrib"] = contrib_page
        self.contrib = tk.Text(contrib_page, wrap="none", font=mono, bg="#fbfbf7")
        contrib_v = ttk.Scrollbar(contrib_page, command=self.contrib.yview)
        contrib_h = ttk.Scrollbar(contrib_page, orient="horizontal",
                                  command=self.contrib.xview)
        self.contrib.configure(yscrollcommand=contrib_v.set, xscrollcommand=contrib_h.set)
        self.contrib.grid(row=0, column=0, sticky="nsew")
        contrib_v.grid(row=0, column=1, sticky="ns")
        contrib_h.grid(row=1, column=0, sticky="ew")
        contrib_page.rowconfigure(0, weight=1)
        contrib_page.columnconfigure(0, weight=1)
        self.render_contrib()

        compare_page = ttk.Frame(nb)
        nb.add(compare_page)
        self._pages["tab_compare"] = compare_page
        self.compare_txt = tk.Text(compare_page, wrap="word", font=mono, bg="#fbfbf7")
        self.compare_txt.pack(side="left", fill="both", expand=True)
        compare_v = ttk.Scrollbar(compare_page, command=self.compare_txt.yview)
        compare_v.pack(side="right", fill="y")
        self.compare_txt.configure(yscrollcommand=compare_v.set)
        self.render_compare()
        for key, page in self._pages.items():
            nb.tab(page, text=T(key))

    # ---------- вспомогательные методы ----------
    def update_act_desc(self):
        name = self.var_act.get()
        self.act_desc.configure(text=f"{name} — {act_desc(name)}")

    def _activation_changed(self):
        self.update_act_desc()
        self._maybe_auto()

    def _maybe_auto(self):
        """Если включён «Автоматический пересчёт» — пересчитать сразу после правки."""
        if self.var_auto.get() and not self._silent:
            self.calculate()

    def set_status(self, key, **kw):
        """Строка состояния: текст берётся из TEXTS по ключу."""
        self.var_status.set(T(key, **kw))

    def field_label(self, kind, i):
        """Человеческое имя поля для сообщения об ошибке (x₁, w₃ или «смещение b»)."""
        if kind == "bias":
            return T("err_bias_label")
        return f"{kind}{sub(i + 1)}"

    def read_number(self, var, kind, i=None):
        raw = var.get().strip().replace(",", ".")
        if raw == "":
            raw = "0"
        try:
            return float(raw), None
        except ValueError:
            return None, (kind, i, raw)

    def parse_all(self):
        """Этапы 1-3: достать из полей входы x, веса w, имена и смещение b."""
        xs, ws, names = [], [], []
        for i in range(N_INPUTS):
            x, err = self.read_number(self.var_x[i], "x", i)
            if err:
                return None, None, None, None, err
            xs.append(x)
            w, err = self.read_number(self.var_w[i], "w", i)
            if err:
                return None, None, None, None, err
            ws.append(w)
            names.append(self.var_name[i].get().strip())
        b, err = self.read_number(self.var_bias, "bias")
        if err:
            return None, None, None, None, err
        return xs, ws, names, b, None

    def show_result(self, text):
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.result.configure(state="disabled")
        self.result.see("1.0")

    def show_contrib(self, text):
        self.contrib.configure(state="normal")
        self.contrib.delete("1.0", "end")
        self.contrib.insert("1.0", text)
        self.contrib.configure(state="disabled")

    def show_compare(self, text):
        self.compare_txt.configure(state="normal")
        self.compare_txt.delete("1.0", "end")
        self.compare_txt.insert("1.0", text)
        self.compare_txt.configure(state="disabled")

    # ---------- печать вкладок на текущем языке ----------
    def render_step_lbl(self):
        kind = self._step_lbl[0]
        if kind == "all":
            self.var_step_lbl.set(T("step_all"))
            return
        n = self._step_lbl[1]
        base = T("step_counter", n=n, total=TOTAL_STEPS)
        self.var_step_lbl.set(base + ": " + step_title(n) if kind == "counter_title" else base)

    def render_result(self):
        kind = self._result_kind
        if kind[0] == "error":
            self.show_error_text()
        elif kind[0] == "empty":
            self.show_result(T("empty_result"))
        elif kind[0] == "reset_result":
            self.show_result(T("reset_result_msg"))
        elif kind[0] == "reset_steps":
            self.show_result(T("reset_steps_msg"))
        else:
            xs, ws, names, b, products, z, y, act = self._snap
            step = kind[1] if kind[0] == "step" else 0
            self.show_result(compute_steps(xs, ws, b, act, step, names))

    def render_contrib(self):
        kind = self._contrib_kind[0]
        if kind == "empty":
            self.show_contrib(T("empty_contrib"))
        elif kind == "reset":
            self.show_contrib(T("reset_contrib_msg"))
        else:
            xs, ws, names, b, products, z, y, act = self._snap
            self.show_contrib(contribution_text(xs, ws, names, products, z))

    def render_compare(self):
        if self._compare_kind[0] == "empty":
            self.show_compare(T("empty_compare"))
        else:
            self.show_compare(compare_activation_text(self._snap[5]))

    def show_error_text(self):
        """Сообщение об ошибке ввода на текущем языке (для «Рассчитать» и смены языка)."""
        kind, i, raw = self._err
        self.show_result(T("err_title") + "\n\n" +
                         T("err_not_number", label=self.field_label(kind, i), raw=repr(raw)) +
                         "\n\n" + T("err_hint"))

    def show_error(self, err):
        self._err = err
        self._result_kind = ("error",)
        self.show_error_text()
        self.set_status("err_status")

    def current_z_y(self):
        """z и y для текущих значений в полях (или None при ошибке)."""
        xs, ws, names, b, err = self.parse_all()
        if err:
            return None
        products = [x * w for x, w in zip(xs, ws)]
        z = sum(products) + b
        act = self.var_act.get()
        state = (xs, ws, names, b, products, z, ACTIVATIONS[act](z))
        self._snap = state + (act,)
        return state

    # ---------- обработчики кнопок ----------
    def calculate(self):
        """Этапы 1-7: полный расчёт + обновление схемы и вкладок."""
        state = self.current_z_y()
        if state is None:
            self.show_error(self.parse_all()[4])
            return
        xs, ws, names, b, products, z, y = state
        self._result_kind = ("full",)
        self._contrib_kind = ("data",)
        self.render_result()
        self.render_contrib()
        self.diagram.set_values(xs, products, z, y)
        self.var_step.set(TOTAL_STEPS)
        self._step_lbl = ("all",)
        self.render_step_lbl()
        self.diagram.highlight(TOTAL_STEPS)
        self.nb.select(0)
        self.set_status("status_calc", z=fmt_result(z), y=fmt_result(y))

    def next_step(self):
        """Пошаговый режим: один шаг за нажатие (0 -> 1 -> ... -> 6)."""
        current = self.var_step.get()
        if current >= TOTAL_STEPS:
            self.set_status("status_step_end")
            return
        nxt = current + 1
        state = self.current_z_y()
        if state is None:
            self.show_error(self.parse_all()[4])
            return
        xs, ws, names, b, products, z, y = state
        self._result_kind = ("step", nxt)
        self._contrib_kind = ("data",)
        self.render_result()
        self.render_contrib()
        self.diagram.set_values(xs, products, z, y)
        self.diagram.highlight(nxt)
        self.var_step.set(nxt)
        self._step_lbl = ("counter_title", nxt)
        self.render_step_lbl()
        self.nb.select(0)
        self.set_status(STEPS[nxt - 1][1])

    def reset_steps(self):
        self.var_step.set(0)
        self._step_lbl = ("counter", 0)
        self.render_step_lbl()
        self.diagram.highlight(0)
        self.diagram.clear_values()
        self._result_kind = ("reset_steps",)
        self.render_result()
        self.set_status("status_steps_reset")

    def compare(self):
        """Сравнение четырёх функций активации при одном и том же z."""
        state = self.current_z_y()
        if state is None:
            self.show_error(self.parse_all()[4])
            return
        z = state[5]
        self._compare_kind = ("data",)
        self.render_compare()
        self.nb.select(3)
        self.set_status("status_compare", z=fmt_result(z))

    def toggle_help(self):
        self.help_lbl.configure(state="normal")
        if self.help_shown:
            self.help_lbl.delete("1.0", "end")
        else:
            self.help_lbl.insert("1.0", guide_text())
        self.help_shown = not self.help_shown
        self.help_lbl.configure(state="disabled")

    def reset(self):
        """Вернуть форму в исходное состояние (все значения — 0)."""
        for var in self.var_x + self.var_w:
            var.set("0")
        for var in self.var_name:
            var.set("")
        self.var_bias.set("0")
        self.var_act.set("ReLU")
        self.var_step.set(0)
        self._step_lbl = ("counter", 0)
        self.render_step_lbl()
        self.diagram.highlight(0)
        self.diagram.clear_values()
        self._example_named = {}
        self._result_kind = ("reset_result",)
        self._contrib_kind = ("reset",)
        self.render_result()
        self.render_contrib()
        self.set_status("status_reset")

    def load_example(self):
        """Учебный пример: x₁=2·w₁=0.5, x₂=3·w₂=0.2, b=0.4, ReLU → y = 2.0."""
        for var in self.var_x + self.var_w:
            var.set("0")
        self._example_named = {0: "example_name1", 1: "example_name2"}
        self.var_name[0].set(T("example_name1"))
        self.var_x[0].set("2")
        self.var_w[0].set("0.5")
        self.var_name[1].set(T("example_name2"))
        self.var_x[1].set("3")
        self.var_w[1].set("0.2")
        self.var_bias.set("0.4")
        self.var_act.set("ReLU")
        self.calculate()


def contribution_text(xs, ws, names, products, z):
    """Вклад каждого входа: произведение x × w и полоска-диаграмма."""
    scale = max(1e-9, max(abs(p) for p in products))
    width = 16
    mid = width // 2
    lines = [T("contrib_title"), "",
             T("contrib_how1"),
             T("contrib_how2"), "",
             f"  {T('th_input'):<14}{T('contrib_th_product'):>21}  {T('contrib_th_bar')}", ""]
    for i, (x, w, p) in enumerate(zip(xs, ws, products), start=1):
        cells = ["·"] * width
        bar = round(mid * abs(p) / scale)
        for k in range(bar):
            cells[mid + k if p >= 0 else mid - 1 - k] = "#"
        lines.append(f"  {input_label(i, names):<14} {fmt(x):>5} × {fmt(w):>5} = "
                     f"{fmt_sign(p):>7} |{''.join(cells)}|")
    lines += ["",
              T("contrib_sum", s=fmt_result(sum(products))),
              T("contrib_bias", z=fmt_result(z)), "",
              T("note_pos"),
              T("note_neg"),
              T("note_zero"), "",
              T("contrib_exp1"),
              T("contrib_exp2")]
    return "\n".join(lines)


if __name__ == "__main__":
    NeuronApp().mainloop()
