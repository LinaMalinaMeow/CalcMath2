import decimal
import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative

dx = 0.00001
offset = 0.000001
steps = 100

predefined_functions = {
    0: lambda x: (x ** 3 - x + 4),
    1: lambda x: (x ** 3 + 4.81 * x ** 2 - 17.37 * x + 10.6),
    2: lambda x: (x / 2 - 2 * (x + 2.39) ** (1 / 3)),
    3: lambda x: (-x / 2 + math.e ** x + 5 * math.sin(x)),
    4: lambda x, y: x ** 2 + y ** 2 - 4,
    5: lambda x, y: y - 3 * x ** 2
}
functions = {
    0: 'x^3-x+4',
    1: 'x^3+4,81x^2-17,37x+5.38',
    2: 'x/2 - 2*(x + 2.39)^(1/3)',
    3: '-x/2 + e^x + 5*sin(x)',
}
methods = {
    1: 'Метод половинного деления',
    2: 'Метод простой итерации',
}


def choose_equation() -> int:
    while True:
        print("Выберите уравнение:")
        for num, func in functions.items():
            print(str(num) + ': ' + func)
        try:
            equation_number = int(input("Введите номер уравнения: "))
        except ValueError:
            print('(!) Вы ввели не число')
            continue
        if equation_number < 0 or equation_number > len(functions):
            print("(!) Такого номера нет.")
            continue
        return equation_number


def choose_method() -> int:
    while True:
        print("Выберите уравнение:")
        for num, method in methods.items():
            print(str(num) + ': ' + method)
        try:
            methods_number = int(input("Введите номер метода: "))
        except ValueError:
            print('(!) Вы ввели не число')
            continue
        if methods_number < 1 or methods_number > len(functions):
            print("(!) Такого номера нет.")
            continue
        return methods_number


def read_initial_data() -> Tuple[float, float, float, int]:
    while True:
        filename = input("Введите имя файла для загрузки исходных данных и интервала "
                         "или пустую строку, чтобы ввести вручную: ")
        if filename == '':
            left = float(input('Введите левую границу интервала: '))
            right = float(input('Введите правую границу интервала: '))
            epsilon = input('Введите погрешность вычисления: ')
            break
        else:
            try:
                f = open(filename, "r")
                left = float(f.readline())
                right = float(f.readline())
                epsilon = f.readline()
                f.close()
                print('Считано из файла:')
                print(f'Левая граница: {left}, правая: {right}, погрешность: {epsilon}')
                break
            except FileNotFoundError:
                print('(!) Файл для загрузки исходных данных не найден.')

    decimal_places = abs(decimal.Decimal(epsilon).as_tuple().exponent)
    epsilon = float(epsilon)

    return left, right, epsilon, decimal_places


def root_exists(left: float, right: float, num: int):
    return (predefined_functions[num](left) * predefined_functions[num](right) < 0) and (
            derivative(predefined_functions[num], left, dx) * derivative(predefined_functions[num], left, dx) > 0)


def check(left: float, right: float, num: int, method: int) -> Tuple[bool, str]:
    # Половинное деление
    if method == 1:
        return root_exists(left, right, num), 'Отсутствует корень на заданном промежутке' if not root_exists else ''
    # Итерационный
    else:
        f = predefined_functions[num]
        max_derivative = max(derivative(f, left, dx), derivative(f, right, dx))
        _lambda = - 1 / max_derivative
        phi = lambda x: x + _lambda * f(x)
        if not root_exists(left, right, num):
            return False, 'Отсутствует корень на заданном промежутке'

        # Достаточное условие сходимости метода |phi'(x)| < 1
        res = f'phi\'(a) = {round(abs(derivative(phi, left, dx)), 10)}\n'
        res += f'phi\'(b) = {round(abs(derivative(phi,right, dx)), 10)}\n'
        for x in np.linspace(left, right, steps, endpoint=True):
            if abs(derivative(phi, x, dx)) >= 1:
                return False, res + 'Не выполнено условие сходимости метода |phi\'(x)| < 1 на интервале'
        return True, res


def print_result(result, output_file_name):
    if output_file_name == '':
        print('\n' + str(result))
    else:
        f = open(output_file_name, "w")
        f.write(str(result))
        f.close()
        print('Результат записан в файл.')


def solve_half_division_method(num: int, left: float, right: float, epsilon: float, decimal_places: int):
    res = ""
    f = predefined_functions[num]
    a = left
    b = right
    epsilon = epsilon
    iteration = 0
    while True:
        iteration += 1
        fa = f(a)
        fb = f(b)
        x = (a + b) / 2
        fx = f(x)
        res += f'{iteration}: a = {a:.3f}, b = {b:.3f}, x = {x:.3f}, f(a) = {fa:.3f}, f(b) = {fb:.3f}, f(x)={fx:.3f}, |a-b| = {abs(a - b):.3f}\n'
        if abs(a - b) < epsilon and abs(fx) < epsilon:
            break
        if fa * fx < 0:
            b = x
        else:
            a = x
    res += '\nРезультат:\n' \
           f'Найденный корень уравнения: {round(x, decimal_places)}\n' \
           f'Значение функции в корне: {round(fx, 8)}\n' \
           f'Число итераций: {iteration}\n'
    return res


def solve_simple_iterations_method(num: int, left: float, right: float, epsilon: float, decimal_places: int):
    res = ""
    f = predefined_functions[num]
    max_derivative = max(derivative(f, left, dx), derivative(f, right, dx))
    _lambda = - 1 / max_derivative
    phi = lambda x: x + _lambda * f(x)
    prev = left

    iteration = 0
    while True:
        iteration += 1
        x = phi(prev)

        diff = abs(x - prev)
        res += f'{iteration}: xk = {prev:.3f}, f(xk) = {f(prev):.3f}, xk+1 = 𝜑(𝑥𝑘) = {x:.3f}, |xk - xk+1| = {diff:.6f}\n'
        # res += f'f={f(x)}\n'
        if diff <= epsilon and abs(f(prev)) <= epsilon:
            break
        prev = x
    res += '\nРезультат:\n' \
           f'Найденный корень уравнения: {round(x,8)}\n' \
           f'Значение функции в корне: {round(f(x), 8)}\n' \
           f'Число итераций: {iteration}\n'
    return res


def get_function_system(eq_type: int):
    if eq_type == 4:
        return lambda x, y: x ** 2 + y ** 2 - 4
    elif eq_type == 5:
        return lambda x, y: y - 3 * x ** 2


def get_function_result(equation_system_type: int, x0: float, y0: float) -> float:
    if equation_system_type == 4:
        return x0 ** 2 + y0 ** 2 - 4
    elif equation_system_type == 5:
        return y0 - 3 * x0 ** 2


def get_derivative_by_x(equation_system_type: int, x0: float, y0: float) -> float:
    if equation_system_type == 4:
        return 2 * x0
    elif equation_system_type == 5:
        return -3 * 2 * x0


def get_derivative_by_y(equation_system_type: int, x0: float, y0: float) -> float:
    if equation_system_type == 4:
        return 2 * y0
    elif equation_system_type == 5:
        return 1


def calculate_jacobian(x0: float, y0: float) -> float:
    return get_derivative_by_x(4, x0, y0) \
           * get_derivative_by_y(5, x0, y0) \
           - get_derivative_by_y(4, x0, y0) \
           * get_derivative_by_x(5, x0, y0)


def get_delta_x(x0: float, y0: float) -> float:
    return get_function_result(4, x0, y0) \
           * get_derivative_by_y(5, x0, y0) \
           - get_derivative_by_y(4, x0, y0) \
           * get_function_result(5, x0, y0)


def get_delta_y(x0: float, y0: float) -> float:
    return get_derivative_by_x(4, x0, y0) \
           * get_function_result(5, x0, y0) \
           - get_function_result(4, x0, y0) \
           * get_derivative_by_x(5, x0, y0)


def solve_system(x0: float, y0: float, accuracy: float, decimal_places: int):
    res = ""
    last_x = x0
    last_y = y0

    jacobian = calculate_jacobian(last_x, last_y)
    if jacobian == 0:
        last_x -= offset
        last_y -=offset
        jacobian = calculate_jacobian(last_x, last_y)

    x = last_x - get_delta_x(last_x, last_y) / jacobian
    y = last_y - get_delta_y(last_x, last_y) / jacobian
    iteration_amount = 0
    print()
    while abs(x - last_x) > accuracy or abs(y - last_y) > accuracy:
        last_x = x
        last_y = y

        jacobian = calculate_jacobian(last_x, last_y)
        # if jacobian == 0:
        #     last_x -= offset
        #     jacobian = calculate_jacobian(last_x, last_y)
        x = last_x - get_delta_x(last_x, last_y) / jacobian
        y = last_y - get_delta_y(last_x, last_y) / jacobian

        res += f'{iteration_amount}: last_x = {last_x:.3f}, x = {x:.3f},|x-last_x| = {abs(x - last_x):.6f} ' \
               f'last_y =  {last_y:.3f}, y = {y:.3f},|y-last_y| = {abs(y - last_y):.6f}\n'

        # res += f'значение первой функции = {get_function_system(4)(x, y)}\n'
        # res += f'значение второй функции = {get_function_system(5)(x, y)}\n'
        iteration_amount += 1
        if iteration_amount > steps:
            return 'Не нашлось решение за максимальное количество операций'
    res += '\nРезультат:\n' \
           f'Найденный x уравнения: {round(x, decimal_places)}\n' \
           f'Найденный y уравнения: {round(y, decimal_places)}\n' \
           f'Число итераций: {iteration_amount}\n'
    res += f'значение первой функции = {get_function_system(4)(x, y)}\n'
    res += f'значение второй функции = {get_function_system(5)(x, y)}\n'
    return res


def draw_graph_equation_system(f=lambda x, y: x ** 2 + y ** 2 - 4, g=lambda x, y: y - 3 * (x ** 2)):
    plt.gcf().canvas.manager.set_window_title("График функции")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    xRange = np.arange(-10, 10, 0.025)
    yRange = np.arange(-10, 10, 0.025)
    X, Y = np.meshgrid(xRange, yRange)
    F = f(X, Y)
    G = g(X, Y)
    plt.contour(X, Y, F, [0])
    plt.contour(X, Y, G, [0])
    plt.show()


def draw_graph_equation(chosen_equation: int):
    x = np.linspace(-8, 8, 1000)
    func = np.vectorize(predefined_functions[chosen_equation])(x)
    plt.title = 'График заданной функции'
    plt.grid(True, which='both')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(y=0, color='gray', label='y = 0')
    plt.plot(x, func, 'blue', label=functions[chosen_equation])
    plt.legend(loc='upper left')
    plt.savefig('graph.png')
    plt.show()


def output(method: int, num: int, left: float, right: float, epsilon: float, decimal_places: int) -> Tuple[
    str, float, float]:
    res = 'Процесс решения: \n'
    try:
        if method == 1:
            res += solve_half_division_method(num, left, right, epsilon, decimal_places)
        elif method == 2:
            res += solve_simple_iterations_method(num, left, right, epsilon, decimal_places)
        elif method == 3:
            res += solve_system(left, right, epsilon, decimal_places)
    except Exception as e:
        print(e)
        print('(!) Что-то пошло не так при решении: ', e)
        exit(-1)
    return res, round(left, 5), round(right, 5)


def main():
    while True:
        typeOfEquation = input(
            "Вы хотите решить систему уравнений или уравнение? Для решения системы введите \"1\", для решения уравнений "
            "введите \"0\": ")
        if typeOfEquation == "0":
            answer = []
            phi = []
            res = ""
            chosen_equation = choose_equation()
            chosen_method = int(choose_method())
            draw_graph_equation(chosen_equation)
            left, right_by_user, epsilon, decimal_places = read_initial_data()
            step = 0.1
            right_by_program = left + step
            while right_by_program < right_by_user:
                while predefined_functions[chosen_equation](left) * predefined_functions[chosen_equation](
                        right_by_program) > 0 and right_by_program < right_by_user:
                    left += step
                    right_by_program += step
                verified, reason = check(left, right_by_program, chosen_equation, chosen_method)
                if verified:
                    # phi for iteration method or empty for etc
                    phi.append(reason)
                    answer.append(
                        output(chosen_method, chosen_equation, left, right_by_program, epsilon, decimal_places))
                left = right_by_program
                right_by_program = left + step

            # чекаем длину нашего ответа, если =0, то корней нема((((
            if len(answer) == 0:
                try:
                    verified, reason = check(left, right_by_user, chosen_equation, chosen_method)
                except TypeError:
                    print('(!) Ошибка при вычислении значения функции, возможно она не определена на всем интервале.')
                    continue
                if not verified:
                    print('(!) Введенные исходные данные для метода некорректны: ', reason)
                    continue
            while True:
                print(f'Было найдено {len(answer)} корней')
                print("Вам будет предложены следующие промежутки на выбор:")
                for i in range(len(answer)):
                    # 1pass
                    print(f'({i}: {answer[i][1]} - {answer[i][2]})\n')
                # 1print(len(answer))
                # вывод всех корней
                #     res += answer[i][0] + "\n"
                # break
                try:
                    chosen_answer = int(input("Введите выбранный промежуток: "))
                except ValueError:
                    continue
                if 0 <= chosen_answer < len(answer):
                    res += phi[chosen_answer] + answer[chosen_answer][0]
                    break
                print("Попробуйте выбрать снова. Вы выбрали неверно!\n")
                continue
            output_file_name = input(
                "Введите имя файла для вывода результата или пустую строку, чтобы вывести в консоль: ")
            print_result(res, output_file_name)
        else:
            draw_graph_equation_system()
            left, right_by_user, epsilon, decimal_places = read_initial_data()
            res = output(3, -1, left, right_by_user, epsilon, decimal_places)
            output_file_name = input(
                "Введите имя файла для вывода результата или пустую строку, чтобы вывести в консоль: ")
            print_result(res[0], output_file_name)
        if input('\nЕще раз? [y/n] ') != 'y':
            break


main()
