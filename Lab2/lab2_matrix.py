"""1. Построить алгоритм произведения матриц через MapReduce
2. Построить алгоритм линейной регрессии через MapReduce
3. Продемонстрировать их работу
"""

from collections import defaultdict
import numpy as np


def run_map_reduce(inputs, mapper, reducer):
    intermediate = defaultdict(list)
    for record in inputs:
        for key, value in mapper(record):
            intermediate[key].append(value)

    outputs = []
    for key, values in intermediate.items():
        for out in reducer(key, values):
            outputs.append(out)

    return outputs


def matrix_to_records(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2

    records = []

    for i in range(m):
        for k in range(n):
            if A[i, k] != 0:
                records.append(("A", i, k, A[i, k]))

    for k in range(n):
        for j in range(p):
            if B[k, j] != 0:
                records.append(("B", k, j, B[k, j]))

    return records, m, p


def mapper_mm_job1(record):
    tag, i_or_k, k_or_j, value = record

    if tag == "A":
        i = i_or_k
        k = k_or_j
        yield (k, ("A", i, value))

    elif tag == "B":
        k = i_or_k
        j = k_or_j
        yield (k, ("B", j, value))


def reducer_mm_job1(k, values):
    a_list = []
    b_list = []

    for tag, index, val in values:
        if tag == "A":
            a_list.append((index, val))
        else:
            b_list.append((index, val))

    for i, a_ik in a_list:
        for j, b_kj in b_list:
            partial = a_ik * b_kj

            yield ((i, j), partial)


def mapper_mm_job2(record):
    (i, j), partial = record

    yield ((i, j), partial)


def reducer_mm_job2(key, values):
    total = sum(values)
    yield (key, total)


def multiply_matrices_mapreduce(A, B):
    records, m, p = matrix_to_records(A, B)

    job1_output = run_map_reduce(records, mapper_mm_job1, reducer_mm_job1)

    job2_output = run_map_reduce(job1_output, mapper_mm_job2, reducer_mm_job2)

    C = np.zeros((m, p), dtype=float)
    for (i, j), value in job2_output:
        C[i, j] = value

    return C


def mapper_lr(record):
    x, y = record
    x = np.asarray(x, dtype=float)
    y = float(y)

    xTx = np.outer(x, x)
    xTy = x * y

    yield ("stats", ("XX", xTx, xTy))


def reducer_lr(key, values):
    S_xx = None
    S_xy = None

    for tag, xTx, xTy in values:
        if S_xx is None:
            S_xx = np.array(xTx, dtype=float)
            S_xy = np.array(xTy, dtype=float)
        else:
            S_xx += xTx
            S_xy += xTy

    yield (key, (S_xx, S_xy))


def linear_regression_mapreduce(dataset):
    mr_output = run_map_reduce(dataset, mapper_lr, reducer_lr)

    _, (S_xx, S_xy) = mr_output[0]

    w = np.linalg.inv(S_xx) @ S_xy
    return w, S_xx, S_xy


def demo_matrix_multiplication():
    print("=== Демонстрация: произведение матриц через MapReduce ===")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)  # 2x3
    B = np.array([[1, 2],
                  [0, 1],
                  [1, 0]], dtype=float)  # 3x2

    print("Матрица A:")
    print(A)
    print("Матрица B:")
    print(B)

    C_mr = multiply_matrices_mapreduce(A, B)
    print("Результат C = A * B (через MapReduce):")
    print(C_mr)

    C_np = A @ B
    print("Проверка (numpy A @ B):")
    print(C_np)
    print()


def demo_linear_regression():
    print("=== Демонстрация: линейная регрессия через MapReduce ===")
    dataset = [
        (np.array([1, 1, 2]), 5),
        (np.array([1, 2, 0]), 4),
        (np.array([1, 3, 1]), 7),
        (np.array([1, 4, 3]), 10),
    ]

    print("Обучающая выборка (x1, x2, y):")
    for x, y in dataset:
        print(f"x1={x[1]}, x2={x[2]}, y={y}")

    w, S_xx, S_xy = linear_regression_mapreduce(dataset)

    print("\nСуммарная матрица S_xx = X^T X:")
    print(S_xx)
    print("\nСуммарный вектор S_xy = X^T y:")
    print(S_xy)

    print("\nОценённые параметры w = (w0, w1, w2):")
    print(w)

    print("\nПроверка предсказаний на обучающей выборке:")
    for x, y in dataset:
        y_pred = float(np.dot(x, w))
        print(f"x = {x}, y = {y}, y_pred = {y_pred:.4f}")
    print()


if __name__ == "__main__":
    demo_matrix_multiplication()
    demo_linear_regression()
