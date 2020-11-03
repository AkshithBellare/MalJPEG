def perf_measure(y_actual, y_hat):
    T = 0
    F = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i]:
            T+=1
        if y_actual[i] != y_hat[i]:
            F+=1
        if y_actual[i] == 0 and y_hat[i] == 100:
            FP += 1
        if y_actual[i] == 100 and y_hat[i] == 100:
            TP += 1
        if y_actual[i] == 0 and y_hat[i] == 0:
            TN += 1
        if y_actual[i] == 100 and y_hat[i] == 0:
            FN += 1
    return (TP, FP,TN, FN)

TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

sensitivity = TP / (TP + FN)
print(sensitivity)

specificity = TN / (TN + FP)
print(specificity)

precision = TP / (TP + FP)
print(precision)

F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
print(F1)