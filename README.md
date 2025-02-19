
* Regressão Linear com Gradiente Descendente
* Exercício Coursera

--- 


def compute_gradient(x, y, w, b):
    """
    Calcula os gradientes para os parâmetros w e b.
    """
    m = x.shape[0]  # Número de exemplos
    f_wb = w * x + b  # Vetor de previsões
    error = f_wb - y  # Diferença entre previsão e valores reais

    dj_dw = (1 / m) * np.dot(error, x)  # Gradiente em relação a w
    dj_db = (1 / m) * np.sum(error)  # Gradiente em relação a b

    return dj_dw, dj_db



---

def compute_cost(x, y, w, b):
    """
    Calcula a função de custo para regressão linear.
    """
    m = len(x)  # Número de exemplos de treinamento
    f_wb = w * x + b  # Vetor de previsões
    cost = np.sum((f_wb - y) ** 2) / (2 * m)  # Soma dos erros quadráticos médios
    return cost
