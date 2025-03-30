import matplotlib.pyplot as plt
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus

# -----------------------------------
# 1) Definição do Problema (Metas)
# -----------------------------------
armazens = ["A1", "A2"]
centros = ["CD1", "CD2", "CD3"]

# Capacidades amplas, permitindo qualquer um dos armazéns suprir toda a procura se for vantajoso
oferta = {"A1": 700, "A2": 700}

# Procura total = 700, com igualdade
procura = {"CD1": 200, "CD2": 250, "CD3": 250}

# Custos: A1 é barato, A2 é bem mais caro, para criar conflito real
custo = {
    ("A1","CD1"): 2.0,  ("A1","CD2"): 3.0,  ("A1","CD3"): 2.5,
    ("A2","CD1"): 5.0,  ("A2","CD2"): 5.5, ("A2","CD3"): 6.0
}

# Metas:
#   1) Custo total ≤ 2500
#   2) Expedir de A2 ≈ 400 (penalizando desvios para mais e para menos)
model = LpProblem("ProgMetas", LpMinimize)

# Variáveis de transporte
x = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0)
     for i in armazens for j in centros}

# Desvios
d1_minus = LpVariable("d1_minus", lowBound=0)
d1_plus  = LpVariable("d1_plus",  lowBound=0)
d2_minus = LpVariable("d2_minus", lowBound=0)
d2_plus  = LpVariable("d2_plus",  lowBound=0)

# Expressão de custo
cost_expr = lpSum(custo[(i,j)] * x[(i,j)] for i in armazens for j in centros)

# Meta 1: cost_expr + d1_minus - d1_plus = 2500
model += cost_expr + d1_minus - d1_plus == 2500, "MetaCusto"

# Meta 2: sum(A2->CDs) + d2_minus - d2_plus = 400
model += lpSum(x[("A2",j)] for j in centros) + d2_minus - d2_plus == 400, "MetaA2"

# Restrições de oferta e procura
model += lpSum(x[("A1",j)] for j in centros) <= oferta["A1"], "CapA1"
model += lpSum(x[("A2",j)] for j in centros) <= oferta["A2"], "CapA2"
for c in centros:
    model += lpSum(x[(i,c)] for i in armazens) == procura[c], f"Proc_{c}"

# Função objetivo: penalizar exceder ou ficar aquém da meta de A2
# e também penalizar ultrapassar a meta de custo
w1, w2 = 1, 1
model += w1*d1_plus + w2*(d2_minus + d2_plus), "ObjMetas"

model.solve()

print("\n===> SOLUÇÃO ÓTIMA PARA PROGRAMAÇÃO POR METAS <===")
print("Status:", LpStatus[model.status])
custo_total = value(cost_expr)
expA2 = value(lpSum(x[("A2",j)] for j in centros))
print(f"Custo Total: {custo_total:.2f} €")
print(f"Expedição de A2: {expA2:.0f} caixas")

# Extrair quantidades para cada centro
distA1 = [value(x[("A1",cd)]) for cd in centros]
distA2 = [value(x[("A2",cd)]) for cd in centros]

# ---------------------------------------
# 2) Figura 4: gráfico de barras empilhadas
# ---------------------------------------
plt.figure(figsize=(8, 5))
indices = np.arange(len(centros))
plt.bar(indices, distA1, color='orange', label="A1")
plt.bar(indices, distA2, bottom=distA1, color='blue', label="A2")
plt.xticks(indices, centros, fontsize=10)
plt.xlabel("Centros de Distribuição", fontsize=10)
plt.ylabel("Quantidade de Caixas", fontsize=10)
plt.title("Figura 4: Distribuição das Caixas (A1 vs. A2) por Centro", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Figura4.png", dpi=300)
plt.close()

# --------------------------------------
# 3) Figura 5: gráfico de linha + marcadores
# --------------------------------------
w2_values = np.linspace(0, 5, 6)  # [0,1,2,3,4,5]
expA2_list = []

for w2test in w2_values:
    m2 = LpProblem("ProgMetas_sens", LpMinimize)
    x2 = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0) for i in armazens for j in centros}
    d1m = LpVariable("d1_minus", lowBound=0)
    d1p = LpVariable("d1_plus",  lowBound=0)
    d2m = LpVariable("d2_minus", lowBound=0)
    d2p = LpVariable("d2_plus",  lowBound=0)
    
    cost_expr2 = lpSum(custo[(i,j)] * x2[(i,j)] for i in armazens for j in centros)
    # Meta 1
    m2 += cost_expr2 + d1m - d1p == 2500
    # Meta 2
    m2 += lpSum(x2[("A2",cd)] for cd in centros) + d2m - d2p == 400

    m2 += lpSum(x2[("A1",cd)] for cd in centros) <= oferta["A1"]
    m2 += lpSum(x2[("A2",cd)] for cd in centros) <= oferta["A2"]
    for c in centros:
        m2 += lpSum(x2[(i,c)] for i in armazens) == procura[c]

    # função objetivo penaliza d1p e (d2m + d2p)
    m2 += 1 * d1p + w2test * (d2m + d2p)
    m2.solve()

    expA2_sens = value(lpSum(x2[("A2",cd)] for cd in centros))
    expA2_list.append(expA2_sens)

plt.figure(figsize=(8, 5))
plt.plot(w2_values, expA2_list, marker='o', linestyle='--', color='purple', linewidth=2)

plt.title("Figura 5: Variação de w₂ e Impacto na Expedição de A2", fontsize=12)
plt.xlabel("Peso w₂ (importância da meta de A2)", fontsize=10)
plt.ylabel("Expedição de A2 (caixas)", fontsize=10)

y_min = min(expA2_list) - 10
y_max = max(expA2_list) + 10
plt.ylim([y_min, y_max])

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Figura5.png", dpi=300)
plt.close()

print("\n[OK] Geradas as figuras: 'Figura4.png' e 'Figura5.png'.")
print("Fim do script metas.py\n")
