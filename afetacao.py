# afetacao.py
import matplotlib.pyplot as plt
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value, LpBinary

# ----------------------------
# 1) Definição do Problema
# ----------------------------
funcionarios = ["F1", "F2", "F3", "F4"]
tarefas = ["Lavagem", "Transporte", "Engarrafamento", "Controlo"]
custo_inicial = {
    ("F1","Lavagem"): 25, ("F1","Transporte"): 40, ("F1","Engarrafamento"): 30, ("F1","Controlo"): 35,
    ("F2","Lavagem"): 35, ("F2","Transporte"): 30, ("F2","Engarrafamento"): 25, ("F2","Controlo"): 45,
    ("F3","Lavagem"): 20, ("F3","Transporte"): 35, ("F3","Engarrafamento"): 40, ("F3","Controlo"): 30,
    ("F4","Lavagem"): 30, ("F4","Transporte"): 25, ("F4","Engarrafamento"): 35, ("F4","Controlo"): 20
}

model = LpProblem("Afetacao", LpMinimize)

# Variáveis binárias: x[i,j] = 1 se funcionário i executa tarefa j
x = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=LpBinary)
     for i in funcionarios for j in tarefas}

# Função objetivo: minimizar custo total
model += lpSum(custo_inicial[(i,j)] * x[(i,j)] for i in funcionarios for j in tarefas), "CustoTotal"

# Restrição: cada funcionário faz exatamente 1 tarefa
for i in funcionarios:
    model += lpSum(x[(i,j)] for j in tarefas) == 1, f"Res_func_{i}"

# Restrição: cada tarefa é atribuída a exatamente 1 funcionário
for j in tarefas:
    model += lpSum(x[(i,j)] for i in funcionarios) == 1, f"Res_tarefa_{j}"

model.solve()

print("\n===> SOLUÇÃO ÓTIMA PARA O PROBLEMA DE AFETAÇÃO <===")
print("Status:", LpStatus[model.status])
print(f"Custo Total de Afetação: {value(model.objective):.2f} €")

# Extrair alocação ótima
alocacao = {}
for i in funcionarios:
    for j in tarefas:
        if value(x[(i,j)]) == 1:
            alocacao[j] = (i, custo_inicial[(i,j)])  # Tarefa j atribuída a i, com custo
            print(f" - Tarefa '{j}' => Funcionário {i} (custo {custo_inicial[(i,j)]} €)")

# ----------------------------------
# 2) Figura 2: Gráfico de barras da alocação
# ----------------------------------
# Vamos pôr cada tarefa no eixo X, e a altura do gráfico é o custo correspondente.
# E no rótulo, indicamos qual funcionário foi atribuído.

tarefas_ord = list(alocacao.keys())
custos_tarefas = [alocacao[t][1] for t in tarefas_ord]
labels_func = [alocacao[t][0] for t in tarefas_ord]

plt.figure(figsize=(6,4))
bars = plt.bar(tarefas_ord, custos_tarefas, color="skyblue")
plt.title("Figura 2: Alocação Ótima das Tarefas (custo por tarefa)")
plt.xlabel("Tarefas")
plt.ylabel("Custo da Atribuição (€)")

# Adicionar o funcionário como texto em cima da barra
for bar, func in zip(bars, labels_func):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height+0.5, func,
             ha='center', va='bottom', fontsize=9, color="blue")

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Figura2.png", dpi=300, bbox_inches='tight')
plt.close()

# --------------------------------------------
# 3) Figura 3: Análise de sensibilidade
# --------------------------------------------
# Exemplo: variar o custo de F4->Controlo de 20 até 50, e ver como muda o custo total

cvals = np.linspace(20, 50, 7)  # [20,25,30,35,40,45,50]
ctotais = []

for cvar in cvals:
    # Copiar custos
    custo_mod = dict(custo_inicial)
    custo_mod[("F4","Controlo")] = cvar

    # Novo modelo
    m_sens = LpProblem("Afetacao_Sens", LpMinimize)
    x_sens = {(i,j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in funcionarios for j in tarefas}
    m_sens += lpSum(custo_mod[(i,j)] * x_sens[(i,j)] for i in funcionarios for j in tarefas), "Obj"
    for i in funcionarios:
        m_sens += lpSum(x_sens[(i,j)] for j in tarefas) == 1
    for j in tarefas:
        m_sens += lpSum(x_sens[(i,j)] for i in funcionarios) == 1
    m_sens.solve()
    ctotais.append(value(m_sens.objective))

plt.figure(figsize=(6,4))
plt.plot(cvals, ctotais, marker='o', color="green")
plt.title("Figura 3: Variação do custo de F4->Controlo vs. Custo Total")
plt.xlabel("Custo(F4->Controlo) (€)")
plt.ylabel("Custo Total de Afetação (€)")
plt.grid(True)
plt.savefig("Figura3.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n[OK] Geradas as figuras: 'Figura2.png' (alocação) e 'Figura3.png' (sensibilidade).")
print("Fim do script afetacao.py\n")
