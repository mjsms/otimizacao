# transporte.py
import matplotlib.pyplot as plt
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus

# -------------------------
# 1) Definição do Problema
# -------------------------
armazens = ["A1", "A2"]
centros = ["CD1", "CD2", "CD3"]
oferta = {"A1": 300, "A2": 500}
procura = {"CD1": 200, "CD2": 250, "CD3": 250}
custo_inicial = {
    ("A1","CD1"): 2.0, ("A1","CD2"): 3.5, ("A1","CD3"): 3.0,
    ("A2","CD1"): 2.5, ("A2","CD2"): 2.0, ("A2","CD3"): 4.0
}

# Criar modelo
model = LpProblem("Transporte_Adega", LpMinimize)

# Variáveis de decisão x[i][j] = quantidade enviada do armazém i ao centro j
x = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous")
     for i in armazens for j in centros}

# Função objetivo: minimizar custo total
model += lpSum(custo_inicial[(i,j)] * x[(i,j)] for i in armazens for j in centros), "Custo_Total"

# Restrições de oferta
for i in armazens:
    model += lpSum(x[(i,j)] for j in centros) <= oferta[i], f"Oferta_{i}"

# Restrições de procura
for j in centros:
    model += lpSum(x[(i,j)] for i in armazens) == procura[j], f"Procura_{j}"

# Resolver
model.solve()

print("\n===> SOLUÇÃO ÓTIMA PARA O PROBLEMA DE TRANSPORTE <===")
print("Status:", LpStatus[model.status])
print(f"Custo Total de Transporte: {value(model.objective):.2f} €")

# Extrair solução para Tabela 1
solucao = []
for i in armazens:
    for j in centros:
        q = value(x[(i,j)])
        if q > 0:
            custo_unit = custo_inicial[(i,j)]
            solucao.append((i, j, q, custo_unit, q*custo_unit))

# Imprimir Tabela 1 em texto
print("\nTABELA 1: Quantidades e Custos")
print(f"{'Armazém':<6} {'Centro':<4} {'Qtd':>8} {'CustoUnit':>10} {'CustoTotal':>12}")
for row in solucao:
    print(f"{row[0]:<6} {row[1]:<5} {row[2]:8.0f} {row[3]:10.2f} {row[4]:12.2f}")

# Gerar Tabela 1 como figura (opcional)
fig, ax = plt.subplots(figsize=(6,2))
ax.axis('off')
col_labels = ["Armazém", "Centro Dist.", "Qtd (caixas)", "Custo Unit.", "Custo Total"]
table_data = []
for (i, j, qtd, cunit, ctotal) in solucao:
    table_data.append([i, j, f"{qtd:.0f}", f"{cunit:.2f}", f"{ctotal:.2f}"])

the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
the_table.scale(1.2, 1.2)
plt.savefig("Tabela1.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------
# 2) Análise de Sensibilidade: Figura 1 (gráfico de linhas)
# ------------------------------------------------------
# Vamos variar o custo de A2->CD3 de 4.0 até 6.0, e ver o impacto no custo total

custos_variados = np.linspace(3.0, 8.0, 11)  # [4.0, 4.4, 4.8, 5.2, 5.6, 6.0]
custo_total_lista = []

for cvar in custos_variados:
    # Criar novo modelo para cada cenário
    model_sens = LpProblem("Transporte_Sens", LpMinimize)
    x_sens = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous")
              for i in armazens for j in centros}
    
    # Atualizar custo de A2->CD3
    custo_atual = dict(custo_inicial)
    custo_atual[("A2","CD3")] = cvar
    
    model_sens += lpSum(custo_atual[(i,j)] * x_sens[(i,j)] for i in armazens for j in centros), "Custo_Total"
    for i in armazens:
        model_sens += lpSum(x_sens[(i,j)] for j in centros) <= oferta[i]
    for j in centros:
        model_sens += lpSum(x_sens[(i,j)] for i in armazens) == procura[j]
    
    model_sens.solve()
    custo_total_lista.append(value(model_sens.objective))

# Figura 1: gráfico de linhas (custo total vs. custo A2->CD3)
plt.figure(figsize=(6,4))
plt.plot(custos_variados, custo_total_lista, marker='o')
plt.title("Figura 1: Impacto de variações no custo A2→CD3 no Custo Total")
plt.xlabel("Custo A2→CD3 (€/caixa)")
plt.ylabel("Custo Total de Transporte (€)")
plt.grid(True)
plt.savefig("Figura1.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n[OK] Geradas as figuras: 'Tabela1.png' (com a solução) e 'Figura1.png' (sensibilidade).")
print("Fim do script transporte.py\n")