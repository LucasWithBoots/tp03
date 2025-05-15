import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  # Necessário para spring_layout e outras operações do networkx


def read_book_text(filepath):
    """
    Lê o arquivo de texto do livro.

    Args:
        filepath (str): Caminho para o arquivo de texto do livro.

    Returns:
        str: Conteúdo do livro.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        return ""


def calculate_character_proximity(text, characters, window_size=100):
    """
    Calcula a proximidade entre personagens no texto.

    Args:
        text (str): Texto completo do livro.
        characters (list): Lista de personagens para analisar (nomes com capitalização original).
        window_size (int): Número de palavras a considerar como proximidade para frente e para trás.

    Returns:
        dict: Matriz de proximidade (simétrica) entre personagens.
              {'CharA': {'CharB': count, ...}, ...}
    """
    words_original_case = text.split()
    words_lower_case = [word.lower() for word in words_original_case]  # Para correspondência

    # Inicializa a matriz de proximidade com nomes de personagens originais como chaves
    proximity = {char1: {char2: 0 for char2 in characters} for char1 in characters}

    # Mapa de nomes minúsculos para nomes originais para facilitar a correspondência
    char_lower_to_original_map = {char.lower(): char for char in characters}

    for i, current_word_lower in enumerate(words_lower_case):
        # Identifica quais personagens (nomes originais) são mencionados na palavra atual (minúscula)
        # Esta verificação 'in' pode corresponder a substrings. Para nomes únicos de GoT, é geralmente aceitável.
        # Uma abordagem mais robusta usaria regex com limites de palavra: r'\b' + re.escape(lc_char) + r'\b'

        chars_found_in_current_word = []  # Armazena nomes originais dos personagens encontrados em words_lower_case[i]
        for lc_char_key, original_char_name_val in char_lower_to_original_map.items():
            if lc_char_key in current_word_lower:
                chars_found_in_current_word.append(original_char_name_val)

        if chars_found_in_current_word:
            # Define os limites da janela em relação ao índice da palavra atual `i`
            start_idx = max(0, i - window_size)
            # end_idx é exclusivo para fatiamento em Python. Para incluir até `i + window_size`,
            # o limite superior da fatia deve ser `i + window_size + 1`.
            end_idx = min(len(words_lower_case), i + window_size + 1)

            # Palavras (minúsculas) na janela de proximidade ao redor de words_lower_case[i]
            window_segment_lower_case = words_lower_case[start_idx:end_idx]

            for current_char_name_original in chars_found_in_current_word:  # Ex: "Arya"
                # Agora, procure por *outros* personagens dentro do window_segment_lower_case
                for other_char_name_original in characters:  # Ex: "Bran", "Jaime", etc.
                    if other_char_name_original == current_char_name_original:
                        continue  # Não contar proximidade consigo mesmo

                    other_char_name_lower = other_char_name_original.lower()

                    # Verifica se other_char_name_lower está presente em alguma palavra da janela
                    # O window_segment_lower_case inclui words_lower_case[i]. Isso é correto.
                    if any(other_char_name_lower in w_lower for w_lower in window_segment_lower_case):
                        proximity[current_char_name_original][other_char_name_original] += 1
                        # A matriz de proximidade se torna simétrica porque quando o loop para `i`
                        # eventualmente atingir uma palavra contendo `other_char_name_original`,
                        # `current_char_name_original` será encontrado em sua janela.
    return proximity


def create_network(proximity_matrix):
    """
    Cria um grafo networkx a partir da matriz de proximidade.

    Args:
        proximity_matrix (dict): Matriz de proximidade entre personagens.

    Returns:
        networkx.Graph: Grafo ponderado das relações entre personagens.
    """
    G = nx.Graph()

    # Adiciona nós (todos os personagens)
    for char_node in proximity_matrix:
        G.add_node(char_node)

    # Adiciona arestas ponderadas
    # char1 itera através das chaves (nomes dos personagens)
    for char1 in proximity_matrix:
        # proximity_matrix[char1] é um dict como {'CharacterB': count, 'CharacterC': count}
        for char2, weight in proximity_matrix[char1].items():
            # Para adicionar arestas apenas uma vez em um grafo não direcionado e evitar auto-loops:
            if char1 < char2:  # Comparação lexicográfica garante (A,B) e não (B,A) depois
                if weight > 0:
                    G.add_edge(char1, char2, weight=weight)
    return G


def invert_edge_weights(G_input):
    """
    Inverte os pesos das arestas mantendo a proporcionalidade.
    Pesos originais maiores (mais forte relação) se tornarão pesos invertidos menores (menor "custo").

    Args:
        G_input (networkx.Graph): Grafo de entrada.

    Returns:
        networkx.Graph: Grafo com pesos das arestas invertidos.
    """
    H = G_input.copy()  # Trabalha em uma cópia

    if H.number_of_edges() == 0:
        return H

    weights = [d['weight'] for u, v, d in H.edges(data=True)]

    if not weights:  # Segurança adicional
        return H

    max_weight = max(weights)
    min_weight = min(weights)

    # Se todos os pesos forem iguais, a fórmula max+min-w = w+w-w = w.
    # Isso significa que seus "custos" permanecem os mesmos em relação uns aos outros (todos iguais).
    # A proporcionalidade é mantida.

    for u, v, data in H.edges(data=True):
        current_weight = data['weight']
        inverted_weight = (max_weight + min_weight) - current_weight
        H[u][v]['weight'] = inverted_weight  # Define o novo peso como 'weight'

    return H


def calculate_betweenness(G_with_inverted_weights):
    """
    Calcula a centralidade de intermediação (betweenness) para cada nó.
    Assume que o atributo 'weight' das arestas representa custo/distância.

    Args:
        G_with_inverted_weights (networkx.Graph): Grafo de entrada com pesos invertidos.

    Returns:
        dict: Centralidade de intermediação para cada nó.
    """
    # Pesos invertidos: maior proximidade original -> menor peso invertido (custo)
    return nx.betweenness_centrality(G_with_inverted_weights, weight='weight', normalized=True)


def calculate_circuit_rank(G):
    """
    Calcula o circuit rank (número ciclomático) do grafo.
    r = e - v + c

    Args:
        G (networkx.Graph): Grafo de entrada.

    Returns:
        int: Circuit rank do grafo.
    """
    e = G.number_of_edges()  # Número de arestas
    v = G.number_of_nodes()  # Número de vértices
    c = nx.number_connected_components(G)  # Número de componentes conexos
    return e - v + c


def visualize_network(G_for_layout_and_nodes, betweenness_centrality, G_original_weights_source):
    """
    Visualiza a rede com tamanhos de nó baseados na centralidade de intermediação
    e espessura das arestas baseada nos pesos originais.

    Args:
        G_for_layout_and_nodes (networkx.Graph): Grafo usado para layout e nós (com pesos invertidos).
        betweenness_centrality (dict): Centralidade de intermediação para cada nó.
        G_original_weights_source (networkx.Graph): Grafo com os pesos de proximidade originais (para espessura das arestas).
    """
    plt.figure(figsize=(13, 9))

    # Tamanhos dos nós baseados na betweenness (calculada em G_for_layout_and_nodes)
    # O fator de escala pode precisar de ajuste
    node_sizes = [betweenness_centrality.get(node, 0) * 20000 + 500 for node in G_for_layout_and_nodes.nodes()]

    # Posições dos nós (layout) usando G_for_layout_and_nodes (com pesos invertidos como 'distância')
    # spring_layout interpreta pesos menores como atrações mais fortes.
    k_value = 0.5 / np.sqrt(
        G_for_layout_and_nodes.number_of_nodes()) if G_for_layout_and_nodes.number_of_nodes() > 0 else 0.5
    pos = nx.spring_layout(G_for_layout_and_nodes, seed=42, weight='weight', k=k_value)

    # Prepara espessuras das arestas com base nos pesos originais de G_original_weights_source
    # para as arestas presentes em G_for_layout_and_nodes.
    edge_widths = []
    edges_to_draw = list(G_for_layout_and_nodes.edges())

    original_weights_for_drawn_edges = []
    if G_original_weights_source.number_of_edges() > 0:
        for u, v in edges_to_draw:
            if G_original_weights_source.has_edge(u, v):
                original_weights_for_drawn_edges.append(G_original_weights_source[u][v]['weight'])
            else:  # Não deve acontecer se G_for_layout_and_nodes é derivado de G_original_weights_source
                original_weights_for_drawn_edges.append(0)

    if not original_weights_for_drawn_edges or max(original_weights_for_drawn_edges, default=0) == 0:
        edge_widths = [1] * len(edges_to_draw)  # Largura padrão
    else:
        max_orig_w = max(original_weights_for_drawn_edges)
        # Mais espesso para peso original maior. Fator de escala para espessura. Espessura mínima de 0.5.
        edge_widths = [(w / max_orig_w * 4.5) + 0.5 for w in original_weights_for_drawn_edges]

    nx.draw_networkx_nodes(G_for_layout_and_nodes, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9,
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G_for_layout_and_nodes, pos, edgelist=edges_to_draw, width=edge_widths,
                           edge_color='grey', alpha=0.7)
    nx.draw_networkx_labels(G_for_layout_and_nodes, pos, font_size=9, font_weight='bold')

    plt.title("Rede de Personagens de Game of Thrones", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Personagens a serem analisados (escolher 10 da lista de 15 fornecida no trabalho)
    # Lista do trabalho: Arya, Bran, Brienne, Catelyn, Cersei, Daenerys, Jaime, Melisandre, Petyr, Robert, Sam, Sansa, Theon, Tyrion, Varys
    selected_characters = [
        'Arya', 'Bran', 'Catelyn', 'Cersei', 'Daenerys',
        'Jaime', 'Petyr', 'Sansa', 'Tyrion', 'Robert'
    ]

    book_filepath = 'public/A-Game-Of-Thrones.txt'  # !! Substitua pelo caminho correto do seu arquivo !!
    # Exemplo: 'A-Game-Of-Thrones.txt' se estiver na mesma pasta
    # ou 'livros/A-Game-Of-Thrones.txt'
    book_text = read_book_text(book_filepath)

    if not book_text:
        print(f"Texto do livro não pôde ser lido em '{book_filepath}'. Saindo.")
        return

    print(f"Lidos {len(book_text):,} caracteres do livro.")

    print("\nCalculando proximidades entre personagens...")
    proximity_matrix = calculate_character_proximity(book_text, selected_characters, window_size=100)

    print("Criando grafo da rede...")
    G_original_weights = create_network(proximity_matrix)
    print(
        f"Rede criada com {G_original_weights.number_of_nodes()} nós e {G_original_weights.number_of_edges()} arestas.")

    if G_original_weights.number_of_edges() == 0:
        print(
            "AVISO: O grafo não possui arestas. Verifique o cálculo de proximidade, a lista de personagens ou o texto do livro.")

    # print("\nPesos originais das arestas (exemplo):")
    # for u, v, data in list(G_original_weights.edges(data=True))[:5]: # Mostra os 5 primeiros
    #     print(f"Aresta ({u}-{v}), Peso Original: {data['weight']}")

    print("Invertendo pesos das arestas para cálculo de betweenness...")
    G_inverted_weights = invert_edge_weights(G_original_weights)

    print("Calculando centralidade de betweenness...")
    betweenness = calculate_betweenness(G_inverted_weights)

    print("\n--- Centralidade de Betweenness (Mais Centrais Primeiro) ---")
    sorted_betweenness = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
    for char, centrality_score in sorted_betweenness:
        print(f"{char}: {centrality_score:.4f}")

    print("\nCalculando circuit rank...")
    c_rank = calculate_circuit_rank(G_original_weights)
    print(f"Circuit Rank (r = e - v + c): {c_rank}")

    print("\nVisualizando a rede...")
    if G_original_weights.number_of_nodes() > 0:
        visualize_network(G_inverted_weights, betweenness, G_original_weights)
    else:
        print("Grafo está vazio, pulando visualização.")


if __name__ == "__main__":
    main()