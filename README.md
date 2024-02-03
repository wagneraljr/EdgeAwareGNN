# AttEdgeAwareGNN: Previsão de Carga em Redes Backbone com GNN Sensível a Arestas

## Visão Geral
O projeto EdgeAwareGNN introduz um modelo inovador de Rede Neural Gráfica (GNN) sensível a arestas aplicado à a previsão de carga em redes backbone. Este projeto visa explorar atributos de arestas como forma de enriquecer as representações de nós geradas por GNNs. Através da comparação com arquiteturas tradicionais de GNNs, o AttEdgeAwareGNN demonstra melhorias significativas na precisão das previsões.

## Arquitetura dos Modelos
O projeto implementa múltiplas arquiteturas de GNN, incluindo os tradicionais GCN e GraphSAGE e o nosso modelo `AttEdgeAwareGCN`, que incorpora mecanismos de atenção às arestas. O arquivo `models.py` descreve esses modelos detalhadamente:

- **GCN**: Modelo inspirado nas redes neurais convolucionais aplicadas especificamente a grafos.
- **GraphSAGE**: Um modelo robusto que utiliza convoluções de grafos para agregação de informações de vizinhança.
- **AttEdgeAwareGNN**: Um avanço sobre o GCN tradicional, este modelo integra informações de aresta de forma eficaz, permitindo uma previsão de carga mais precisa.

## Preparação dos Dados
Os modelos são treinados usando dados históricos de tráfego em uma rede backbone, especificamente a rede Abilene, representada pelo arquivo `Abilene.gml`. As matrizes de tráfego foram medidas 

## Otimização de Hiperparâmetros
A otimização de hiperparâmetros é realizada pelos scripts `hyper_day.py` e `hyper_week.py`, que ajustam os modelos usando dados de um dia e uma semana, respectivamente. Esses scripts empregam a biblioteca `optuna` para encontrar a configuração ideal de hiperparâmetros que maximiza a precisão das previsões. Este processo é crucial para garantir que os modelos estejam bem ajustados às características específicas dos dados de tráfego.

## Avaliação e Teste
Após a otimização, os scripts `test_day.py` e `test_week.py` avaliam o desempenho dos modelos. Estes scripts carregam os hiperparâmetros otimizados e usam novos conjuntos de dados para prever as cargas, comparando as previsões com valores reais. As métricas de avaliação incluem:

- **Erro Médio Absoluto (MAE)**
- **Erro Quadrático Médio (MSE)**
- **Coeficiente de Determinação (R²)**

Estas métricas ajudam a quantificar o quão próximo as previsões estão dos valores reais, proporcionando uma medida clara da eficácia do modelo.

## Visualização dos Resultados
Os resultados das previsões são visualizados utilizando `matplotlib`, gerando gráficos que comparam as cargas previstas com as reais. Esta visualização facilita a identificação de discrepâncias e destaca a precisão das previsões do modelo. Exemplos de tais gráficos podem ser gerados pelos scripts de teste e são fundamentais para a análise de desempenho.

## Conclusão
O EdgeAwareGNN representa um passo significativo na previsão de carga em redes backbone, demonstrando o potencial dos modelos GNN sensíveis a arestas. Encorajamos a comunidade a explorar, estender e aplicar este trabalho em outros contextos de redes complexas, contribuindo para o avanço das técnicas de previsão baseadas em GNN.

