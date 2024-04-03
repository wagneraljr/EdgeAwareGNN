# AttEdgeAwareGNN: Previsão de Carga em Redes Backbone com GNN Sensível a Arestas

## Visão Geral
### O Projeto
O projeto EdgeAwareGNN introduz um modelo inovador de Rede Neural de Grafos (GNN) sensível a arestas aplicado à a previsão de carga em redes *backbone*, nomeado AttEdgeAwareGNN. Este projeto visa explorar atributos de arestas como forma de enriquecer as representações de nós geradas por GNNs. Através da comparação com arquiteturas tradicionais de GNNs, o AttEdgeAwareGNN demonstra melhorias significativas na precisão das previsões.

### Rede *Backbone* Abilene
Os modelos são treinados usando dados históricos de tráfego em uma rede backbone, especificamente a rede Abilene. Criada em 1999 e encerrada em 2007, a rede Abilene era composta por 11 nós e 14 links. O conjunto de dados usado neste trabalho contém seis meses de dados de tráfego entre os nós da rede Abilene, medidos a cada cinco minutos.

### Arquitetura dos Modelos
O projeto utiliza múltiplas arquiteturas de GNN, incluindo os tradicionais [GCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) e [GraphSAGE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html), e implementa o nosso novo modelo AttEdgeAwareGNN.

Informações acerca do modelo AttEdgeAwareGNN serão  posteriormente detalhadas, após revisão do artigo.
## Requisitos
* python3
* pip
## Configuração Inicial
Instale as dependências do projeto:
```
pip install -r requirements.txt
```
## Estrutura do Projeto
A estrutura do projeto pode ser compreendida da seguinte forma:
* **checkpoints:** Contém os modelos salvos após treinamento em arquivos *.pth*.
* **datasets:** Mantém os conjuntos de dados utilizados no projeto.
* **graphics:** Responsável por manter os gráficos gerados após etapa de avaliação para comparação dos modelos.
* **hyperparameters_config**: Contém informações acerca dos hiperparâmetros a serem utilizados nos modelos, salvos em arquivos *json*.
* **images:** Mantém imagens relacionadas ao projeto.
* **src:** Possui o código fonte principal.
* **train_results:** Contém informações do treinamento dos modelos, como as *losses* e a predição dos dados de teste após treinamento.

## Reprodutibilidade
Para prever as cargas dos nós da rede abilene, os modelos foram treinados em dois cenários: um onde os dados de treinamento contém informações da carga da rede durante um dia, e outro durante uma semana. Para que fosse possível a reprodução dos resultados obtidos, uma *seed* foi salva para cada modelo em cada cenário. Posteriomente, instruções de uso de *scripts* prontos para reprodução do artigo serão abordadas.

### Fluxo de Treinamento e Avaliação dos Modelos
O treinamento e avaliação dos modelos consiste das seguintes etapas:

1. Otimizações dos hiperparâmetros dos modelos através da biblioteca [optuna](https://optuna.org/) (mais detalhes sobre esta etapa em [Otimização de Hiperparâmetros](#otimização-de-hiperparâmetros)). 
2. Configuração manual dos hiperparâmetros obtidos em um arquivo *json* (podem ser exploradas na pasta `hyperparameters_config`).
3. Criação e treinamento dos modelos a partir dos hiperparâmetros obtidos.
    
    a. Ao final do treinamento, as *losses* e predições obtidas no arquivo de teste são exportadas para um arquivo *json* em `train_results`.
4. Avaliação dos modelos com base nas métricas métricas MAE (Erro Médio Absoluto), MSE (Erro Quadrático Médio) e R²  (Coeficiente de Determinação) a partir dos dados salvos em `train_results`, além da geração de gráficos com os resultados das avaliações e treinamento dos modelos para simples comparação.

Este fluxo é repetido para cada cenário.

### Reprodução dos Experimentos
Para simplificar a reprodução dos experimentos foi disponibilizado um shell *script* (`run_experiments.sh`) na pasta raíz do projeto. Este *script* executará o script python `train.py` para cada modelo em cada cenário, onde ao final de cada cenário será executado o *script* `generate_graphics.py` para avaliação do modelo e geração dos gráficos. O progresso de treinamento pode ser acompanhado através do terminal. 

Para executar o shell *script* certifique-se que ele tenha permissão para execução, e caso positivo, execute:
```
./run_experiments.sh
```

Caso este arquivo não tenha permissão para execução, execute o comando abaixo e logo após o comando anterior:
```
chmod +x run_experiments.sh
```

## Otimização de Hiperparâmetros

A otimização de hiperparâmetros é realizada pelos scripts `hyper_day.py` e `hyper_week.py`, que ajustam os modelos usando dados de um dia e uma semana anteriores à data alvo da previsão, respectivamente. Esses scripts empregam a biblioteca `optuna` para encontrar a configuração ideal de hiperparâmetros que maximiza a precisão das previsões. Este processo é crucial para garantir que os modelos estejam bem ajustados às características específicas dos dados de tráfego.

