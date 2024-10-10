# Identificação de Raças de Cães usando TensorFlow e MobileNetV2-Adam

## Visão Geral

Bem-vindo ao projeto de Identificação de Raças de Cães! Este projeto tem como objetivo desenvolver um modelo de aprendizado de máquina para identificar raças de cães a partir de imagens, utilizando TensorFlow e a arquitetura MobileNetV2 com o otimizador Adam. O objetivo é criar um modelo preciso que possa reconhecer várias raças de cães, fornecendo insights sobre a visão canina e auxiliando em aplicações como cuidados com animais de estimação e reconhecimento de raças.

**Explore o projeto diretamente no Google Colab:** [Identificação de Raças de Cães](https://colab.research.google.com/drive/1vI332JRfo66w7emKpQr9cutqkb9uBW8r?usp=sharing)

## Descrição do Projeto

Identificar raças de cães a partir de imagens envolve analisar características visuais específicas de cada raça. Este projeto utiliza técnicas de aprendizado profundo para construir um modelo capaz de reconhecer e classificar raças de cães com base em imagens de entrada. A utilização do TensorFlow e da arquitetura MobileNetV2 garante um treinamento eficiente e alta precisão na identificação de raças.

## Arquitetura do Modelo

A arquitetura do modelo utilizada neste projeto é baseada no MobileNetV2, uma rede neural convolucional (CNN) de última geração otimizada para aplicações de visão em dispositivos móveis e incorporados. O MobileNetV2 oferece um bom equilíbrio entre tamanho do modelo e precisão, tornando-o adequado para implantação em dispositivos com recursos limitados.

## Conjunto de Dados

O conjunto de dados utilizado para treinamento e avaliação consiste em imagens de várias raças de cães. É essencial ter um conjunto de dados diversificado e representativo para garantir a capacidade do modelo de generalizar para dados não vistos de forma eficaz. O conjunto de dados pode incluir raças de cães populares, raças raras e variações de pose, condições de iluminação e fundos para aumentar a robustez do modelo.

Os dados que estamos usando são da Competição de Identificação de Raças de Cães do Kaggle.
**Dados:** https://www.kaggle.com/c/dog-breed-identification/data

## Pré-processamento de Dados

O pré-processamento de dados desempenha um papel crucial na preparação do conjunto de dados para o treinamento. As etapas comuns de pré-processamento podem incluir:

- Redimensionamento de imagens para um tamanho padrão compatível com a entrada do modelo.
- Normalização dos valores dos pixels para uma escala comum (por exemplo, [0, 1]).
- Aumento do conjunto de dados com técnicas como rotação, inversão e alteração de cores para aumentar a variabilidade e melhorar a generalização do modelo.

## Treinamento

O modelo é treinado usando a estrutura TensorFlow com a arquitetura MobileNetV2. O otimizador Adam é empregado para otimizar os parâmetros do modelo e minimizar a perda de classificação. Durante o treinamento, o modelo aprende a extrair características significativas de imagens de entrada e fazer previsões sobre a raça de cães presente em cada imagem.

## Avaliação

A avaliação envolve gerar probabilidades de previsão para cada raça de cão de cada imagem de teste. Essas probabilidades de previsão podem ser comparadas com rótulos verdadeiros para avaliar o desempenho do modelo na classificação precisa de raças de cães. O processo de avaliação ajuda a fornecer insights sobre a eficácia do modelo em identificar raças de cães a partir de imagens.

## Uso

Para utilizar o modelo de Identificação de Raças de Cães, siga estas etapas:

1. Clone o repositório para sua máquina local.

   ```bash
   git clone https://github.com/sarthaklambaa/dog-vision-identification.git
   ```
2. Instale as dependências necessárias, incluindo TensorFlow e outras bibliotecas requeridas.
3. Pré-processe seu conjunto de dados, garantindo o redimensionamento, normalização e aumento adequados.
4. Treine o modelo usando os scripts ou Notebooks Jupyter fornecidos.
5. Avalie o desempenho do modelo em um conjunto de dados de validação separado e ajuste os hiperparâmetros conforme necessário.

Depois de ficar satisfeito com o desempenho do modelo, implemente-o para tarefas de identificação de raças em sua aplicação ou ambiente.

## Arquivos do Projeto

### `main.py`

Este arquivo é responsável por treinar nosso próprio modelo. Ele executa o treinamento utilizando o conjunto de dados disponível e exibe um teste visual do modelo em ação, permitindo visualizar os resultados das previsões realizadas pelo modelo treinado.

### `predict_full_set.py` e `predict_minified_set.py`

Estes arquivos utilizam modelos pré-treinados para realizar previsões e análises. `predict_full_set.py` executa previsões em um conjunto completo de dados, enquanto `predict_minified_set.py` é projetado para trabalhar com um conjunto de dados reduzido, permitindo testar rapidamente a eficácia do modelo em um ambiente controlado.

*podemos encontrar informações detalhadas sobre os modelos, os dados e o gerenciamento completo do ml no arquivo dog_vision.ipynb*