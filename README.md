# Sistema-Recomendacao-Filmes
Mini Projeto 2 desenvolvido ao final do módulo 16 do curso de Machine Learning da Formação Cientista de Dados da Data Science Academy (DSA).

Neste projeto foi construído um sistema de recomendação de filmes totalmente automatizado e com cada função  desenvolvida  em  linguagem  Python. Você encontra ainda o código necessário para calcular a similaridade pelo coeficente de Pearson, a similaridade cosine e a similaridade Jaccard. A medida de similaridade será um dos parâmetros a ser usado pelo sistema de recomendação. Além disso, foi feito a unificação de duas técnicas de filtro colaborativo, item based e user based para trabalhar com a técnica Boosted Collaborative Filtering, formando na verdade um sistema de recomendação híbrido.

O dataset de entrada possui os ratings (avaliações) de usuários a uma série de filmes. 
O objetivo do sistema é processar esses dados e gerar como saída a lista de usuários e os filmes que devem ser recomendados. 

Esse não é um sistema online e seu processamento leva algumas horas para o processamento FULL.

# EXECUÇÃO:

- 1- Abra um terminal ou prompt de comando.
- 2- Navegue até o diretório onde estão os arquivos que você baixou.
- 3- Execute o aplicativo escolhendo o dataset e o algoritmo de similaridade a ser utilizado.

# MODELO EXECUÇÃO:

$ python recommender.py ratings.csv pearson [3374,673] userBased

- python – nome do interpretador
- recommender.py – nome do seu aplicativo Python (nome do script)
- ratings.csv – nome do arquivo comos ratings dos usuários
- pearson – medida de similaridade (pode ser ainda cosine ou Jaccard e o script está preparado para executar com uma das 3 medidas)
- [3374,673] - Lista UserID : MovieID para teste indivual (passando o valor null é executado para todos os dados)
- userBased - Tipo de técnica utilizada para recomendação (pode ser ainda: itemBased ou userItemBased)

# TO DO:

- Finalizar a implementação da técnica userItemBased.
- Listar Top 10 Filmes para recomendação baseado em userBased, ItemBased e userItemBased.
