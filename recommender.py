# Sistema de Recomendação

import csv
import operator
import math
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn import model_selection
from sys import argv


# Carrega datset CSV
def load_Dataset(Filename):
	data_opened = open(Filename)
	data = csv.reader(data_opened)
	data_list = list(data)

	return data_list

# Mapeamento por Filmes
def MapMoviesUser(rating):
	listData = load_Dataset(rating)

	MovieRatings = {}

	# Prepara os dados ordenando por MovieID
	Movies = sorted(listData, key=lambda x: x[1])

	# Mapeando cada Filme com usuário e rating
	Movie_prev = 0;

	for u in Movies:
		userID = u[0]
		movieID = u[1]
		movieRating = u[2]

		# Se o ID do usuário for igual ao anterior, apenas adiciona no dict a avaliação: {Movie : Rating}
		if(Movie_prev == movieID):
			MovieRatings[movieID][userID] = movieRating
			u_prev = userID
		else:
			MovieRatings[movieID] = {}
			MovieRatings[movieID][userID] = movieRating
			Movie_prev = movieID

	# Formato do retorno: {MovieID : {UserID : Rating,
	# 								 UserID : Rating, ...
	# 								},
	# 					   MovieID2 : {UserID : Rating,
	# 								  UserID : Rating, ...
	#  								}
	# 					   }
	return MovieRatings


# Mapeamento por Usuário
def MapUserMovies(rating):
	listData = load_Dataset(rating)

	userRatings = {}

	# Prepara os dados ordenando por UserID
	users = sorted(listData, key=operator.itemgetter(0))

	# Mapeando cada preferência de filme para cada usuário
	u_prev = 0;

	for u in users:
		userID = u[0]
		movieID = u[1]
		movieRating = u[2]

		# Se o ID do usuário for igual ao anterior, apenas adiciona no dict a avaliação: {Movie : Rating}
		if(u_prev == userID):
			userRatings[userID][movieID] = movieRating
			u_prev = userID
		else:
			userRatings[userID] = {}
			userRatings[userID][movieID] = movieRating
			u_prev = userID

	# Formato do retorno: {UserID1 : {MovieID : Rating,
	# 								 MovieID : Rating, ...
	# 								},
	# 					   UserID2 : {MovieID : Rating,
	# 								  MovieID : Rating, ...
	#  								}
	# 					   }
	return userRatings


# Recupera informações do usuário
def get_info_users(userID, userList):
	for u in userList:
		user = u[0]
		Sex = u[1]
		Age = u[2]

		if int(user) == int(userID):
			break;

	return user, Sex, Age


def ListTop10Movie(dataset_list, userID):
	dataset_loaded = load_Dataset(dataset_list)

	Movie_dataset = load_Dataset('movies.csv')

	dataset_loaded = sorted(dataset_loaded, key=operator.itemgetter(2))
	dataset_loaded.reverse()

	var_control = 0

	for i in dataset_loaded:
		user = i[0]
		movie = i[1]
		rating = i[2]

		if int(user) == int(userID) or int(userID) == 0:
			for i2 in Movie_dataset:
				if movie == i2[0] and var_control < 10:
					print(i2[1], i2[2], '###', rating)
					var_control += 1


# Mapeando cada filme avaliado por usuários e transformando o userRatings para item based  
def transposeRankings(ratings):
	transposed = {}
	for user in ratings:
		for item in ratings[user]:
			transposed.setdefault(item, {})
			transposed[item][user] = ratings[user][item]

	# Formato do retorno: {MovieID : {UserID : Rating,
	# 								 UserID : Rating, ...
	# 								},
	# 					   MovieID2 : {UserID : Rating,
	# 								  UserID : Rating, ...
	#  								}
	# 					   }
	return transposed


# Calculando a similaridade usando Correlação Pearson 
def sim_pearson(ratings, user_1, user_2):
	similarity = {}
	for item in ratings[user_1]:
		if item in ratings[user_2]:
			similarity[item] = 1

	# Quantidade de Movies similares entre os usuários foi identificado
	numSim = len(similarity)
	#print(similarity)
	# Se não houver similaridade, retorna 0 
	if numSim == 0:
		return 0

	# Obtem o rating dado pelos usuários.
	userOneSimArray = ([ratings[user_1][i] for i in similarity])
	userOneSimArray = map(int, userOneSimArray)
	# Soma total dos ratings
	sum_1 = sum(userOneSimArray)

	userTwoSimArray = ([ratings[user_2][i] for i in similarity])
	userTwoSimArray = map(int, userTwoSimArray)
	#print(userTwoSimArray)

	sum_2 = sum(userTwoSimArray)

	# Soma dos quadrados
	sum_1_sq = sum([pow(int(ratings[user_1][i]),2) for i in similarity])
	sum_2_sq = sum([pow(int(ratings[user_2][i]),2) for i in similarity])

	# Soma dos produtos (mutiplicação)
	productSum = sum([int(ratings[user_1][i]) * int(ratings[user_2][i]) for i in similarity])
	num = productSum - (sum_1*sum_2/numSim)
	den = math.sqrt((sum_1_sq - pow(sum_1,2)/numSim) * (sum_2_sq - pow(sum_2,2)/numSim))

	if den == 0:
		return 0
	r = num / den
	return r


# Calculando a similaridade Jaccard
def compute_jaccard_similarity(ratings, user_1, user_2):
	# similarity = {}
	# for item in ratings[user_1]:
	# 	if item in ratings[user_2]:
	# 		similarity[item] = 1

	# numSim =len(similarity)

	# if numSim == 0:
	# 	return 0

	userOneRatingsArray = ([ratings[user_1][item] for item in ratings[user_1]])
	userOne = set(userOneRatingsArray)
	userTwoRatingsArray = ([ratings[user_2][item] for item in ratings[user_2]])
	userTwo = set(userTwoRatingsArray)

	return (len(set(userOne.intersection(userTwo))) / float(len(userOne.union(userTwo))))


# Calculando a similaridade Cosine  
def compute_cosine_similarity(ratings, user_1, user_2):
	similarity = {}
	for item in ratings[user_1]:
		if item in ratings[user_2]:
			similarity[item] = 1

	numSim =len(similarity)

	if numSim == 0:
		return 0

	userOneRatingsArray = ([ratings[user_1][s] for s in similarity])
	userOneRatingsArray = list(map(int, userOneRatingsArray))

	userTwoRatingsArray = ([ratings[user_2][s] for s in similarity])
	userTwoRatingsArray = list(map(int, userTwoRatingsArray))

	sum_xx, sum_yy, sum_xy = 0,0,0

	for i in range(len(userOneRatingsArray)):
		x = list(userOneRatingsArray)[i]
		y = list(userTwoRatingsArray)[i]
		sum_xx += x*x
		sum_yy += y*y
		sum_xy += x*y

	return sum_xy/math.sqrt(sum_xx*sum_yy)


# Teste: Calculando a similaridade
def closeMatches(ratings, person, similarity):
	first_person = person
	scores = [(similarity(ratings, first_person, second_person), second_person) for second_person in ratings if second_person != first_person]
	scores.sort()
	scores.reverse()
	# Retorno: ( Similaridade, Item(Movie) )
	return scores


# Teste: Item based collaborative filtering
def similarItems(ratings, similarity):
	itemList = {}

	itemsRatings = transposeRankings(ratings)

	file = open('TesteSimilarItens.txt', 'a')

	c = 0
	for item in itemsRatings:
		c = c + 1
		if c == 10:
			break;

		matches = closeMatches(itemsRatings, item, similarity)
		itemList[item] = matches

		file.write(str(itemList) + '\n')
	# Retorno: dictionary {'MovieID' : [( Similaridade, 'MovieID_2_Similar' )] }
	return itemList


# Recomendações para um usuário, com base no peso dos ratings de outros usuário
def userBasedRecommendations(ratings, wantedPredictions, similarity):
	file = open('userBasedRecomendationsResult.csv', 'a')

	# Recupera o dataset de usuários somente uma única vez, fora do loop para otimização
	userlist = load_Dataset('users.csv')
	userList = sorted(userlist, key=operator.itemgetter(0))

	for tuple in wantedPredictions:
		user = tuple[0]
		movieAsked = tuple[1]

		ranks = {}
		total = {}
		similaritySums = {}

		usuID, Sex, Idade = get_info_users(user, userlist)

		control = -1

		# Percorre a chave (que é o UserID) do dict
		for second_person in ratings:
			# Se os usuários forem iguais, pula para o proximo usuário do FOR.
			# Instrução continue interrompe a execução do ciclo e avança para o próximo item da execução
			if second_person == user:
				continue

			# Aplica o controle de similaridade somente comparando usuários que possuem a mesma faixa de idade e sexo.
			if second_person != control:
				second_usuID, second_Sex, second_Idade = get_info_users(second_person, userlist)
				control = second_person
				if Sex != second_Sex or Idade != second_Idade:
					continue

			s = similarity(ratings, user, second_person)

			if s <= 0:
				continue
			# Percorre os filmes do usuario
			for item in ratings[second_person]:
				if item not in ratings[user] or ratings[user][item] == 0:
					total.setdefault(item, 0)
					total[item] += int(ratings[second_person][item])*s
					similaritySums.setdefault(item, 0)
					similaritySums[item] += s
					ranks[item] = total[item]/similaritySums[item]

		if movieAsked in ranks.keys():
			file.write(str(user)+','+str(movieAsked)+','+str(ranks[movieAsked])+'\n')


# Recomendações para um usuário, com base no peso dos ratings dado pelo próprio usuário em filmes correlacionados
def itemBasedRecommendations(ratings, wantedPredictions, similarity):
	file = open('itemBasedRecomendationsResult.csv', 'a')

	itemsRatings = transposeRankings(ratings)

	for tuple in wantedPredictions:
		user = tuple[0]
		movieAsked = tuple[1]

		uRatings = ratings[user]
		scores = {}
		total = {}
		ranks = {}

		for (Movie, rating) in uRatings.items():
			# usuário já avaliou o filme.
			if movieAsked == Movie:
				continue

			s = similarity(itemsRatings, movieAsked, Movie)

			if s <= 0:
				continue

			scores.setdefault(Movie, 0)
			scores[Movie] += s * int(rating)

			# Soma das similaridades
			total.setdefault(Movie, 0)
			total[Movie] += s

			if total[Movie] == 0:
				ranks[Movie] = 0
			else:
				ranks[Movie] = scores[Movie] / total[Movie]

		if Movie in ranks.keys():
			file.write(str(user)+','+str(movieAsked)+','+str(ranks[Movie])+'\n')


# Combinação de item based e user bases. Isso chama-se: Content - Boosted Collaborative Filtering
def itemBasedRecommendationsForCBCF(ratings, wantedPredictions, similarity):
	file = open('Item-UserBasedRecomendationsResult.txt', 'a')

	for user in ratings:
		uRatings = ratings[user]
		scores = {}
		total = {}
		#ranks = {}


		# Itens avaliados pelo usuário
		for(item, rating) in uRatings.items():
		# Itens que são similares a esse
			for(similarity,item_2) in itemToMatch[item]:
			# Não considera se o usuário já avaliou este item
				if item_2 in uRatings:
					uRatings[item_2] = uRatings[item_2]
				else:
					scores.setdefault(item_2, 0)
					scores[item_2] += similarity*int(rating)

					# Soma das similaridades
					total.setdefault(item_2,0)
					total[item_2] += similarity
					if total[item_2] == 0:
						uRatings[item_2] = 1
					else:
						uRatings[item_2] = scores[item_2]/total[item_2]

	return ratings

# Habilite os comandos abaixo, caso queira fazer algum teste específico
# itemBasedRecommendations(userRatings, simItems, toBeRatedList)
# userBasedRecommendations(userRatings, toBeRatedList, compute_cosine_similarity)
# userBasedRecommendations(userRatings, '3371', sim_pearson)
# itemBasedReco = itemBasedRecommendationsForCBCF(userRatings, simItems)
# userRecosBasedOnDenseMatrix = userBasedRecommendations(itemBasedReco, toBeRatedList, compute_cosine_similarity)

def mainFunction():
	fileName = argv[1]
	similarityName = argv[2]
	Movie = argv[3]
	KindOfRecommendation = argv[4]

	if fileName == 'Top10MoviesUserBased':
		ListTop10Movie('userBasedRecomendationsResult.csv', Movie)
		return;
	elif fileName == 'Top10MoviesItemBased':
		ListTop10Movie('itemBasedRecomendationsResult.csv', Movie)
		return;

	if Movie == 'null':
		MovieSuggest = ''
	else:
		MovieSuggest = Movie.strip('][').split(',') # Trasnforma o parametro de entrada em uma Lista

	print("#######################################################################")
	print("#### Aplicando o mapeamento de UserID-Movie-Rating ####")
	print("#######################################################################\n")
	userRatings = MapUserMovies(fileName)

	# Dados que deverão ser sugeridos Filmes
	if MovieSuggest == '':
		wantedPredictions = load_Dataset("toBeRated.csv")
	else:
		wantedPredictions = []
		wantedPredictions.append(MovieSuggest)

	print('Medida de Similaridade Escolhida: %s \n' % similarityName)

	if similarityName == 'cosine':
		sim = compute_cosine_similarity
	elif similarityName == 'pearson':
		sim = sim_pearson
	else:
		sim = compute_jaccard_similarity

	if KindOfRecommendation == 'userBased':
		print("#######################################################################")
		print("#### Avaliação Escolhida: UserBased ####")
		print("O resultado será um rank entre 0 e 5 baseado na correlação")
		print("e avaliação que outros usuário deram para o respectivo filme.")
		print("#######################################################################\n")
		userBasedRecommendations(userRatings, wantedPredictions, sim)
	elif KindOfRecommendation == 'itemBased':
		print("#######################################################################")
		print("#### Avaliação Escolhida: ItemBased ####")
		print("O resultado será um rank entre 0 e 5 baseado na correlação")
		print("e avaliações que o próprio usuário fez em filmes relacionados.")
		print("#######################################################################\n")
		itemBasedRecommendations(userRatings, wantedPredictions, sim)
	elif KindOfRecommendation == 'userItemBased':
		print("#######################################################################")
		print("#### Avaliação Escolhida: UserItemBased ####")
		print("O resultado será um rank entre 0 e 5 baseado na correlação")
		print("e avaliações que o próprio usuário fez em filmes relacionados.")
		print("#######################################################################\n")
		itemBasedRecommendationsForCBCF(userRatings, wantedPredictions, sim)

mainFunction()