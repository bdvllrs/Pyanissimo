from LSTM import LSTM
import numpy as np

nb_inputs = 2  # Nombre de paramètre en entrée
learning_rate = 0.1  # Vitesse d'apprentissage
debug = True  # Affiche des infos sur l'avancement des paramètrages

lstm = LSTM(nb_inputs, learning_rate, debug)
lstm.add_lstm_layer(5)  # Ajoute une couche lstm de 5 neurones
lstm.add_simple_layer(2)  # Ajoute une couche simple, la dernière sera la couche de sortie
lstm.init_graph()  # Initialise le graph de calcul (peut-être long car calcul des dérivées et compilation de fonction predict, cost, BPTT et train)
lstm.add_progress_callback(1)  # Ajoute une information sur l'avancement tous les 1 exemples traités
lstm.add_snapshot_callback(1)  # Enregistre les poids tous les 1 exemples traites
# lstm.add_callback(callback, period)  # Ajoute un callback personnel, callback(cost, out, count, total)
# cost est l'erreur au moment du callback
# out est la sortie du réseau au moment du callback
# count est le numéro de l'exemple en cours
# total est le nombre d'exemple total à traiter

## Définition des entrées et sorties

x1 = np.asarray([[1, 5], [2, 5], [3, 4]])  	# A chaque étape dans le temps, on donne un vecteur de 2 entrées ici car nb_inputs = 2
											# (3 étapes dans le temps)
x2 = np.asarray([[2, 5], [1, 2]])  # Ici, on a 2 étapes dans le temps (peut être différent entre les différentes entrées)

y1 = np.asarray([[0.1, 0.8], [0.2, 0.3], [0.4, 0.6]])  # Sorties attendues pour x1 (3 étapes dans le temps de sortie de 2 entrées car nb_outputs = 2)
y2 = np.asarray([[0.8, 0.1], [0.1, 0.8]])  # Sorties attendues pour x2

x_train = [x1, x2]  # Ensembles des exemples à traiter
y_train = [y1, y2]

epochs = 1  # Nombre de fois que l'on réapprend chaque exemple (à chaque epoch, l'ordre d'apprentissage de x_train est différent)

lstm.train(x_train, y_train, epochs)