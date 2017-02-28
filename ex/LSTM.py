from LSTMGraph import LSTMGraph
import numpy as np

class LSTM:
    def __init__(self, nb_inputs, learning_rate=0.01, debug=False):
        self.graph = LSTMGraph(nb_inputs, learning_rate, debug=debug)
        self.learning_rate = learning_rate
        self.callbacks = []
        self.callback_periods = []

    def add_callback(self, callback, period):
        """
        Ajoute un callback à appeler toutes les periodes fois
        :param callback: callback (prend en paramètre l'erreur, la sortie, le nombre d'exemple ont déjà été traités et le nombre d'exple total) 
        callback(cost, out, count, total)
        :param period: on appelle le callback tous les period périodes
        """
        self.callbacks.append(callback)
        self.callback_periods.append(period)
        return self

    def add_progress_callback(self, period):
        """
        Ajoute une progression
        :param period: affiche la progression tous les period exemples
        """
        callback = lambda cost, _, count, total: print(100*count/total, '% exécuté, erreur : ', cost, sep='')
        self.add_callback(callback, period)
        return self

    def add_snapshot_callback(self, period):
        """
        Enregistre les poids d'entrainement toutes les periodes
        :param period: période
        """
        callback = lambda _, __, count, total: self.save_weights('data/snapshot' + str(count)+'.dat')
        self.add_callback(callback, period)
        return self

    def add_lstm_layer(self, length, weights=None, biases=None):
        """
        Ajoute une couche LSTM
        :param length: taille de la couche
        :param weights: poids à donner (défaut générés aléatoirements)
        :param biases: poids à donner aux biases
        """
        self.graph.add_lstm_layer(length, weights, biases)
        return self

    def add_simple_layer(self, length, weights=None, biases=None):
        """
        Ajoute une couche simple
        :param length: taille de la couche
        :param weights: poids à donner (défaut générés aléatoirements)
        :param biases: poids à donner aux biases
        """
        self.graph.add_simple_layer(length, weights, biases)
        return self

    def init_graph(self):
        """
        Initialise les fonctions du graphe
        """
        self.graph.init_graph()
        return self

    def load_weights(self, file):
        """
        Récupère les poids d'un fichier pickle
        :param file: fichier
        """
        self.graph.load_weights(file)
        return self

    def save_weights(self, file):
        """
        Sauvegarde les poids dans un fichier picke
        """
        self.graph.save_weights(file)
        return self

    def train(self, x_train, y_train, epochs=1):
        """
        Entraine le réseau
        :param x_train: liste des entrées
        :param y_train: liste des sorties attendues
        :param epochs: nombre de fois 
        """
        count = 0  # On compte le nombre d'exemple on a vu pour le callback
        total = epochs * len(x_train)
        nb_callbacks = len(self.callbacks)
        for epoch in range(epochs): # Nombre de fois qu'on re-entraine avec les mêmes données
            for k in np.random.permutation(len(x_train)):  # On utilise un ordre différent à chaque fois
                x, y = x_train[k], y_train[k]
                cost, out = self.graph.train(x, y, self.learning_rate)
                # On appelle les callbacks ajoutés
                for num_callback in range(nb_callbacks):
                    if count % self.callback_periods[num_callback] == 0:
                        self.callbacks[num_callback](cost, out, count+1, total)
                count += 1
        return self

    def predict(self, x, stop_condition, max_temps):
        """
        Execute le reseau et donne la valeur suivante
        :param x: entrée
        :param stop_condition: valeur de la sortie pour arreter la propadation
        :param max_temps: valeur de tour de boucle max dans tous les cas
        :return: liste de toutes les sorties au cours du temps
        """
        return self.graph.predict(x, stop_condition, max_temps)