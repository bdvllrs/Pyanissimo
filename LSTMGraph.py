# coding: utf-8

import numpy as np
from theano import function, shared, scan, config, scan_module
from theano.ifelse import ifelse
import theano.printing as pr 
import theano.tensor as T
import pickle


class LSTMGraph:
    def __init__(self, nb_inputs=None, learning_rate=0.01, bptt_truncate=-1, debug=False):
        """
        Initialise le réseau
        """
        self.nb_inputs = nb_inputs
        self.learning_rate = learning_rate
        self.bptt_truncate = bptt_truncate
        self.debug = debug
        self.print_callback = pr._print_fn
        self.train = lambda: print('Utiliser init_train pour activer cette fonction.')
        self.cost = lambda: print('Utiliser init_train pour activer cette fonction.')
        self.BPTT = lambda: print('Utiliser init_train pour activer cette fonction.')
        self.predict = lambda: print('Utiliser init_train pour activer cette fonction.')

        self.layers = []  # Contient toutes les couches : tailles, poids, biases
        self.add_input_layer(self.nb_inputs)  # On ajoute la première couche

        self.predict_stopping_condition = lambda a, b: T.ge(a[-1], 0.6)

    def add_layer(self, length, layer_type, **params):
        """
        Ajoute une couche dans la liste des couches
        :param length: nombre de neurones pour la couche
        :param layer_type: type de couche (input, lstm ou simple)
        :param params: dict de paramètres, possibles 'activation_function' (par défaut 'sigmoid', peut aussi valoir 'tanh')
        """
        params_default = {
            'activation_function': 'sigmoid'
        }
        params_default.update(params)
        layer = {
            'length': length,
            'type': layer_type,
            'params': params_default,
            'weights': {},
            'biases' : {}
        }
        self.layers.append(layer)
        return len(self.layers)-1

    def add_input_layer(self, nb_inputs):
        self.add_layer(nb_inputs, 'input_layer')  # Pas de poids ni rien pour la première couche
        return self

    def add_simple_layer(self, length, weights=None, biases=None, **params):
        """
        Ajoute une couche simple
        :param length: taille de la couche
        :param weights: poids à donner (défaut générés aléatoirements)
        :param biases: poids à donner aux biases
        :param params: possibles : activation_function (defaut 'sigmoid' | 'tanh')
        """
        nb = self.add_layer(length, 'simple')
        self.init_simple_layer_weights(nb, weights, biases)
        return self

    def add_lstm_layer(self, length, weights=None, biases=None, **params):
        """
        Ajoute une couche LSTM
        :param length: taille de la couche
        :param weights: poids à donner (défaut générés aléatoirements)
        :param biases: poids à donner aux biases
        """
        nb = self.add_layer(length, 'lstm')
        self.init_lstm_layer_weights(nb, weights, biases)
        return self

    def error(self, expected, out):
        """
        :param expected: valeur attendue
        :param out: valeur obtenue
        """
        return 0.5*T.sqr(expected - out)

    def debug_print(self, text):
        """
        Affiche un message de débuggage que si le mode debug est activé
        :param text: texte à afficher
        """
        if self.debug:
            print(text)

    def init_graph(self):
        """
        Créé la fonction d'apprentissage
        :param T: longueur des inputs
        """
        self.debug_print('Initialisation du graphe...')

        x = T.dmatrix()  # On créé un vecteur d'entrée de type double
        expected = T.dmatrix('expected')  # Valeur attendue
        learning_rate = T.scalar('learning_rate')  # vitesse d'apprentissage

        outputs_info = [None] # paramètres évolutifs à fournir au model_layers, None correspond à la sortie finale du réseau qui n'a pas de valeur initiale

        for layer in self.layers:
            if layer['type'] == 'lstm':
                outputs_info.append(T.zeros(layer['length']))  # h_prev
                outputs_info.append(T.zeros(layer['length']))  # c_prev

        self.debug_print('Définition de la structure des couches.')

        outputs, updates = scan(  # On fait une boucle sur le model (dans le temps)
            fn=self.model_layers,  # fonction appliqué à chaque étape
            sequences=x,  # on boucle sur le nombre d'entrée
            truncate_gradient=self.bptt_truncate,  # Nombre d'étape pour le truncate BPTT (backpropagation through time)
                                                    # Si égale à -1, on utilise le BPTT classique
            outputs_info=outputs_info # Initialisation des paramètres données à fn
        )

        last_output = outputs[0]  # Sortie finale, les autres sont les valeurs des mémoires intermédiaires

        self.debug_print('Définition de la fonction d\'erreur.')

        cost = T.sum(self.error(expected, last_output))  # Calul de l'erreur avec fonction d'entropie

        self.debug_print('Calcul des gradients.')

        updates = []  # Modifs
        grads = []  # Gradients

        for k in range(len(self.layers)):
            layer_type = self.layers[k]['type']
            if layer_type != 'input': # pas de gradient pour l'entrée
                for w in self.layers[k]['weights'].keys():
                    grads.append(T.grad(cost, self.layers[k]['weights'][w]))
                    update = (self.layers[k]['weights'][w], self.layers[k]['weights'][w] - learning_rate * grads[-1])
                    updates.append(update)  # Liste des modifs à faire pour la propagtion du gradient

        self.debug_print('Compilation des fonctions BPTT et cost.')

        self.BPTT = function([x, expected], grads)
        self.cost = function([x, expected], cost)

        self.debug_print('Compilation de la fonction de prédiction.')

        input = T.dvector()
        arret = T.dvector()
        max_temps = T.iscalar()

        outputs_info = [input] # paramètres évolutifs à fournir au model_layers, None correspond à la sortie finale du réseau qui n'a pas de valeur initiale

        for layer in self.layers:
            if layer['type'] == 'lstm':
                outputs_info.append(T.zeros(layer['length']))  # h_prev
                outputs_info.append(T.zeros(layer['length']))  # c_prev

        o, updated = scan(  # On fait une boucle sur le model (dans le temps)
            fn=self.model_layers_predict,  # fonction appliqué à chaque étape
            non_sequences=arret,  
            outputs_info=outputs_info, # Initialisation des paramètres données à fn
            n_steps=max_temps
        )
        o = o[0]

        self.predict = function([input, arret, max_temps], o)  # Donne la prédiction à partir de l'entrée

        self.debug_print('Compilation de la fonction d\'apprentissage.')

        # Création de la fonction d'entrainement
        self.train = function([x, expected, learning_rate], [cost, last_output], updates=updates)

        self.debug_print('Graphe initialisé.')

        return self

    def generate_weights(self, rows, cols):
        """
        Génère des poids de taille donnée
        """
        spread = .1/np.sqrt(cols)
        w = np.random.uniform(-spread, spread, (rows, cols))
        if rows == 1:
            w = w.flatten()
        return w

    def init_simple_layer_weights(self, layer, weights=None, biases=None):
        """
        Définie les poids pour une couche classique
        """
        """
        Définie les poids pour une couche LSTM
        :param layer: numéro de la couche
        :param weights: poids à initialiser
        :param biases: biases à initialiser
        """
        nb_inputs = self.layers[layer-1]['length']
        nb_neurons = self.layers[layer]['length']

        if weights is None:  # Si rien de renseigné, on génère les poids aléatoirement
            weights = {
                'W': self.generate_weights(nb_inputs, nb_neurons)
            }
        if biases is None:  # Et là les biases
            biases = {
                'b': self.generate_weights(1, nb_neurons)
            }
        # On initialise les variables hybrides de theano
        for w, weight in weights.items():
            self.layers[layer]['weights'][w] = shared(weight.astype(config.floatX))
        for b, biase in biases.items():
            self.layers[layer]['biases'][b] = shared(biase.astype(config.floatX))

    def init_lstm_layer_weights(self, layer, weights=None, biases=None):
        """
        Définie les poids pour une couche LSTM
        :param layer: numéro de la couche
        :param weights: poids à initialiser
        :param biases: biases à initialiser
        """
        nb_inputs = self.layers[layer-1]['length']
        nb_neurons = self.layers[layer]['length']

        if weights is None:  # Si rien de renseigné, on génère les poids aléatoirement
            weights = {
                'Uf': self.generate_weights(nb_neurons, nb_neurons),
                'Wf': self.generate_weights(nb_inputs, nb_neurons),
                'Ui': self.generate_weights(nb_neurons, nb_neurons),
                'Wi': self.generate_weights(nb_inputs, nb_neurons),
                'Uo': self.generate_weights(nb_neurons, nb_neurons),
                'Wo': self.generate_weights(nb_inputs, nb_neurons),
                'Uc': self.generate_weights(nb_neurons, nb_neurons),
                'Wc': self.generate_weights(nb_inputs, nb_neurons)
            }
        if biases is None:  # Et là les biases
            biases = {
                'bf': self.generate_weights(1, nb_neurons),
                'bi': self.generate_weights(1, nb_neurons),
                'bo': self.generate_weights(1, nb_neurons),
                'bc': self.generate_weights(1, nb_neurons)
            }
        # On initialise les variables hybrides de theano
        for w, weight in weights.items():
            self.layers[layer]['weights'][w] = shared(weight.astype(config.floatX))
        for b, biase in biases.items():
            self.layers[layer]['biases'][b] = shared(biase.astype(config.floatX))

    def model_layers(self, x, *args):
        """
        Définie le model pour toutes les couches
        :param x: entrée
        :param args: liste des entrées (du temps précédent) pour les lstm
        """
        args = list(args)
        num_arg = 0
        outputs = []
        # debug_printing = self.print_callback('Activation du réseau...')
        for k in range(len(self.layers)):
            layer = self.layers[k]
            # res = debug_printing(k)
            if layer['type'] == 'simple':  # si c'est un simple on utilise le model simple
                x = self.model_simple_layer(x, k)
            elif layer['type'] == 'lstm':  # sinon celui de lstm
                h, c = self.model_lstm_layer(x, args[num_arg], args[num_arg+1], k)
                x = h  # la sortie est h
                num_arg += 2
                outputs.append(h)  # nouveau h
                outputs.append(c)  # nouveau x
        outputs = [x] + outputs  # les sorties sont la sortie finale x et les valeurs intermédiaires à repasser au réseau au temps suivant
        return tuple(outputs)  # x (output), vals_t, ...

    def model_layers_predict(self, x, *args):
        """
        Définie le model pour toutes les couches
        :param x: entrée
        :param args: liste des entrées (du temps précédent) pour les lstm
        """
        args = list(args)
        stop_condition = args[-1]
        outputs = list(self.model_layers(x, *args))
        # cond = T.eq(T.argmax(x), T.argmax(stop_condition))
        cond = self.predict_stopping_condition(x, stop_condition)
        return tuple(outputs), scan_module.until(cond)  # x (output), vals_t, ..., condition d'arret

    def model_simple_layer(self, x, num_layer):
        """
        Model pour une couche simple
        :param x: entrée
        :param num_layer: le numéro de la couche à traiter
        """
        if self.layers[num_layer]['params']['activation_function'] == 'tanh':
            return T.tanh(T.dot(x, self.layers[num_layer]['weights']['W']) + self.layers[num_layer]['biases']['b'])
        return T.nnet.sigmoid(T.dot(x, self.layers[num_layer]['weights']['W']) + self.layers[num_layer]['biases']['b'])

    def model_lstm_layer(self, x, h_prev, c_prev, num_layer):
        """
        Model pour une couche LSTM
        :param x: entrée
        :param h_prev: sortie de la couche au temps précédent
        :param c_prev: mémoire de la couche au temps précédent
        :return: memoire de cette couche, sortie de cette couche, sortie du reseau
        """
        # Forget gate
        f = T.nnet.sigmoid(T.dot(x, self.layers[num_layer]['weights']['Wf']) + T.dot(h_prev, self.layers[num_layer]['weights']['Uf']) + self.layers[num_layer]['biases']['bf'])
        # Input gate
        i = T.nnet.sigmoid(T.dot(x, self.layers[num_layer]['weights']['Wi']) + T.dot(h_prev, self.layers[num_layer]['weights']['Ui']) + self.layers[num_layer]['biases']['bi'])
        # Output gate
        o = T.nnet.sigmoid(T.dot(x, self.layers[num_layer]['weights']['Wo']) + T.dot(h_prev, self.layers[num_layer]['weights']['Uo']) + self.layers[num_layer]['biases']['bo'])
        # Mémoire
        c = f * c_prev + i * T.tanh(T.dot(x, self.layers[num_layer]['weights']['Wc']) + T.dot(h_prev, self.layers[num_layer]['weights']['Uc']) + self.layers[num_layer]['biases']['bc'])
        h = o * T.tanh(c)
        return h, c

    def save_weights(self, file):
        """
        Sauvegarde les poids dans un fichiers
        """
        layers = []
        for layer in self.layers:
            layers.append({
                'length': layer['length'],
                'weights': {},
                'biases': {},
                'type': layer['type'],
                'params': layer['params']
            })
            for w, val in layer['weights'].items():
                layers[-1]['weights'][w] = val.get_value()
            for w, val in layer['biases'].items():
                layers[-1]['biases'][w] = val.get_value()
        pickle.dump(layers, open(file, 'ab'))
        return self

    def load_weights(self, layer, file):
        """
        Utilise les fichiers pour initialiser les poids
        """
        layers = pickle.load(open(file, 'rb'))
        self.layers = []
        for layer in layers:
            if layer['type'] == 'lstm':
                self.add_lstm_layer(layer['length'], layer['weights'], layer['biases'])
            elif layer['type'] == 'simple':
                self.add_simple_layer(layer['length'], layer['weights'], layer['biases'])
        return self

    def set_bptt_truncate(self, nb_truncate=-1):
        """
        Change le nombre de troncature pour faire une BPTT truncate (défaut pas de troncature)
        :param nb_truncate: nombre de troncature
        """
        self.bptt_truncate = nb_truncate
        return self

    def set_learning_rate(self, learning_rate):
        """
        Change le taux d'apprentissage
        :param learning_rate: taux d'apprentissage
        """
        self.learning_rate = learning_rate
        return self

    def set_nb_input(self, size):
        """
        Définie la taille de l'entrée
        :param size: taille
        """
        self.nb_inputs = size
        return self

    def set_print_callback(self, fn):
        """
        Change printing callback in theano
        :param fn: function to use fn(op, xin)
        """
        self.print_callback = fn
        return self

    def set_predict_stopping_condition(self, fn):
        """
        Change theano stopping condition
        :param fn: function f(a, b)
        """
        self.predict_stopping_condition = fn
        return self