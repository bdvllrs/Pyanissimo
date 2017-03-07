# coding: utf-8
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.messagebox
import os
import threading
import file
import time
import random
from LSTM import LSTM


class FileSelector(tk.Frame):
    """
    Widget de selection des fichiers à utiliser
    """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, relief=tk.RIDGE, bd=2)
        self.files = {}
        self.discover_files()
        self.enabled = {}
        self.checks = {}
        for opt in sorted(list(k for k in self.files)):
            self.enabled[opt] = tk.Variable()
            self.checks[opt] = tk.Checkbutton(self, text=opt+' ('+str(len(self.files[opt]))+')', variable=self.enabled[opt], bd=0, state=tk.ACTIVE)
            self.checks[opt].select()
            self.checks[opt].deselect()
            self.checks[opt].pack(anchor=tk.W)

    def discover_files(self):
        """
        Cherche la liste des fichiers présents dans le dossier 'music/format 0'
        """
        # obtient la liste des artistes
        artists = []
        for _, dirs, _ in os.walk('musics/format 0'):
            artists = dirs
            break
        
        # obtient la liste des fichiers disponibles par artiste
        for ar in artists:
            for _, _, files in os.walk('musics/format 0/'+ar):
                if len(files)>0:
                    self.files[ar] = files
                break


class StatusBar(tk.Frame):
    """
    Widget de barre d'état, intégrant une barre de progression et des messages d'état
    """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bd=2, relief=tk.RIDGE)
        
        # barre de progression
        self.progressValue = tk.IntVar()
        self.progressBar = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate', length=80, variable=self.progressValue)
        self.progressBar.pack(side=tk.LEFT)
        
        # message d'état
        self.info = tk.Label(self, text='Initialisation')
        self.info.pack(side=tk.LEFT, anchor=tk.W)

    def update(self, step=-1, msg=None):
        """
        Callback de mise à jour des informations
        """
        if msg:
            self.info.config(text=msg)
        if step != -1:
            clamp = min(max(step,0),100)
            self.progressValue.set(int(clamp))


class Grapher(tk.Canvas):
    """
    Widget dessinant un graphique

    Utilisations types:
        * Ajout d'une courbe toto de type 'simple'
        * Ajout de n points x_i sur cette courbe
        -> la courbe sera le tracé y = f(i) = x_i,
           avec les axes x[0, xmax], y[0, n]

        * Ajout d'une courbe tata de type 'point'
        * Ajout de n points (x_i, y_i) sur cette courbe
        -> la courbe sera le tracé des points reliés (x_i, y_i),
           avec les axes x[xmin, xmax], y[ymin, ymax]
        Dans le cas de plusieurs courbes, on prend l'interval le plus petit contennant toute les courbes
    """
    def __init__(self, parent, width, height, backColor='white'):
        tk.Canvas.__init__(self, parent, width=width, height=height)
        self.backColor = backColor
        self.create_rectangle((0, 0), (width, height), fill=backColor)
        self.curves = {}
        self.absciss = ''

    def add_curve(self, name, color, typ):
        """
        Ajoute une courbe à la liste des courbes
        """
        # vérifie le type
        assert typ in ['simple', 'point'], "Type not recognised"
        # vérifie l'axe des abscisses
        if not self.absciss:
            self.absciss = typ
        assert typ == self.absciss, "Incompatible type"
        # ajoute la courbe
        self.curves[name] = [color, []]

    def add_point(self, curvename, pt, redraw=True):
        """
        Ajoute un point à la courbe 'curvename'
        """
        if curvename not in self.curves:
            return
        self.curves[curvename][1].append(pt)
        if redraw:
            self.redraw()

    def add_points(self, curvename, pointlist, redraw=True):
        """
        Ajoute plusieurs points à la courbe 'curvename'
        """
        for pt in pointlist:
            self.add_point(curvename, pt, False)
        if redraw:
            self.redraw()

    def map_scr(self, pt, xspan, yspan):
        # get size
        width = self.winfo_width()
        height = self.winfo_height()
        # unpack
        x, y = pt
        xmin, xmax = xspan
        ymin, ymax = yspan
        # map coords
        xscr = int(width*(x-xmin)/(xmax-xmin))
        yscr = height - int(height*(y-ymin)/(ymax-ymin))
        return xscr, yscr

    def redraw(self):
        """
        Redessine entièrement le graphique
        """
        # check existence
        if len(self.curves) <= 0:
            return
        elif all(len(self.curves[n][1]) <= 1 for n in self.curves):
            return
        drawing = [n for n in self.curves if len(self.curves[n][2]) > 1]
        # get size
        width = self.winfo_width()
        height = self.winfo_height()
        # get x and y spans
        if self.absciss == 'simple':
            xspan = [0, max(len(self.curves[n][1]) for n in drawing)]
            yspan = [0, max(self.curves[n][1] for n in drawing)]
        elif self.absciss == 'point':
            # take first point of a curve as initialisation point
            first_x, first_y = self.curves[drawing[0]][1][0]
            xspan = [first_x, first_x]
            yspan = [first_y, first_y]
            for n in drawing:
                for x, y in self.curves[n][1]:
                    xspan[0] = min(x, xspan[0])
                    xspan[1] = max(x, xspan[1])
                    yspan[0] = min(y, yspan[0])
                    yspan[1] = max(y, yspan[1])
        # clear previous drawings
        self.delete('ALL')
        # draw background
        self.create_rectangle((0, 0), (width, height), fill=self.backColor)
        # draw curves
        for n in drawing:
            x, y = self.map_scr(self.curves[n][1][0], xspan, yspan)
            for nx, ny in self.curves[n][1][1:]:
                nxs, nys = self.map_scr((nx, ny), xspan, yspan)
                self.create_line((x, y), (nxs, nys), fill=self.curves[n][0])
                x, y = nxs, nys


class TrainingDialog(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)

        # frame de progression
        self.progFrame = tk.Frame(self)
        self.progFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # indicateur global
        self.fileLabel = tk.Label(self.progFrame, text='Fichier:')
        self.fileLabel.pack(anchor=tk.W)
        
        # barre de progression totale
        self.progFileValue = tk.IntVar()
        self.progFileMax = 0
        self.progFile = ttk.Progressbar(self.progFrame, orient=tk.HORIZONTAL, mode='determinate', length=400, variable=self.progFileValue)
        self.progFile.pack(side=tk.TOP)

        # indicateur du fichier actuel
        self.advLabel = tk.Label(self.progFrame, text='Etape:')
        self.advLabel.pack(anchor=tk.W)

        # barre de progression du fichier actuel
        self.progAdvValue = tk.IntVar()
        self.progAdvMax = 0
        self.progAdv = ttk.Progressbar(self.progFrame, orient=tk.HORIZONTAL, mode='determinate', length=400, variable=self.progAdvValue)
        self.progAdv.pack(side=tk.TOP)

        # progression de l'erreur
        self.errorGraph = Grapher(self, 300, 200)
        self.errorGraph.pack(side=tk.RIGHT, anchor=tk.E)
        self.errorGraph.add_curve('error', 'blue', 'simple')
        
    def set_file_qty(self, value):
        """
        Nombre de fichier total à parcourir
        """
        self.progFile.config(maximum=value)
        self.progFileMax = value

    def set_file_max(self, value):
        """
        Taille du fichier à parcourir (en nombre d'échantillons)
        """
        self.progAdv.config(maximum=value)
        self.progAdvMax = value

    def update_file(self, info, num):
        """
        Affiche le numéro et le nom du fichier actuel
        """
        self.fileLabel.config(text=info)
        self.progFileValue.set(num)

    def update_adv(self, state, num):
        """
        Affiche la progression dans le fichier actuel
        """
        self.advLabel.config(text='Etape:'+state)
        self.progAdvValue.set(num)


class NumberEntry(tk.Frame):
    """
    Widget contenant un label et un Entry limité aux nombres,
    avec minval <= x < maxval
    """
    def __init__(self, parent, text='', minval=0, maxval=10000, defaultval=1, numtype='int'):
        tk.Frame.__init__(self, parent)

        self.minval = minval
        self.maxval = maxval
        self.defaultval = defaultval
        self.numtype = numtype

        self.label = tk.Label(self, text=text)
        self.label.pack(side=tk.LEFT)

        validatecommand = self.register(self.validate_entry)
        self.entry = tk.Entry(self, validate='key', validatecommand=(validatecommand, '%P', '%V'))
        self.entry.pack(side=tk.LEFT)
        if self.validate_entry(defaultval, 'init'):
            self.entry.insert(0, str(defaultval))

    def get_value(self):
        """
        Valeur numérique entrée
        """
        if self.numtype == 'int':
            return int(self.entry.get())
        elif self.numtype == 'float':
            return float(self.entry.get())
        raise ValueError('Numeric type not supported')

    def validate_entry(self, newtext, reason):
        """
        Validate a change in the Entry widget
        """
        if reason == 'forced': # changement de la variable interne à l'Entry
            return True

        try:
            if self.numtype == 'int':
                val = int(newtext)
            elif self.numtype == 'float':
                val = float(newtext)
            else:
                return False
        except ValueError:
            return False
        return self.minval <= val < self.maxval


class Interface(tk.Tk):
    """
    Widget principal de l'application
    """
    def __init__(self):
        tk.Tk.__init__(self)
        
        # commandes
        self.cmdFrame = tk.Frame(self)
        self.cmdFrame.grid(row=0, column=0)

        # self.btLoad = tk.Button(self.cmdFrame, text='Charger un réseau')
        # self.btLoad.pack()
        
        # self.btSave = tk.Button(self.cmdFrame, text='Sauver le réseau')
        # self.btSave.pack()

        self.btTrain = tk.Button(self.cmdFrame, text="Commencer l'entrainement", command=self.confirm_train)
        self.btTrain.pack(fill=tk.X, expand=True)

        self.btCreate = tk.Button(self.cmdFrame, text="Ecrire la musique", command=self.start_creation)
        self.btCreate.pack(fill=tk.X, expand=True)

        self.btLoad = tk.Button(self.cmdFrame, text='Charger des poids', command=self.load_weights)
        self.btLoad.pack(fill=tk.X, expand=True)

        self.entryEpoch = NumberEntry(self.cmdFrame, text='Nombre d\'époques', minval=1, maxval=1000, defaultval=10, numtype='int')
        self.entryEpoch.pack()

        self.entrySpeed = NumberEntry(self.cmdFrame, text='Vitesse d\'aprentissage', minval=0., maxval=1., defaultval=0.1, numtype='float')
        self.entrySpeed.pack()
        
        # sélection des fichiers
        self.selectedFiles = FileSelector(self)
        self.selectedFiles.grid(row=0, column=1)
        
        # statut
        self.status = StatusBar(self)
        self.status.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E)

        # entrainement
        self.training = False
        self.stoppingTraining = False
        self.waitingForStop = False
        self.trainingThread = None
        self.trainingDialog = None
        # lecture des données
        self.readingThread = None
        self.dataLock = threading.Lock()
        self.dataQueue = []
        self.dataSpace = 1 # nombre de fichiers chargés en avance
        # écriture d'un nouveau fichier
        self.creationThread = None
        self.creationFinished = True

        self.reseau = LSTM(131, self.entrySpeed.get_value(), True)

    def init_reseau(self, custom_update=None):
        """
        Initialise le réseau LSTM
        """
        def update(t):
            self.status.update(0, 'Initialisation du réseau:'+t)
            print('Initialisation du réseau:'+t)
            if custom_update:
                custom_update(t)
        update('')
        self.reseau.add_simple_layer(200)
        self.reseau.add_lstm_layer(200)
        self.reseau.add_lstm_layer(200)
        self.reseau.add_simple_layer(131)
        self.reseau.graph.debug_print = update
        self.reseau.init_graph()
        update('Terminée')

    def confirm_train(self):
        """
        Affiche une boite de confirmation de démarrage de l'apprentissage
        """
        num = sum(len(self.selectedFiles.files[f]) for f in self.selectedFiles.files if self.selectedFiles.enabled[f].get()=='1')
        resp = tk.messagebox.askquestion('Confirmation', "Êtes-vous sûr de vouloir lancer\nl'apprentissage avec "+str(num)+" fichiers")
        if resp == 'yes' and not self.training:
            self.start_training_thread()

    def on_training_stop(self):
        """
        Arrêt de l'entrainement (fermeture du dialogue)
        """
        print('stopping')
        self.stoppingTraining=True
        self.waitingForStop = True

    def load_weights(self):
        """
        Chargement des poids depuis un fichier *.dat
        """

        # demande le fichier
        filename = tk.filedialog.askopenfilename(defaultextension='.dat',
                                                 filetypes=[('weight files','.dat'), ('all files', '.*')],
                                                 parent=self, title='Choix des poids')
        # charge les poids
        if filename:
            self.reseau.load_weights(filename)
            self.init_reseau()
            print('Chargement terminé!')
        else:
            print('Chargement non effectué')

    def start_training_thread(self):
        """
        Démarrage du thread d'entrainement du réseau
        """
        # changement des variables d'état
        self.training = True                    # alerte le reste du widget
        self.stoppingTraining = False
        self.btTrain.config(state=tk.DISABLED)  # empèche un nouveau clic
        self.btLoad.config(state=tk.DISABLED)
        self.btCreate.config(state=tk.DISABLED)

        # configuration du thread
        fileList = []
        for art in self.selectedFiles.files:
            if self.selectedFiles.enabled[art].get() == '1':
                fileList.extend([art+'/'+n for n in self.selectedFiles.files[art]])
        print('files:',fileList)
        self.trainingThread = threading.Thread(target=self._train, args=[fileList])

        self.readingThread = threading.Thread(target=self._load_data, args=[fileList])

        # configuration du dialogue de progression
        self.trainingDialog = TrainingDialog(self)
        self.trainingDialog.set_file_qty((len(fileList)+1)*self.entryEpoch.get_value())
        self.trainingDialog.protocol('WM_DELETE_WINDOW', self.on_training_stop)

        # lancement
        self.after(100, self.periodic_joiner)
        self.trainingThread.start()
        self.readingThread.start()

    def start_creation(self):
        """
        Démarre la fabrication d'un morceau par le réseau
        """
        if not self.creationFinished:
            print('Création en cours')
            return
        self.creationFinished = False
        self.creationThread = threading.Thread(target=self._create)
        self.creationThread.start()
        self.after(100, self.periodic_create_joiner)

    def periodic_joiner(self):
        """
        Callback automatique de l'interface appellé toutes les 100ms
        qui essaye de rejoindre le thread d'entrainement
        """
        if self.stoppingTraining and not self.waitingForStop:
            print('joining...')
            self.trainingThread.join()
            self.readingThread.join()
            print('joined')
            self.trainingDialog.destroy()
            self.training = False
            self.btTrain.config(state=tk.ACTIVE)
            self.btLoad.config(state=tk.ACTIVE)
            self.btCreate.config(state=tk.ACTIVE)
        else:
            self.after(500, self.periodic_joiner)

    def periodic_create_joiner(self):
        """
        Callback automatique de l'interface appellé toutes les 100ms
        qui essaye de rejoindre le thread de création
        """
        if self.creationFinished:
            self.creationThread.join()
        else:
            self.after(500, self.periodic_create_joiner)

    def _load_data(self, fileList):
        """
        Fonction dans un thread séparé de chargement des données
        """
        if not fileList:
            return
        for k in range(self.entryEpoch.get_value()):
            rdFileList = fileList[:]
            random.shuffle(rdFileList)
            for name in rdFileList:
                # attend d'avoir à charger un fichier
                while self.dataSpace == 0 and not self.stoppingTraining:
                    time.sleep(0.1)
                if self.stoppingTraining:
                    print('DATA STOP')
                    return
                # lis le fichier
                data = file.loadFile('musics/format 0/'+name)
                # charge les données dans la file
                self.dataLock.acquire()
                self.dataQueue.append( (name, data) )
                self.dataSpace -= 1
                self.dataLock.release()
        return

    def _train(self, fileList):
        """
        Fonction qui tourne dans un thread séparé et entraine le réseau
        """
        if not fileList:
            return
        # Crée le réseau
        if not self.reseau.graph.is_init:
            self.init_reseau(lambda t: self.trainingDialog.update_file('Initialisation du réseau:'+t, -1))

        # Utilise les données
        if not self.stoppingTraining:
            self.trainingDialog.update_adv('Chargement des données',-1)
        else:
            print('STOP!')
            self.waitingForStop = False
            return
        num_ex = 0
        for k in range(self.entryEpoch.get_value()):
            for i in range(len(fileList)):
                # attend d'avoir des données disponibles
                while len(self.dataQueue) == 0:
                    time.sleep(0.1)
                # récupère une page de donnée
                self.dataLock.acquire()
                name, data = self.dataQueue.pop(0)
                self.dataSpace += 1
                self.dataLock.release()
                # màj les informations du dialog
                tx = '(E:'+str(k+1)+'/'+str(self.entryEpoch.get_value())+' F:'+str(i+1)+'/'+str(len(fileList))+') '
                self.trainingDialog.update_file(tx+name, (i+1)*(k+1))
                self.trainingDialog.update_adv('Chargement', 0)
                self.status.update((i+1)*(k+1)*100/((len(fileList)+1)*self.entryEpoch.get_value()), 'Entrainement...')
                if self.stoppingTraining:
                    print('STOP!')
                    self.waitingForStop = False
                    return
                # transforme en entrées -> sorties
                x = data[:-1]
                y = data[1:]
                # entraine
                cost, _, _ = self.reseau.graph.train(x, y, self.reseau.learning_rate)
                self.trainingDialog.errorGraph.add_point('error', cost)
                if num_ex % 20 == 0:  # Sauvegarde le résultat tous les 10 exemples
                    self.reseau.save_weights('data/snapshots/snapshot_' + str(num_ex) + '.dat')
                num_ex += 1

                if self.stoppingTraining:
                    print('STOP!')
                    self.waitingForStop = False
                    return

        # sauvegarde le réseau
        print('Saving')
        self.reseau.save_weights('data/final_weights.dat')
        
        print('END!')
        self.stoppingTraining = True
        self.waitingForStop = False
        return

    def _create(self):
        """
        Fonction qui tourne dans un thread séparé et génère une musique
        """
        # check d'existence du réseau
        if not self.reseau.graph.is_init:
            self.creationFinished = True
            print('pas de réseau :-(')
            return
        print('Start generation')
        # génère la musique
        frames = self.reseau.predict([1*(i==129) for i in range(128+3)], [0], 50000)
        frames = frames.tolist()
        # fixe à 0 ou 1 les notes
        for n in range(len(frames)):
            for i in range(128):
                frames[n][i] = int(frames[n][i]+0.5)
        print('Saving '+str(len(frames))+' frames')
        # enregistre
        file.makeFile(frames, 'test.mid')
        self.creationFinished = True
        print('Musique écrite!')
        

if __name__ == '__main__':
    win = Interface()
    win.mainloop()
