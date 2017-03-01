# coding: utf-8
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.messagebox
import os
import threading
import file
import time
import numpy as np
from LSTM import LSTM

class FileSelector(tk.Frame):
    """
    Widget de selection des fichiers à utiliser
    """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, relief=tk.RIDGE, bd=2)
        self.files = {}
        self.discoverFiles()
        self.enabled = {}
        self.checks = {}
        for opt in sorted(list(k for k in self.files)):
            self.enabled[opt] = tk.Variable()
            self.checks[opt] = tk.Checkbutton(self, text=opt+' ('+str(len(self.files[opt]))+')', variable=self.enabled[opt], bd=0, state=tk.ACTIVE)
            self.checks[opt].select()
            self.checks[opt].deselect()
            self.checks[opt].pack(anchor=tk.W)

    def discoverFiles(self):
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

class TrainingDialog(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)

        # indicateur global
        self.fileLabel = tk.Label(self, text='Fichier:')
        self.fileLabel.pack(anchor=tk.W)
        
        # barre de progression totale
        self.progFileValue = tk.IntVar()
        self.progFileMax = 0
        self.progFile = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate', length=400, variable=self.progFileValue)
        self.progFile.pack(side=tk.TOP)

        # indicateur du fichier actuel
        self.advLabel = tk.Label(self, text='Etape:')
        self.advLabel.pack(anchor=tk.W)

        # barre de progression du fichier actuel
        self.progAdvValue = tk.IntVar()
        self.progAdvMax = 0
        self.progAdv = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate', length=400, variable=self.progAdvValue)
        self.progAdv.pack(side=tk.TOP)
        
    def setFileQty(self, value):
        """
        Nombre de fichier total à parcourir
        """
        self.progFile.config(maximum=value)
        self.progFileMax = value

    def setFileMax(self, value):
        """
        Taille du fichier à parcourir (en nombre d'échantillons)
        """
        self.progAdv.config(maximum=value)
        self.progAdvMax = value
        
    def updateFile(self, info, num):
        """
        Affiche le numéro et le nom du fichier actuel
        """
        self.fileLabel.config(text=info)
        self.progFileValue.set(num)
        
    def updateAdv(self, state, num):
        """
        Affiche la progression dans le fichier actuel
        """
        self.advLabel.config(text='Etape:'+state)
        self.progAdvValue.set(num)

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

        self.btTrain = tk.Button(self.cmdFrame, text="Commencer l'entrainement", command=self.confirmTrain)
        self.btTrain.pack()

        self.btCreate = tk.Button(self.cmdFrame, text="Ecrire la musique", command=self.startCreation)
        self.btCreate.pack()

        self.btLoad = tk.Button(self.cmdFrame, text='Charger des poids', command=self.loadWeights)
        self.btLoad.pack()
        
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

        self.reseau = None

    def confirmTrain(self):
        """
        Affiche une boite de confirmation de démarrage de l'apprentissage
        """
        print([self.selectedFiles.enabled[f].get() for f in self.selectedFiles.enabled])
        num = sum(len(self.selectedFiles.files[f]) for f in self.selectedFiles.files if self.selectedFiles.enabled[f].get()=='1')
        resp = tk.messagebox.askquestion('Confirmation', "Êtes-vous sûr de vouloir lancer\nl'apprentissage avec "+str(num)+" fichiers")
        if resp == 'yes' and not self.training:
            self.startTrainingThread()

    def onTrainingStop(self):
        print('stopping')
        self.stoppingTraining=True
        self.waitingForStop = True

    def loadWeights(self):
        # Crée le réseau
        if not self.reseau:
            self.status.update(0, 'Initialisation du réseau:')

            self.reseau = LSTM(131, 0.1, True)
            self.reseau.add_lstm_layer(131)
            self.reseau.add_simple_layer(131)
            self.reseau.graph.debug_print = lambda t:print('Initialisation du réseau:'+t)
            # TODO: connect progressbar
            self.reseau.init_graph()

        # demande le fichier
        file = tk.filedialog.askopenfilename(defaultextension='.dat',
                                             filetypes=[('weight files','.dat'), ('all files', '.*')],
                                             parent=self, title='Choix des poids')
        # charge les poids
        self.reseau.load_weights(file)
        print('Chargement terminé!')

    def startTrainingThread(self):
        # changement des variables d'état
        self.training = True                    # alerte le reste du widget
        self.stoppingTraining = False
        self.btTrain.config(state=tk.DISABLED)  # empèche un nouveau clic

        # configuration du thread
        fileList = []
        for art in self.selectedFiles.files:
            if self.selectedFiles.enabled[art].get() == '1':
                fileList.extend([art+'/'+n for n in self.selectedFiles.files[art]])
        self.trainingThread = threading.Thread(target=lambda :self._train(fileList))

        self.readingThread = threading.Thread(target=lambda :self._loadData(fileList))

        # configuration du dialogue de progression
        self.trainingDialog = TrainingDialog(self)
        self.trainingDialog.setFileQty(len(fileList)+1)
        self.trainingDialog.protocol('WM_DELETE_WINDOW', self.onTrainingStop)

        # lancement
        self.after(100, self.periodicJoiner)
        self.trainingThread.start()
        self.readingThread.start()

    def startCreation(self):
        """
        Démarre la fabrication d'un morceau par le réseau
        """
        if not self.creationFinished:
            print('Création en cours')
            return
        self.creationFinished = False
        self.creationThread = threading.Thread(target=self._create)
        self.creationThread.start()
        self.after(100, self.periodicCreateJoiner)

    def periodicJoiner(self):
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
        else:
            self.after(100, self.periodicJoiner)

    def periodicCreateJoiner(self):
        if self.creationFinished:
            self.creationThread.join()
        else:
            self.after(100, self.periodicCreateJoiner)

    def _loadData(self, fileList):
        """
        Fonction dans un thread séparé de chargement des données
        """
        for name in fileList:
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
            self.dataQueue.append(data)
            self.dataSpace -= 1
            self.dataLock.release()
        return

    def _train(self, fileList):
        """
        Fonction qui tourne dans un thread séparé et entraine le réseau
        """
        # Crée le réseau
        if not self.reseau:
            self.trainingDialog.updateFile('Initialisation du réseau',-1)
            self.status.update(0, 'Initialisation du réseau')

            self.reseau = LSTM(131, 0.1, True)
            self.reseau.add_lstm_layer(131)
            self.reseau.add_simple_layer(131)
            self.reseau.graph.debug_print = lambda t: self.trainingDialog.updateAdv(t, 0)
            # TODO: connect progressbar
            self.reseau.init_graph()

        # Utilise les données
        if not self.stoppingTraining:
            self.trainingDialog.updateAdv('Chargement des données',-1)
        else:
            print('STOP!')
            self.waitingForStop = False
            return
        for i in range(len(fileList)):
            # màj les informations du dialog
            name = fileList[i]
            self.trainingDialog.updateFile(name, i+1)
            self.trainingDialog.updateAdv('Chargement', 0)
            self.status.update((i+1)*100/(len(fileList)+1), 'Entrainement...')
            # attend d'avoir des données disponibles
            while len(self.dataQueue) == 0:
                time.sleep(0.1)
            if self.stoppingTraining:
                print('STOP!')
                self.waitingForStop = False
                return
            # récupère une page de donnée
            self.dataLock.acquire()
            data = self.dataQueue.pop(0)
            self.dataSpace += 1
            self.dataLock.release()
            # transforme en entrées -> sorties
            x = data[:-1]
            y = data[1:]

            self.reseau.graph.train(x, y, self.reseau.learning_rate)

            if self.stoppingTraining:
                print('STOP!')
                self.waitingForStop = False
                return

        # sauvegarde le réseau
        print('Saving')
        self.reseau.save_weights('data/snapshot.dat')
        
        print('END!')
        self.stoppingTraining = True
        self.waitingForStop = False
        return

    def _create(self):
        """
        Fonction qui tourne dans un thread séparé et génère une musique
        """
        # check d'existence du réseau
        if not self.reseau:
            self.creationFinished = True
            print('pas de réseau :-(')
            return
        print('Start generation')
        # génère la musique
        frame = np.asarray( [1*(i==129) for i in range(128+3)] )
        frames = [frame]
        count = 0
        while frame[130] < 0.5 and count < 500:
            self.status.update(0, 'Génération:'+str(count))
            count += 1
            frame = self.reseau.predict(frames)[-1]
            frames.append(frame)
        # fixe à 0 ou 1 les notes
        for n in range(len(frames)):
            for i in range(128):
                frames[n][i] = int(frames[n][i]+0.5)
        print('Saving')
        # debug
        tstf = open('testing.txt', 'w')
        print(frames, file=tstf)
        tstf.close()
        # enregistre
        file.makeFile(frames, 'test.mid')
        self.creationFinished = True
        print('Musique écrite!')
        

if __name__ == '__main__':
    win = Interface()
    win.mainloop()
    
## tests
'''
    
m = midi.MidiFile()
# m.open('musics/format 0/albeniz/alb_esp2_format0.mid')
# m.open('musics/format 0/clementi/clementi_opus36_1_1_format0.mid')
m.open('musics/format 0/chopin/chpn_op25_e1_format0.mid')
m.read()
m.close()

print(m.tracks[0])
    
# étendue des notes
lowK = min(e.pitch for e in m.tracks[0].events if e.type == 'NOTE_ON')
highK = max(e.pitch for e in m.tracks[0].events if e.type == 'NOTE_ON')
print('range:',lowK,'-',highK)

# évenements de pédale
if e.type == 'CONTROLLER_CHANGE':
    if e.pitch == 64:
        events.append(e)
tx = ''
for e in events:
    tx += str(e)+'\n'
print(tx[:-1])

'''
