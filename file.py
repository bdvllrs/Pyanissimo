import midi
import numpy as np


def loadFile(name, step=10, timeLimit=-1):
    """
    Charge un fichier, et le converti en tableau de notes échantillonées
    :param step: pas de quantification de la musique (ms)
    :param timeLimit: temps total limite de la musique (ms)
                      ignoré si négatif
    """
    m = midi.MidiFile()
    m.open(name)
    m.read()
    m.close()

    # print(m.tracks[0])
    # conversion en notes jouées toutes les <step> ms
    noteTimeline = {}
    currentNotes = {p: [-1, 0] for p in range(128)}
    currentTempo = 500000
    lastTempoTimeMicro = 0
    lastTempoTimeTick = 0
    for e in m.tracks[0].events:
        if e.type == 'SET_TEMPO':  # changement de tempo dans la piste
            t = (e.time-lastTempoTimeTick)*currentTempo/m.ticksPerQuarterNote + lastTempoTimeMicro # durée en µs
            currentTempo = e.data
            lastTempoTimeMicro = t
            lastTempoTimeTick = e.time
        elif e.type == 'NOTE_ON':  # début/fin d'une note
            t = (e.time-lastTempoTimeTick)*currentTempo/m.ticksPerQuarterNote + lastTempoTimeMicro # durée en µs
            n = int( t/(step*1000) )
            if e.velocity != 0:
                # début d'une note -> ajout aux notes actuelles
                currentNotes[e.pitch] = [n, e.velocity]
            else:
                # fin d'une note: rempli le tableau de 1 depuis le début de cette note
                if currentNotes[e.pitch][0] == -1:
                    continue
                for k in range(currentNotes[e.pitch][0], n):
                    if k not in noteTimeline:
                        noteTimeline[k] = []  # ajout de la clé de temps <k>
                    vel = currentNotes[e.pitch][1]
                    noteTimeline[k].append( (e.pitch, vel) )
                currentNotes[e.pitch] = [-1, 0]
                
    # conversion en tableau des notes par étapes
    timeline = [[0 for k in range(128 + 3)] for n in range(max(noteTimeline.keys())+2)]
    print('File', name, 'of length:', len(timeline), 'loaded.')
    for n in noteTimeline:
        vel = 0
        for p, v in noteTimeline[n]:
            vel += v
            timeline[n+1][p] = 1
        timeline[n+1][128] = (vel/len(noteTimeline[n]))/128 # moyennage de la vitesse

    # trim les étapes vides au début
    while all(tm == 0 for tm in timeline[0]):
        timeline.pop(0)
    # trim les étapes selon la limite de temps
    if timeLimit < 0:
        # trim les étapes vides à la fin
        while all(tm == 0 for tm in timeline[-1]):
            timeline.pop()
    else:
        # trim les étapes jusqu'à la limite de temps
        timeline = timeline[:int(timeLimit/step)]

    # signaux de début et fin
    timeline = [[1*(i==129) for i in range(128+3)]] + timeline + [[1*(i==130) for i in range(128+3)]]

    # Test : added to try just with one note
    # for t in range(len(timeline)):
    #     found = False
    #     for k in range(len(timeline[t])-4, -1, -1):  # We keep only the highest note
    #         if timeline[t][k] == 1 and found:
    #             timeline[t][k] = 0
    #         elif timeline[t][k] == 1 and not found:
    #             found = True
    # print(timeline)
    # End Test
    return np.asarray(timeline)

def load_file_mod(name, step=10, timeLimit=-1):
    """
    Charge un fichier, et le converti en tableau de notes échantillonées
    :param step: pas de quantification de la musique (ms)
    :param timeLimit: temps total limite de la musique (ms)
                      ignoré si négatif
    """
    m = midi.MidiFile()
    m.open(name)
    m.read()
    m.close()

    # print(m.tracks[0])
    # conversion en notes jouées toutes les <step> ms
    noteTimeline = {}
    currentNotes = {p: [-1, 0] for p in range(36)}
    currentTempo = 500000
    lastTempoTimeMicro = 0
    lastTempoTimeTick = 0
    for e in m.tracks[0].events:
        pitch = e.pitch%36
        if e.type == 'SET_TEMPO':  # changement de tempo dans la piste
            t = (e.time-lastTempoTimeTick)*currentTempo/m.ticksPerQuarterNote + lastTempoTimeMicro # durée en µs
            currentTempo = e.data
            lastTempoTimeMicro = t
            lastTempoTimeTick = e.time
        elif e.type == 'NOTE_ON':  # début/fin d'une note
            t = (e.time-lastTempoTimeTick)*currentTempo/m.ticksPerQuarterNote + lastTempoTimeMicro # durée en µs
            n = int( t/(step*1000) )
            if e.velocity != 0:
                # début d'une note -> ajout aux notes actuelles
                currentNotes[pitch] = [n, e.velocity]
            else:
                # fin d'une note: rempli le tableau de 1 depuis le début de cette note
                if currentNotes[pitch][0] == -1:
                    continue
                for k in range(currentNotes[pitch][0], n):
                    if k not in noteTimeline:
                        noteTimeline[k] = []  # ajout de la clé de temps <k>
                    vel = currentNotes[pitch][1]
                    noteTimeline[k].append( (pitch, vel) )
                currentNotes[pitch] = [-1, 0]
                
    # conversion en tableau des notes par étapes
    timeline = [[0 for k in range(36 + 3)] for n in range(max(noteTimeline.keys())+2)]
    print('File', name, 'of length:', len(timeline), 'loaded.')
    for n in noteTimeline:
        vel = 0
        for p, v in noteTimeline[n]:
            vel += v
            timeline[n+1][p] = 1
        timeline[n+1][36] = (vel/len(noteTimeline[n]))/36 # moyennage de la vitesse

    # trim les étapes vides au début
    while all(tm == 0 for tm in timeline[0]):
        timeline.pop(0)
    # trim les étapes selon la limite de temps
    if timeLimit < 0:
        # trim les étapes vides à la fin
        while all(tm == 0 for tm in timeline[-1]):
            timeline.pop()
    else:
        # trim les étapes jusqu'à la limite de temps
        timeline = timeline[:int(timeLimit/step)]

    # signaux de début et fin
    timeline = [[1*(i==37) for i in range(39)]] + timeline + [[1*(i==38) for i in range(39)]]
    return np.asarray(timeline)

def to3bytes(num):
    """
    Converti un nombre en 3 octets
    """
    L = []
    for i in range(3):
        L.append(num % 256)
        num = num >> 8
    return bytes(L[::-1])


def makeFile(data, filename, step=10):
    """
    écrit un fichier à partir d'une liste échantillonée de notes
    """
    # ajout d'une frame vide pour l'arrêt des notes
    data += [0 for i in range(131)]
    # création du fichier
    m = midi.MidiFile()

    m.ticksPerQuarterNote = 480
    m.format = 0

    tr = midi.MidiTrack(0)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # sequence_track_name (nom général)
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'SEQUENCE_TRACK_NAME'
    e.data = b'Generated file at '+filename.encode('latin')
    tr.events.append(e)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # sequence_track_name (nom de la piste)
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'SEQUENCE_TRACK_NAME'
    e.data = b'Generated track'
    tr.events.append(e)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # text_event (détails)
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'TEXT_EVENT'
    e.data = b'Generated by a LSTM'
    tr.events.append(e)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # smtpe_offset (obligatoire?)
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'SMTPE_OFFSET'
    e.data = b'`\x00\x03\x00\x00'
    tr.events.append(e)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # key_signature (obligatoire?)
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'KEY_SIGNATURE'
    e.data = b'\xff\x00'
    tr.events.append(e)
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # set_tempo
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'SET_TEMPO'
    e.data = to3bytes(int(3*step*480*10))  # on choisi 30 ticks/frame, et 480 ticks/noire (limite au 1/8eme de croche)
    tr.events.append(e)
    # ajout des notes
    currentlyPlaying = [0 for i in range(128)]
    lastChange = -1
    print(data[-1])
    for n in range(len(data)):
        # print(len(data[n]))
        if type(data[n]) == type([]):
            frame = data[n][:128]
            changes = [i for i in range(128) if frame[i] != currentlyPlaying[i]]
            if len(changes) > 0:
                # delta_time
                e = midi.DeltaTime(tr)
                e.time = 30*(n-lastChange)
                tr.events.append(e)
                # note 0
                e = midi.MidiEvent(tr)
                e.channel = 1
                e.type = 'NOTE_ON'
                e.pitch = changes[0]
                if frame[changes[0]] == 1:
                    e.velocity = max(min(int(data[n][128]*128),127),0)
                else:
                    e.velocity = 0
                tr.events.append(e)
                for i in range(len(changes)-1):
                    # delta_time
                    e = midi.DeltaTime(tr)
                    e.time = 0
                    tr.events.append(e)
                    # note 0
                    e = midi.MidiEvent(tr)
                    e.channel = 1
                    e.type = 'NOTE_ON'
                    e.pitch = changes[i+1]
                    if frame[changes[i+1]] == 1:
                        e.velocity = max(min(int(data[n][128]*128),127),0)
                    else:
                        e.velocity = 0
                    tr.events.append(e)

                lastChange = n
            currentlyPlaying = frame
    # delta_time 0
    e = midi.DeltaTime(tr)
    e.time = 0
    tr.events.append(e)
    # fin de piste
    e = midi.MidiEvent(tr)
    e.channel = None
    e.type = 'END_OF_TRACK'
    e.data = b''
    tr.events.append(e)
    m.tracks.append(tr)
    # débug
    f = open('midi_write.txt', 'w')
    print(m.tracks[0], file=f)
    f.close()
    # écriture
    m.open(filename, 'wb')
    m.write()
    m.close()

if __name__ == '__main__':
    print('load file')
    # data = loadFile('musics/format 0/albeniz/alb_esp1_format0.mid')
    data = []

    for k in range(0, 120, 1):
        data += [[1*(i==k) for i in range(131)]]*10
    for i in range(len(data)):
        data[i][128] = 0.8
    print(data)
    print('write file')
    makeFile(data, 'test.mid')
    print('written!')
    dt = loadFile('test.mid')

