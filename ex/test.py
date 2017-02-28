from pickle import load

layers = load(open('data/snapshot2.dat', 'rb'))

print(layers)