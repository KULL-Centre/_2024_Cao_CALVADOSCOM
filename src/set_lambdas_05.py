import pandas as pd

r = pd.read_csv('residues.csv').set_index('one')
r.lambdas = 0.5
r = r[['three','MW','lambdas','sigmas','q']]
r.to_csv('residues.csv')
