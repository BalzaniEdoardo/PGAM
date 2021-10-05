from factor_analyzer.factor_analyzer import FactorAnalyzer
import pandas as pd
import matplotlib.pylab as plt
df = pd.read_csv('/Users/edoardo/Downloads/Affairs.csv')
df.drop(['Unnamed: 0','gender', 'education', 'age','children'],axis=1,inplace=True)

fa = FactorAnalyzer(rotation='varimax')
fa.fit(df)
ev, v = fa.get_eigenvalues()


plt.plot(range(1,1+df.shape[1]),v)