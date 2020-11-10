import pandas as pd
import numpy as np
from datetime import datetime as dt

class Dataset():
    def __init__(self, initial_n=1000, n_years=3):
        cdi_n = int(round(initial_n * .8))
        cdd_n = initial_n - cdi_n
        for i in range(n_years):
            self.year = dt.now().year - (n_years - (i + 1))
            if i == 0:
                self.df = self.first_year(cdi_n, cdd_n, True)
                self.df.columns = ['matricule'] + [f'{col} {self.year}' for col in self.df.columns[1:]]
            else:
                ny_df = self.next_year()
                ny_df.columns = ['matricule'] + [f'{col} {self.year}' for col in ny_df.columns[1:]]
                self.df = self.df.merge(ny_df, on='matricule', how='outer')
                self.df.loc[
                    (self.df[f'contrat {self.year-1}']=='CDD') &
                    (self.df[f'contrat {self.year}']=='CDI')
                    ,
                    f'turnover {self.year}'
                ] = 'CDD > CDI'
                self.df.loc[(self.df[f'contrat {self.year-1}'].notna()) & (self.df[f'contrat {self.year}'].isna()),
                            f'turnover {self.year}'] = 'Sorties'
                self.df.loc[(self.df[f'contrat {self.year-1}'].isna()) & (self.df[f'contrat {self.year}'].notna()),
                            f'turnover {self.year}'] = 'Entrées'

    def get_seniority(self, age):
        start = min(np.random.randint(18, 25), age)
        max_seniority = age - start
        return round(max_seniority * np.random.random())

    def first_year(self, cdi_n, cdd_n, initial_year=False):
        # instanciate
        n = cdi_n + cdd_n
        df = pd.DataFrame(index=np.arange(n))

        # matricules
        if initial_year:
            df['matricule'] = np.arange(n)
        else:
            df['matricule'] = np.arange(self.df.matricule.max() + 1, self.df.matricule.max() + 1 + n)

        # contract
        p = cdi_n / n
        df['contrat'] = np.random.choice(['CDI', 'CDD'], p=[p, 1-p], size=n)
        cdi = df.contrat == 'CDI'
        cdd = df.contrat == 'CDD'

        # gender
        df.loc[cdd, 'genre'] = np.random.choice(['F', 'H'], p=[.6, .4], size=cdd.sum())
        df.loc[cdi, 'genre'] = np.random.choice(['F', 'H'], p=[.45, .55], size=cdi.sum())

        # age
        df.loc[cdi, 'age'] = pd.DataFrame(np.random.normal(42, 6, size=n), columns=['age']).round()
        df.loc[cdd, 'age'] = pd.DataFrame(np.random.normal(26, 2, size=n), columns=['age']).round()

        # seniority
        if initial_year:
            df.loc[cdi, 'ancienneté'] = df.loc[cdi, 'age'].apply(self.get_seniority)
            df.loc[cdd, 'ancienneté'] = np.random.choice([0, 1, 2], p=[.3, .6, .1], size=df[cdd].shape[0])
        else:
            df['ancienneté'] = 0

        # rank
        ranks = ['1. Employés', '2. Techniciens', '3. Cadres', '4. Directeurs']
        df.loc[cdd, 'catégorie'] = np.random.choice(ranks,
                                                    p=[.5, .4, .1, .0], size=cdd.sum())
        df.loc[cdi, 'catégorie'] = np.random.choice(ranks,
                                                    p=[.2, .4, .35, .05], size=cdi.sum())

        # wage
        df['salaire'] = 18 + \
                     .05 * df.ancienneté + \
                     np.random.random() * 1 * (df.genre == 'H') + \
                     .1 * (df.genre == 'H') * df.catégorie.astype('category').cat.codes + \
                     3 * (df.genre == 'H') * df.catégorie.astype('category').cat.codes ** 2 + \
                     .08 * (df.genre == 'F') * df.catégorie.astype('category').cat.codes + \
                     2.4 * (df.genre == 'F') * df.catégorie.astype('category').cat.codes ** 2 + \
                     .1 * df.catégorie.astype('category').cat.codes +  3 * df.catégorie.astype('category').cat.codes ** 2 + \
                     np.random.random() * 2
        df.salaire = df.salaire.round(2)

        return df

    def next_year(self):
        # start from current df
        df = self.df.copy()

        # keep only latest year data
        df = df.loc[df[f'contrat {self.year-1}'].notna(),
                    ['matricule'] + [col for col in df.columns if col.endswith(str(self.year-1))
                                                         and not (
                                                             col.startswith('turnover') |
                                                             col.startswith('promotion')
                                                         )]]

        # remove year from column labels
        df.columns = ['matricule'] + [col[:-5] for col in df.columns[1:]]

        # define filters
        cdi = df.contrat == 'CDI'
        cdd = df.contrat == 'CDD'

        # departures
        df['dep'] = False
            # retirements
        df.loc[df[f'age'] > 67, 'dep'] = True
            # temporary contract endings
        df.loc[cdd & (df.ancienneté >= 2), 'dep'] = True

            # turnover
        cdd_dep = df.groupby('contrat').dep.value_counts().unstack()[True].fillna(0).loc['CDD']
        if cdd_dep < df[cdd].shape[0] * .2:
            p = .2 - cdd_dep / df[cdd].shape[0]
            idx = df[cdd & ~df.dep].index
            idx = np.random.choice(idx, replace=False, size=int(round(idx.size * p + np.random.normal(0, .01))))
            df.loc[df.index.isin(idx), 'dep'] = True
        p = np.random.normal(.1, .02)
        idx = df[cdi & ~df.dep].index
        idx = np.random.choice(idx, replace=False, size=int(round(idx.size * p)))
        df.loc[df.index.isin(idx), 'dep'] = True
        cdi_dep = df.groupby('contrat').dep.value_counts().unstack()[True].fillna(0).loc['CDI']
        cdd_dep = df.groupby('contrat').dep.value_counts().unstack()[True].fillna(0).loc['CDD']
        df = df.loc[~df.dep, :].drop('dep', axis=1)
        cdi = df.contrat == 'CDI'
        cdd = df.contrat == 'CDD'

        # contract conversions (~5%)
        p = .2 + np.random.normal(0, .01)
        df['conversion'] = False
        df.loc[cdd, 'conversion'] = np.random.choice([True, False], p=[p, 1-p], size=cdd.sum())
        conv = df['conversion'].sum()
        df.loc[df['conversion'], 'contrat'] = 'CDI'
        df = df.drop('conversion', axis=1)

        # increment age and seniority
        df.age += 1
        df.ancienneté += 1

        # promotions
        df['promotion'] = False
        ranks = ['1. Employés', '2. Techniciens', '3. Cadres', '4. Directeurs']
        for i in range(len(ranks[:-1])):
            p = np.random.normal(1, .1) * [.2, .05, .01][i]
            mask = (df.catégorie==ranks[i]) & cdi
            df.loc[mask, 'promotion'] = np.random.choice([False, True], p=[1-p, p], size=mask.sum())
        promoted = df.promotion
        df.loc[promoted, 'catégorie'] = df.loc[promoted, 'catégorie'].apply(
            lambda rank: ranks[ranks.index(rank) + 1]
        )

        # raise
        df['augmentation'] = False
        df.loc[promoted, 'augmentation'] = True
        for i in range(len(ranks)):
            p = [np.random.normal(.15, .01), np.random.normal(.15, .01),
                 np.random.normal(.25, .02), np.random.normal(.45, .05)][i]
            mask = (df.catégorie==ranks[i]) & ~promoted
            df.loc[mask, 'augmentation'] = np.random.choice([False, True], p=[1-p, p], size=mask.sum())
        aug = df.augmentation
        top = df.catégorie == ranks[-1]
        df.loc[promoted, 'augmentation (%)'] = np.random.normal(.25, .02, promoted.sum()).round(4)
        mask = aug & top
        df.loc[mask, 'augmentation (%)'] = np.random.normal(.15, .02, mask.sum()).round(4)
        mask = aug & ~promoted & ~top
        df.loc[mask, 'augmentation (%)'] = np.random.normal(.03, .01, mask.sum()).round(4)
        df['augmentation (K€)'] = (df.salaire * df['augmentation (%)']).round(2)
        df.salaire += df['augmentation (K€)'].fillna(0)

        # arrivals
        cdi_n = max(0, int(round((cdi_dep - conv) * (1 + np.random.normal(0, .1)))))
        cdd_n = max(0, int(round((cdd_dep + conv) * (1 + np.random.normal(0, .1)))))

        return pd.concat([df, self.first_year(cdi_n, cdd_n)])
