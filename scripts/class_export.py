"""
Module de génération d'exports au format ESG
"""

import pandas as pd
import numpy as np
from pathlib import Path


class OutputGenerator:
    """
    Génère un fichier CSV consolidé au format ESG contenant toutes les simulations.
    Compatible avec la structure de données du projet ESG.
    """
    
    def __init__(
        self,
        T: int,
        N: int,
        dividend_rate: float,
        rental_rate: float,
        zc_prices_dict: dict,
        equity_index: pd.DataFrame,
        real_estate_index: pd.DataFrame,
        deflator: pd.DataFrame,
        template_path: str
    ):
        """
        Initialise le générateur d'exports.
        
        Parameters
        ----------
        T : int
            Horizon de projection (années)
        N : int
            Nombre de simulations
        dividend_rate : float
            Taux de dividende fixe pour les actions (en décimal, ex: 0.02 pour 2%)
        rental_rate : float
            Taux de loyer fixe pour l'immobilier (en décimal, ex: 0.02 pour 2%)
        zc_prices_dict : dict
            Dictionnaire des prix ZC par maturité {maturity: DataFrame(N × T+1)}
        equity_index : pd.DataFrame
            DataFrame des indices actions (N × T+1), base 100
        real_estate_index : pd.DataFrame
            DataFrame des indices immobilier (N × T+1), base 100
        deflator : pd.DataFrame
            DataFrame du déflateur Vasicek (N × T+1)
        template_path : str
            Chemin vers le template CSV avec colonnes: CLASS;MEASURE;OS_TERM;0
        """
        self.T = T
        self.N = N
        self.dividend_rate = dividend_rate
        self.rental_rate = rental_rate
        self.zc_prices_dict = zc_prices_dict
        self.equity_index = equity_index
        self.real_estate_index = real_estate_index
        self.deflator = deflator
        self.template_path = template_path
        
        # Charger le template (attention au séparateur ';')
        self.template = pd.read_csv(template_path, sep=';')
        
    def generate_all_simulations(self) -> pd.DataFrame:
        """
        Génère un DataFrame consolidé avec toutes les simulations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame avec colonnes: Simulation,CLASS,MEASURE,OS_TERM,0,1,2,...,T
        """
        all_simulations = []
        
        for i in range(self.N):
            # Clone le template
            sim_df = self.template.copy()
            
            # Ajoute la colonne Simulation
            sim_df.insert(0, 'Simulation', i + 1)
            
            # Ajoute les colonnes temporelles de 1 à T
            for t in range(1, self.T + 1):
                sim_df[str(t)] = 0.0
            
            # Remplit les valeurs selon les règles
            self._fill_simulation_values(sim_df, i)
            
            all_simulations.append(sim_df)
        
        # Concatène tous les DataFrames
        consolidated_df = pd.concat(all_simulations, ignore_index=True)
        
        return consolidated_df
    
    def _fill_simulation_values(self, df: pd.DataFrame, sim_idx: int):
        """
        Remplit les valeurs d'une simulation selon les règles de mapping.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de la simulation à remplir (modifié in-place)
        sim_idx : int
            Index de la simulation (0-based, correspond aux lignes des DataFrames)
        """
        # L'index des DataFrames va de 1 à N, donc on utilise sim_idx + 1
        sim_number = sim_idx + 1
        
        for idx, row in df.iterrows():
            class_val = row['CLASS']
            measure = row['MEASURE']
            os_term = row['OS_TERM']
            
            # EQUITIES - RET_IDX (indices actions)
            if class_val == 'EQUITIES' and measure == 'RET_IDX' and os_term == 0:
                for t in range(self.T + 1):
                    df.at[idx, str(t)] = self.equity_index.loc[sim_number, t]
            
            # REAL_ESTATE - RET_IDX (indices immobilier)
            elif class_val == 'REAL_ESTATE' and measure == 'RET_IDX' and os_term == 0:
                for t in range(self.T + 1):
                    df.at[idx, str(t)] = self.real_estate_index.loc[sim_number, t]
            
            # ZCB - PRICE (prix zéro-coupon par maturité)
            elif class_val == 'ZCB' and measure == 'PRICE':
                maturity = int(os_term)
                if maturity in self.zc_prices_dict:
                    zc_df = self.zc_prices_dict[maturity]
                    for t in range(self.T + 1):
                        df.at[idx, str(t)] = zc_df.loc[sim_number, t]
            
            # VALN - DEF (déflateur)
            elif class_val == 'VALN' and measure == 'DEF' and os_term == 0:
                for t in range(self.T + 1):
                    df.at[idx, str(t)] = self.deflator.loc[sim_number, t]
            
            # EQUITIES - RNY_PC (taux de dividende)
            elif class_val == 'EQUITIES' and measure == 'RNY_PC' and os_term == 0:
                for t in range(self.T + 1):
                    df.at[idx, str(t)] = self.dividend_rate
            
            # REAL_ESTATE - RNY_PC (taux de loyer)
            elif class_val == 'REAL_ESTATE' and measure == 'RNY_PC' and os_term == 0:
                for t in range(self.T + 1):
                    df.at[idx, str(t)] = self.rental_rate
            
            # Toutes les autres lignes restent à 0 (déjà initialisées)
            # (INFLN, BANK_DEPOSIT_RET, TME, OAT10, OAT1)
    
    def export_to_csv(self, output_path: str) -> pd.DataFrame:
        """
        Génère et exporte le DataFrame consolidé vers un fichier CSV.
        
        Parameters
        ----------
        output_path : str
            Chemin du fichier CSV de sortie
            
        Returns
        -------
        pd.DataFrame
            Le DataFrame consolidé généré
        """
        consolidated_df = self.generate_all_simulations()
        consolidated_df.to_csv(output_path, index=False)
        
        return consolidated_df
    
    def get_simulation_preview(self, sim_number: int) -> pd.DataFrame:
        """
        Génère un aperçu d'une simulation spécifique.
        
        Parameters
        ----------
        sim_number : int
            Numéro de la simulation (1-based)
        
        Returns
        -------
        pd.DataFrame
            DataFrame de la simulation demandée
        """
        if sim_number < 1 or sim_number > self.N:
            raise ValueError(f"sim_number doit être entre 1 et {self.N}")
        
        # Clone le template
        sim_df = self.template.copy()
        sim_df.insert(0, 'Simulation', sim_number)
        
        # Ajoute les colonnes temporelles
        for t in range(1, self.T + 1):
            sim_df[str(t)] = 0.0
        
        # Remplit les valeurs (index 0-based)
        self._fill_simulation_values(sim_df, sim_number - 1)
        
        return sim_df
    
    def get_export_info(self) -> dict:
        """
        Retourne des informations sur l'export à générer.
        
        Returns
        -------
        dict
            Dictionnaire contenant les infos: N, T, template_rows, total_rows
        """
        return {
            'N': self.N,
            'T': self.T,
            'template_rows': len(self.template),
            'total_rows': self.N * len(self.template),
            'dividend_rate': self.dividend_rate,
            'rental_rate': self.rental_rate
        }