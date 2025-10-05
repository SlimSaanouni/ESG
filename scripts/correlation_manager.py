"""
Module de gestion de la structure de dépendance et des corrélations entre modèles
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize


class CorrelationManager:
    """
    Classe pour gérer les matrices de corrélation entre les différents modèles
    """
    
    def __init__(self, model_types, nb_weiner_dict, calibrated_parameters=None):
        """
        Initialise le gestionnaire de corrélations
        
        :param model_types: Dictionnaire {asset_class: model_name}
        :param nb_weiner_dict: Dictionnaire {model_name: nb_browniens}
        :param calibrated_parameters: Dictionnaire {asset_class: params_dict} avec les paramètres calibrés
        """
        self.model_types = model_types
        self.nb_weiner_dict = nb_weiner_dict
        self.calibrated_parameters = calibrated_parameters if calibrated_parameters is not None else {}
        self.total_brownians = 0
        self.brownian_mapping = {}
        
        self._calculate_total_brownians()
    
    def _calculate_total_brownians(self):
        """
        Calcule le nombre total de browniens nécessaires
        """
        idx = 0
        for asset_class, model_name in self.model_types.items():
            nb_brownians = self.nb_weiner_dict.get(model_name, 1)
            self.brownian_mapping[asset_class] = {
                'model': model_name,
                'start_idx': idx,
                'end_idx': idx + nb_brownians,
                'nb_brownians': nb_brownians
            }
            idx += nb_brownians
        
        self.total_brownians = idx
    
    def get_total_brownians(self):
        """Retourne le nombre total de browniens"""
        return self.total_brownians
    
    def get_brownian_mapping(self):
        """Retourne le mapping des browniens par classe d'actifs"""
        return self.brownian_mapping
    
    def build_theoretical_correlation_matrix(self, empirical_corr):
        """
        Construit la matrice de corrélation théorique complète à partir
        de la matrice de corrélation empirique entre classes d'actifs
        
        :param empirical_corr: DataFrame de corrélation empirique (3x3)
        :return: Matrice de corrélation théorique complète
        """
        n = self.total_brownians
        theoretical_corr = np.eye(n)
        
        # Liste des classes d'actifs dans l'ordre
        asset_classes = list(self.model_types.keys())
        
        # Remplissage de la matrice selon les corrélations empiriques
        for i, asset_i in enumerate(asset_classes):
            for j, asset_j in enumerate(asset_classes):
                if i == j:
                    # Corrélation intra-actif
                    self._fill_intra_asset_correlation(
                        theoretical_corr, 
                        asset_i
                    )
                else:
                    # Corrélation inter-actifs
                    self._fill_inter_asset_correlation(
                        theoretical_corr,
                        asset_i,
                        asset_j,
                        empirical_corr.loc[asset_i, asset_j]
                    )
        
        return pd.DataFrame(
            theoretical_corr,
            columns=[f"W{i+1}" for i in range(n)],
            index=[f"W{i+1}" for i in range(n)]
        )
    
    def _fill_intra_asset_correlation(self, corr_matrix, asset_class):
        """
        Remplit la corrélation pour les browniens d'une même classe d'actifs
        
        :param corr_matrix: Matrice à remplir
        :param asset_class: Classe d'actifs concernée
        """
        mapping = self.brownian_mapping[asset_class]
        start = mapping['start_idx']
        end = mapping['end_idx']
        nb_brownians = mapping['nb_brownians']
        
        if nb_brownians == 2:
            # Pour les modèles à 2 browniens (Heston, G2++)
            model_name = mapping['model']
            
            if model_name == 'Heston':
                # Récupérer le rho calibré si disponible
                if asset_class in self.calibrated_parameters:
                    params = self.calibrated_parameters[asset_class]
                    rho = params.get('rho', -0.5)  # Valeur par défaut si non trouvé
                else:
                    rho = -0.5  # Valeur par défaut si pas de calibration
                
                corr_matrix[start, start + 1] = rho
                corr_matrix[start + 1, start] = rho
                
            elif model_name == 'G2++':
                # Pour G2++, récupérer la corrélation calibrée si disponible
                if asset_class in self.calibrated_parameters:
                    params = self.calibrated_parameters[asset_class]
                    # G2++ pourrait avoir un paramètre de corrélation spécifique
                    rho_g2 = params.get('rho', 0.3)
                else:
                    rho_g2 = 0.3  # Valeur par défaut
                
                corr_matrix[start, start + 1] = rho_g2
                corr_matrix[start + 1, start] = rho_g2
    
    def _fill_inter_asset_correlation(self, corr_matrix, asset_i, asset_j, empirical_corr):
        """
        Remplit la corrélation entre browniens de différentes classes d'actifs
        
        :param corr_matrix: Matrice à remplir
        :param asset_i: Première classe d'actifs
        :param asset_j: Deuxième classe d'actifs
        :param empirical_corr: Corrélation empirique entre les deux classes
        """
        mapping_i = self.brownian_mapping[asset_i]
        mapping_j = self.brownian_mapping[asset_j]
        
        # Indices des browniens
        start_i = mapping_i['start_idx']
        end_i = mapping_i['end_idx']
        start_j = mapping_j['start_idx']
        end_j = mapping_j['end_idx']
        
        # Remplissage : on met la corrélation empirique entre le premier brownien
        # de chaque classe (simplification)
        corr_matrix[start_i, start_j] = empirical_corr
        corr_matrix[start_j, start_i] = empirical_corr
        
        # Pour les browniens supplémentaires, on met des corrélations plus faibles
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                if i == start_i and j == start_j:
                    continue  # Déjà rempli
                # Corrélation diminuée pour les browniens secondaires
                corr_matrix[i, j] = empirical_corr * 0.5
                corr_matrix[j, i] = empirical_corr * 0.5
    
    def is_psd(self, corr_matrix):
        """
        Vérifie si une matrice est positive semi-définie
        
        :param corr_matrix: Matrice à vérifier
        :return: (bool, eigenvalues) - True si PSD, et les valeurs propres
        """
        if isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = corr_matrix.values
        
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        is_psd = np.all(eigenvalues >= +1e-10)  # Tolérance numérique
        
        return is_psd, eigenvalues
    
    def make_psd_higham(self, corr_matrix):
        """
        Projette une matrice de corrélation sur l'ensemble des matrices PSD
        en utilisant l'algorithme de Higham (1988)
        
        :param corr_matrix: Matrice de corrélation à corriger
        :return: Matrice de corrélation PSD
        """
        if isinstance(corr_matrix, pd.DataFrame):
            index = corr_matrix.index
            columns = corr_matrix.columns
            corr_matrix = corr_matrix.values
        else:
            index = None
            columns = None
        
        # Algorithme de Higham
        max_iterations = 100
        tolerance = 1e-9
        epsilon = 1e-8
        
        X = corr_matrix.copy()
        Y = corr_matrix.copy()
        
        for iteration in range(max_iterations):
            # Projection sur les matrices PSD
            eigenvalues, eigenvectors = eigh(Y)
            
            # Mettre les valeurs propres négatives à zéro
            eigenvalues[eigenvalues < 0] = epsilon
            
            # Reconstruction
            X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Projection sur les matrices de corrélation (diagonale = 1)
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            
            # Vérification de convergence
            if np.linalg.norm(Y - X) < tolerance:
                break
        
        # S'assurer que la matrice est symétrique
        Y = (Y + Y.T) / 2
        
        # S'assurer que la diagonale est exactement 1
        np.fill_diagonal(Y, 1.0)
        
        # Retourner sous forme de DataFrame si nécessaire
        if index is not None and columns is not None:
            return pd.DataFrame(Y, index=index, columns=columns)
        
        return Y
    
    def calculate_correction_stats(self, original_matrix, corrected_matrix):
        """
        Calcule les statistiques sur la correction effectuée
        
        :param original_matrix: Matrice originale
        :param corrected_matrix: Matrice corrigée
        :return: DataFrame avec les statistiques de correction
        """
        if isinstance(original_matrix, pd.DataFrame):
            original = original_matrix.values
        else:
            original = original_matrix
        
        if isinstance(corrected_matrix, pd.DataFrame):
            corrected = corrected_matrix.values
        else:
            corrected = corrected_matrix
        
        # Calcul des écarts
        diff = np.abs(corrected - original)
        
        # Extraction des éléments hors diagonale
        n = original.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag_diffs = diff[mask]
        
        stats = {
            'Max Absolute Difference': np.max(diff),
            'Mean Absolute Difference': np.mean(off_diag_diffs),
            'Median Absolute Difference': np.median(off_diag_diffs),
            'Std Absolute Difference': np.std(off_diag_diffs),
            'Number of Modified Elements': np.sum(diff > 1e-6)
        }
        
        return pd.DataFrame([stats]).T.rename(columns={0: 'Value'})
    
    def get_difference_matrix(self, original_matrix, corrected_matrix):
        """
        Calcule la matrice des différences élément par élément
        
        :param original_matrix: Matrice originale
        :param corrected_matrix: Matrice corrigée
        :return: DataFrame des différences
        """
        if isinstance(original_matrix, pd.DataFrame):
            diff = corrected_matrix.values - original_matrix.values
            return pd.DataFrame(
                diff,
                index=original_matrix.index,
                columns=original_matrix.columns
            )
        else:
            return corrected_matrix - original_matrix