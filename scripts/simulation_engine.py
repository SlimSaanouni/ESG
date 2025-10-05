"""
Module de génération de trajectoires corrélées pour les modèles financiers
"""

import numpy as np
import pandas as pd


class CorrelatedSimulationEngine:
    """
    Moteur de simulation pour générer des trajectoires corrélées
    entre différentes classes d'actifs
    """
    
    def __init__(self, models_dict, correlation_matrix, corr_manager, T, N, rfr, seed=None):
        """
        Initialise le moteur de simulation
        
        :param models_dict: Dict {asset_class: {'model_name': str, 'params': dict}}
        :param correlation_matrix: Matrice de corrélation finale (PSD) - DataFrame
        :param corr_manager: Instance de CorrelationManager pour le mapping
        :param T: Horizon de projection en années
        :param N: Nombre de simulations
        :param rfr: Fonction des taux sans risque
        :param seed: Graine aléatoire (optionnel, auto-généré si None)
        """
        self.models_dict = models_dict
        self.correlation_matrix = correlation_matrix
        self.corr_manager = corr_manager
        self.T = T
        self.N = N
        self.rfr = rfr
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        
        # Résultats
        self.trajectories = {}
        self.brownian_motions = None
        self.metadata = {}
    
    def generate_correlated_brownians(self, dt=1/252):
        """
        Génère les mouvements browniens corrélés selon la matrice de corrélation
        
        :param dt: Pas de temps (par défaut journalier)
        :return: Dict {brownian_index: array de shape (N, nb_steps)}
        """
        # Définir la graine
        np.random.seed(self.seed)
        
        # Nombre de pas de temps
        nb_steps = int(self.T / dt)
        
        # Nombre total de browniens
        total_brownians = self.corr_manager.get_total_brownians()
        
        # Décomposition de Cholesky de la matrice de corrélation
        corr_matrix_np = self.correlation_matrix.values
        try:
            L = np.linalg.cholesky(corr_matrix_np)
        except np.linalg.LinAlgError:
            raise ValueError("La matrice de corrélation n'est pas définie positive. "
                           "Assurez-vous qu'elle a été corrigée avec l'algorithme de Higham.")
        
        # Génération des browniens indépendants
        # Shape: (total_brownians, N, nb_steps)
        Z = np.random.standard_normal((total_brownians, self.N, nb_steps))
        
        # Application de la corrélation via Cholesky
        # Pour chaque simulation et chaque pas de temps
        W = np.zeros_like(Z)
        for sim in range(self.N):
            for step in range(nb_steps):
                # Transformer le vecteur de browniens indépendants
                W[:, sim, step] = L @ Z[:, sim, step]
        
        # Normalisation des moments (optionnel mais recommandé)
        for i in range(total_brownians):
            for step in range(nb_steps):
                W_temp = W[i, :, step]
                W[i, :, step] = (W_temp - W_temp.mean()) / W_temp.std()
        
        # Stockage dans un dictionnaire par index de brownien
        brownian_dict = {}
        for i in range(total_brownians):
            brownian_dict[i] = W[i, :, :]  # Shape (N, nb_steps)
        
        self.brownian_motions = brownian_dict
        return brownian_dict
    
    def run_simulations(self, dt=1/252, S0=100):
        """
        Lance toutes les simulations avec les browniens corrélés
        
        :param dt: Pas de temps
        :param S0: Valeur initiale pour les modèles d'indices
        :return: Dict {asset_class: DataFrame des trajectoires}
        """
        # Générer les browniens corrélés si pas déjà fait
        if self.brownian_motions is None:
            self.generate_correlated_brownians(dt)
        
        # Mapping des browniens
        brownian_mapping = self.corr_manager.get_brownian_mapping()
        
        # Pour chaque classe d'actifs
        for asset_class, model_info in self.models_dict.items():
            model_name = model_info['model_name']
            model_params = model_info['params']
            
            # Récupérer les indices des browniens pour cette classe
            mapping = brownian_mapping[asset_class]
            start_idx = mapping['start_idx']
            end_idx = mapping['end_idx']
            
            # Extraire les browniens pour ce modèle
            model_brownians = []
            for i in range(start_idx, end_idx):
                model_brownians.append(self.brownian_motions[i])
                        
            # Lancer la projection selon le type de modèle
            if model_name in ['Black-Scholes', 'Dupire', 'Heston']:
                # Modèles d'indices
                trajectories = self._run_index_model(
                    model_name, model_params, model_brownians, dt, S0
                )
            elif model_name in ['Vasicek', 'G2++']:
                # Modèles de taux
                trajectories = self._run_rate_model(
                    model_name, model_params, model_brownians, dt
                )
            else:
                raise NotImplementedError(f"Le modèle {model_name} n'est pas supporté")
            
            # Stocker les trajectoires
            self.trajectories[asset_class] = trajectories
        
        # Stocker les métadonnées
        self.metadata = {
            'seed': self.seed,
            'T': self.T,
            'N': self.N,
            'dt': dt,
            'S0': S0,
            'timestamp': pd.Timestamp.now()
        }
        
        return self.trajectories
    
    def _run_index_model(self, model_name, params, brownians, dt, S0):
        """
        Lance la projection pour un modèle d'indice
        
        :param model: Instance du modèle
        :param model_name: Nom du modèle
        :param params: Paramètres calibrés
        :param brownians: Liste des browniens pour ce modèle
        :param dt: Pas de temps
        :param S0: Valeur initiale
        :return: DataFrame des trajectoires
        """
        nb_steps = int(self.T / dt)
        
        # Initialisation
        paths = np.zeros((self.N, nb_steps + 1))
        paths[:, 0] = S0
        
        # Préparation des taux sans risque
        adj_rfr_list = [0]
        adj_rfr_list.extend([
            self.rfr(i + 1) * (i + 1) - self.rfr(i) * i 
            for i in range(0, self.T)
        ])
        
        if model_name == 'Black-Scholes':
            sigma = params['sigma']
            W = brownians[0]  # Un seul brownien
            
            for t in range(1, nb_steps + 1):
                rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
                paths[:, t] = paths[:, t-1] * np.exp(
                    (rfr_temp - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[:, t-1]
                )
        
        elif model_name == 'Dupire':
            base_sigma = params['sigma']
            vol_of_vol = params.get('vol_of_vol', 0.1)
            W = brownians[0]
            
            for t in range(1, nb_steps + 1):
                current_prices = paths[:, t-1]
                
                # Volatilité locale
                local_vols = base_sigma * (1 + vol_of_vol * (current_prices / S0 - 1))
                
                rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
                paths[:, t] = paths[:, t-1] * np.exp(
                    (rfr_temp - 0.5 * local_vols**2) * dt + local_vols * np.sqrt(dt) * W[:, t-1]
                )
        
        elif model_name == 'Heston':
            v0 = params['v0']
            kappa = params['kappa']
            theta = params['theta']
            sigma_v = params['sigma']
            
            W1 = brownians[0]  # Brownien du prix
            W2 = brownians[1]  # Brownien de la variance
            
            # Trajectoires de variance
            variance_paths = np.zeros((self.N, nb_steps + 1))
            variance_paths[:, 0] = v0
            
            for t in range(1, nb_steps + 1):
                v_prev = np.maximum(variance_paths[:, t-1], 0)
                S_prev = paths[:, t-1]
                
                # Mise à jour de la variance (CIR)
                variance_paths[:, t] = np.maximum(
                    v_prev + kappa * (theta - v_prev) * dt + 
                    sigma_v * np.sqrt(v_prev * dt) * W2[:, t-1],
                    0.0001
                )
                
                # Mise à jour du prix
                rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
                paths[:, t] = S_prev * np.exp(
                    (rfr_temp - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1[:, t-1]
                )
        
        # Créer le DataFrame final (seulement les années entières)
        df_paths = pd.DataFrame(
            paths[:, ::int(1/dt)],  # Sélectionner seulement les points annuels
            columns=range(0, self.T + 1),
            index=range(1, self.N + 1)
        )
        
        return df_paths
    
    def _run_rate_model(self, model_name, params, brownians, dt):
        """
        Lance la projection pour un modèle de taux
        
        :param model: Instance du modèle
        :param model_name: Nom du modèle
        :param params: Paramètres calibrés
        :param brownians: Liste des browniens pour ce modèle
        :param dt: Pas de temps
        :return: Dict des trajectoires (format spécifique aux taux)
        """
        nb_steps = int(self.T / dt)
        
        if model_name == 'Vasicek':
            kappa = params['kappa']
            theta = params['theta']
            sigma = params['sigma']
            r0 = params['r0']
            
            W = brownians[0]
            
            # Trajectoires des taux courts
            paths = np.zeros((self.N, nb_steps + 1))
            paths[:, 0] = r0
            
            for t in range(1, nb_steps + 1):
                paths[:, t] = (paths[:, t-1] + 
                              kappa * (theta - paths[:, t-1]) * dt + 
                              sigma * np.sqrt(dt) * W[:, t-1])
            
            # DataFrame des taux courts (années entières)
            df_paths = pd.DataFrame(
                paths[:, ::int(1/dt)],
                columns=range(0, self.T + 1),
                index=range(1, self.N + 1)
            )
            
            # Calculer les prix ZC pour chaque maturité
            dict_paths = {}
            
            for T_mat in range(1, self.T + 1):
                B = (1 - np.exp(-kappa * T_mat)) / kappa
                A = ((theta - (sigma**2) / (2 * kappa**2)) * (B - T_mat) - 
                     (sigma**2) * (B**2) / (4 * kappa))
                
                # Prix ZC pour chaque simulation
                df_path_temp = np.exp(A - B * df_paths)
                dict_paths[T_mat] = df_path_temp.copy()
            
            # Calcul du déflateur
            deflator_df = dict_paths[1]
            adj_df = deflator_df.cumprod(axis=1)
            dict_paths["Deflator"] = adj_df / deflator_df
            
            return dict_paths
        
        elif model_name == 'G2++':
            # À implémenter selon ton modèle G2++
            raise NotImplementedError("G2++ pas encore implémenté dans le moteur de simulation")
        
        return None
    
    def get_trajectories(self):
        """Retourne les trajectoires générées"""
        return self.trajectories
    
    def get_metadata(self):
        """Retourne les métadonnées de la simulation"""
        return self.metadata
    
    def get_seed(self):
        """Retourne la graine utilisée"""
        return self.seed
    
    def export_to_dict(self):
        """
        Export complet pour stockage dans session_state
        
        :return: Dict contenant toutes les données de simulation
        """
        return {
            'seed': self.seed,
            'trajectories': self.trajectories,
            'metadata': self.metadata,
            'correlation_matrix': self.correlation_matrix
        }