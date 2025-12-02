"""
Script pour démarrer le serveur FastAPI et tester les endpoints.

Ce script:
1. Démarre le serveur en arrière-plan
2. Attend que le serveur soit prêt
3. Teste les endpoints
4. Affiche les résultats
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://127.0.0.1:8000"
SERVER_SCRIPT = "deployment/fastapi_server.py"


def check_server_running():
    """Vérifie si le serveur est déjà en cours d'exécution."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return True
    except:
        return False


def start_server():
    """Démarre le serveur FastAPI."""
    print("=" * 60)
    print("Démarrage du serveur FastAPI...")
    print("=" * 60)
    
    if check_server_running():
        print("Le serveur est déjà en cours d'exécution!")
        return True
    
    try:
        # Démarrer le serveur
        process = subprocess.Popen(
            [sys.executable, SERVER_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Attendre que le serveur démarre
        print("Attente du démarrage du serveur (max 10 secondes)...")
        for i in range(10):
            time.sleep(1)
            if check_server_running():
                print(f"[OK] Serveur demarre avec succes apres {i+1} secondes")
                return True
            print(f"  Tentative {i+1}/10...")
        
        print("[ERROR] Le serveur n'a pas demarre dans les temps")
        return False
        
    except Exception as e:
        print(f"[ERROR] Erreur lors du demarrage: {e}")
        return False


def run_health_test():
    """Teste le endpoint /health."""
    print("\n" + "=" * 60)
    print("Test du endpoint /health")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"Status HTTP: {response.status_code}")
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model_loaded')}")
        print(f"Model type: {data.get('model_type', 'N/A')}")
        
        if data.get('status') == 'ok' or data.get('status') == 'healthy':
            print("[OK] Health check reussi!")
            return True
        else:
            print("[WARNING] Health check retourne un statut inattendu")
            return True  # Le serveur fonctionne même si le modèle n'est pas chargé
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Impossible de se connecter au serveur")
        print("  Assurez-vous que le serveur est demarre")
        return False
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return False


def run_load_model_test():
    """Teste le chargement d'un modèle."""
    print("\n" + "=" * 60)
    print("Test du chargement d'un modèle")
    print("=" * 60)
    
    model_path = "models/baselines/logistic_regression.pkl"
    
    if not Path(model_path).exists():
        print(f"⚠ Modèle non trouvé: {model_path}")
        print("  Passez cette étape ou entraînez d'abord un modèle")
        return False
    
    try:
        response = requests.post(
            f"{BASE_URL}/load_model",
            params={
                "model_path": model_path,
                "model_type": "logistic_regression",
                "threshold": 0.5
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Message: {data.get('message')}")
        
        if data.get('status') == 'success':
            print("[OK] Modele charge avec succes!")
            return True
        else:
            print("[ERROR] Echec du chargement du modele")
            return False
            
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return False


def run_predict_test():
    """Teste le endpoint /predict."""
    print("\n" + "=" * 60)
    print("Test du endpoint /predict")
    print("=" * 60)
    
    from datetime import datetime, timedelta
    
    # Créer des données de test
    time_series = []
    for i in range(48):
        time_series.append({
            "timestamp": (datetime.now() - timedelta(hours=48-i)).isoformat(),
            "views": 1000 + i * 10,
            "likes": 50 + i * 2,
            "comments": 10 + i,
            "shares": 5 + i,
        })
    
    request_data = {
        "id": "test_video_001",
        "time_series": time_series
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 503:
            print("[WARNING] Modele non charge. Chargez d'abord un modele avec /load_model")
            return False
        
        response.raise_for_status()
        data = response.json()
        
        print(f"ID: {data.get('id')}")
        print(f"Score: {data.get('score'):.4f}")
        print(f"Label: {data.get('label')}")
        print(f"Is Fake: {data.get('is_fake')}")
        print(f"Model Type: {data.get('model_type')}")
        print(f"Threshold: {data.get('threshold')}")
        
        print("[OK] Prediction reussie!")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Erreur HTTP {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        return False


def main():
    """Fonction principale."""
    print("=" * 60)
    print("Test du serveur FastAPI")
    print("=" * 60)
    
    # Démarrer le serveur
    if not start_server():
        print("\n[ERROR] Impossible de demarrer le serveur")
        print("\nPour demarrer manuellement:")
        print(f"  python {SERVER_SCRIPT}")
        return
    
    # Tester les endpoints
    results = []
    
    results.append(("Health Check", run_health_test()))
    results.append(("Load Model", run_load_model_test()))
    results.append(("Predict", run_predict_test()))
    
    # Résumé
    print("\n" + "=" * 60)
    print("Résumé des tests")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[OK] PASSE" if passed else "[ERROR] ECHOUE"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n[OK] Tous les tests sont passes!")
    else:
        print("\n[WARNING] Certains tests ont echoue")
        print("\nLe serveur est toujours en cours d'exécution.")
        print(f"Accédez à la documentation: {BASE_URL}/docs")
        print(f"Pour arrêter le serveur, appuyez sur Ctrl+C dans le terminal où il tourne")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur")
    except Exception as e:
        print(f"\n[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()

