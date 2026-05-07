# Recommandation de Verres Optiques

## Changer de modele LLM

L'application utilise une couche d'abstraction unique: changez uniquement `LLM_PROVIDER` et les noms de modeles dans `.env`.

### Variables d'environnement

Copiez `.env.example` vers `.env`, puis adaptez:

- `LLM_PROVIDER=anthropic|ollama|openai_compat`
- `LLM_NLU_MODEL=...`
- `LLM_FORMAT_MODEL=...`
- `OLLAMA_BASE_URL=...`
- `OPENAI_BASE_URL=...`
- `ANTHROPIC_API_KEY=...`
- `OPENAI_API_KEY=...`

## Ajout de fiches produits

L'onglet `Ajouter un produit` de `app.py` permet de creer une fiche produit optique sous forme de formulaire, puis de generer un JSON structure selon le modele suivant :

```json
{
	"nom_produit": "Orma 1.50",
	"concept": "La lumiere sous controle",
	"plage_performance": {"min": -300, "max": 200},
	"avantages": [],
	"recommande_pour": "",
	"references": [],
	"notes": []
}
```

La fiche est enregistree dans `document_Produit/`, peut etre telechargee depuis l'interface, et est indexee dans la collection RAG choisie (`types_verres`, `indices`, `traitements` ou `couleurs`).

### Providers supportes

- Anthropic: Claude Haiku/Sonnet via SDK Anthropic
- Ollama local: via SDK OpenAI avec `base_url=http://localhost:11434/v1`
- OpenAI-compatible: LM Studio, vLLM, Groq, Together AI

### Installation Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Telecharger via https://ollama.ai

# Modeles recommandes
ollama pull mistral
ollama pull phi3:mini
ollama pull llama3:8b

# Verification
ollama list
curl http://localhost:11434/v1/models
```

### Lancer l'app

```bash
pip install -r requirements.txt
python rag_builder.py
LLM_PROVIDER=ollama python app.py
```

### Notes

- Le code metier ne depend pas du provider LLM.
- `core/decision_engine.py` lit exclusivement `core/dynamic_rules.json`.
- En mode `ollama`, aucune cle API n'est requise.

## Validation expert et historique

L'application principale `app.py` inclut maintenant une interface protegee pour l'expert :

- generation automatique de la recommandation ;
- creation d'un enregistrement d'historique en statut `en attente de correction` ;
- validation sans correction ou saisie d'une correction expert ;
- conservation de la recommandation automatique originale ;
- consultation de l'historique complet dans l'onglet `Historique`.

Les donnees sont stockees dans `corrections.db` avec deux tables :

- `corrections` : compatibilite avec les overrides expert deja existants ;
- `recommendation_history` : historique complet des recommandations generees, corrigees ou validees.

Aucune ancienne donnee n'est supprimee. Au demarrage, les anciennes corrections sont migrees dans l'historique si elles n'y sont pas deja.

### Version cloud avec Supabase

Pour laisser l'application ouverte a l'expert pendant un mois meme quand le laptop est ferme, utiliser un hebergement cloud pour Streamlit et une base Supabase PostgreSQL.

Dans Supabase :

1. Ouvrir le projet Supabase.
2. Cliquer sur `Connect`.
3. Si `Direct` affiche `Not IPv4 compatible`, choisir `Session Pooler` dans `Pooler settings`.
4. Copier la `Connection string` au format URI, pas les variables `NEXT_PUBLIC_*`.
5. Remplacer `[YOUR-PASSWORD]` par le mot de passe de la base Supabase.

La valeur `Direct` ressemble a ceci, mais elle peut ne pas fonctionner sur un reseau IPv4 :

```text
postgresql://postgres:VOTRE_MOT_DE_PASSE@db.xxxxx.supabase.co:5432/postgres
```

Pour Streamlit Cloud, utiliser de preference la valeur `Session Pooler`, qui ressemble a ceci :

```text
postgresql://postgres.xxxxx:VOTRE_MOT_DE_PASSE@aws-0-eu-west-3.pooler.supabase.com:5432/postgres
```

Dans Streamlit Cloud, ajouter ces secrets :

```toml
OPTI_RECO_PASSWORD = "un-mot-de-passe-fort"
DATABASE_URL = "postgresql://postgres.xxxxx:VOTRE_MOT_DE_PASSE@aws-0-eu-west-3.pooler.supabase.com:5432/postgres"
```

Ne pas utiliser les variables `NEXT_PUBLIC_*` : elles servent aux projets Next.js, pas a cette application Python/Streamlit.

Au premier demarrage, l'application cree automatiquement les tables `corrections` et `recommendation_history` dans Supabase.

### Visualiser la base de donnees

Le plus simple pour l'expert est l'onglet `Historique` dans l'application. Il affiche les donnees d'entree, la recommandation automatique, la correction expert et le statut.

Pour consulter directement SQLite, ouvrir le fichier :

```text
corrections.db
```

Option graphique recommandee : installer DB Browser for SQLite, puis ouvrir `corrections.db` et regarder la table `recommendation_history`.

```powershell
winget install DBBrowserForSQLite.DBBrowserForSQLite
```

Requete utile dans DB Browser ou sqlite3 :

```sql
SELECT
	id,
	created_at,
	updated_at,
	status,
	expert_name,
	input_json,
	recommendation_auto,
	recommendation_expert,
	notes
FROM recommendation_history
ORDER BY created_at DESC;
```

Sans logiciel externe, PowerShell peut afficher l'historique avec Python :

```powershell
.\.venv\Scripts\python.exe -c "from core.corrections_store import CorrectionsStore; import json; rows=CorrectionsStore().list_history(50); print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))"
```

### Lancer localement avec mot de passe

Depuis PowerShell, a la racine du projet :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
$env:OPTI_RECO_PASSWORD = "change-moi"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Acces sur la meme machine :

```text
http://localhost:8501/
```

Important : `localhost` signifie toujours "cette machine". Si vous donnez `http://localhost:8501/` a une autre personne, son navigateur cherchera l'application sur son propre PC, pas sur votre PC. Donc `localhost` ne fonctionne que sur la machine qui lance Streamlit.

Acces depuis un autre poste du meme reseau :

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "169.*" -and $_.IPAddress -ne "127.0.0.1"} | Select-Object IPAddress,InterfaceAlias
```

Puis ouvrir :

```text
http://ADRESSE_IP_DE_LA_MACHINE:8501/
```

Si l'autre poste ne peut pas ouvrir la page, autoriser le port 8501 dans le pare-feu Windows de la machine qui heberge l'application :

```powershell
New-NetFirewallRule -DisplayName "OptiReco Streamlit 8501" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
```

Tester depuis une autre machine Windows du meme reseau :

```powershell
Test-NetConnection ADRESSE_IP_DE_LA_MACHINE -Port 8501
```

Si `TcpTestSucceeded` vaut `True`, le lien reseau local fonctionne.

Le mot de passe par defaut est `expert123` si `OPTI_RECO_PASSWORD` n'est pas defini. Pour un usage reel, definir toujours `OPTI_RECO_PASSWORD` avant de lancer Streamlit.

### Garder le lien actif pendant environ un mois

Option recommandee si l'expert travaille sur le meme reseau : garder la machine locale allumee, branchée au secteur, avec le terminal Streamlit ouvert. Configurer Windows pour ne pas mettre la machine en veille.

Option recommandee si l'expert est a distance : utiliser un tunnel securise.

Cloudflare Tunnel temporaire :

```powershell
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8501
```

Cloudflare Tunnel stable avec domaine :

```powershell
cloudflared tunnel login
cloudflared tunnel create optireco-pfe
cloudflared tunnel route dns optireco-pfe optireco.votre-domaine.com
cloudflared tunnel run optireco-pfe
```

Ngrok :

```powershell
winget install ngrok.ngrok
ngrok config add-authtoken VOTRE_TOKEN_NGROK
ngrok http 8501
```

Pour un lien stable sur un mois avec ngrok, utiliser un domaine statique reserve dans ngrok puis lancer le tunnel vers `8501`.

### Tester

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m py_compile app.py core\decision_engine.py core\corrections_store.py
```
