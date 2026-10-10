# SQL Agent

Posez une question sur vos données en langage courant. SQL Agent trouve les bonnes tables, écrit la requête SQL, l'exécute et répond avec les chiffres clés, un graphique et le tableau complet.

## Bases de données

Choisissez une base dans le sélecteur en haut à gauche. Changer de base démarre une nouvelle conversation.

| Base | Contenu |
|---|---|
| **Chinook** | Un magasin de musique en ligne : artistes, albums, morceaux, genres, playlists, clients, factures |
| **Northwind** | Une entreprise de négoce alimentaire : produits, catégories, fournisseurs, clients, commandes, employés, transporteurs |

## Comment il répond

1. **Il comprend la question.** Une relance comme *« et en 2022 ? »* est réécrite en question complète.
2. **Il sélectionne les données.** Les tables et colonnes utiles sont retrouvées dans le schéma.
3. **Il écrit la requête.** Il planifie la requête, écrit le SQL et le vérifie. Seules les requêtes `SELECT` en lecture seule sont autorisées. Si la base rejette la requête, l'agent lit l'erreur et la corrige tout seul.
4. **Il affiche le résultat.** Vous obtenez une réponse courte, et une fiche de résultat avec :
   - **Graphique** : une courbe pour les évolutions dans le temps, des barres pour les classements et les répartitions, un grand chiffre pour une valeur unique
   - **Tableau** : toutes les lignes, triables en cliquant sur l'en-tête d'une colonne
   - **SQL** : la requête exacte qui a été exécutée
   - **CSV** : un téléchargement des résultats

Dépliez **SQL Agent** au-dessus d'une réponse pour suivre chaque étape.

## Conseils

- Dites ce que vous voulez mesurer et comment le regrouper : *« chiffre d'affaires par pays »*, *« commandes par mois en 2022 »*.
- Posez des questions de relance : l'agent se souvient de la conversation.
- Si l'agent a dû interpréter un terme, il le signale à la fin de sa réponse.
- Au plus 100 lignes sont lues par requête. La réponse vous indique quand les résultats sont partiels.
