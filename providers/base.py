"""Interface commune à tous les providers LLM."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Interface commune à tous les providers LLM, qu'ils soient locaux/gratuits
    (Ollama) ou distants/payants (Gemini, Haiku).

    Chaque provider concret doit implémenter evaluer_un(), qui a exactement le
    même contrat que l'ancienne fonction evaluer_sentiment_llm() du script
    principal, pour permettre un remplacement sans changement de comportement.
    """

    @abstractmethod
    def evaluer_un(self, texte_article, nom_entreprise, prompt_version="v1"):
        """
        Évalue le sentiment d'un article pour une entreprise donnée.

        Retourne un tuple (resultat, statut) :
          - statut == "ok"     -> resultat est un dict {"note_llm": int, "justification": str}
          - statut == "failed" -> resultat est None (échec définitif après retries,
                                   ou erreur non retryable). Ne JAMAIS interpréter
                                   ça comme une note 0.

        Peut lever llm_common.FatalLLMError pour les erreurs de config/auth
        (400/401/403/404), qui doivent stopper le run entier plutôt que d'être
        comptées comme un échec par article.
        """
        raise NotImplementedError


class BatchLLMProvider(LLMProvider):
    """
    Mixin optionnel pour les providers distants qui supportent en plus une
    soumission par lot (batch), à -50% de coût mais avec un délai pouvant
    aller jusqu'à 24h. evaluer_un() reste disponible (mode synchrone), ces
    trois méthodes ajoutent le mode batch en 3 étapes : soumettre, attendre,
    récupérer.

    Implémenté par GeminiProvider et HaikuProvider.
    """

    @abstractmethod
    def soumettre_batch(self, requetes, prompt_version="v1"):
        """
        requetes : liste de dicts {"custom_id": str, "texte_article": str, "nom_entreprise": str}.
        custom_id doit être unique dans le batch (voir llm_common.construire_custom_id).

        Retourne l'identifiant de batch (str) à conserver pour statut_batch()/
        recuperer_resultats(). Peut lever FatalLLMError si la soumission échoue
        pour une raison de configuration/auth (ex: clé API manquante/invalide).
        """
        raise NotImplementedError

    @abstractmethod
    def statut_batch(self, batch_id):
        """
        Retourne 'in_progress' ou 'ended'. 'ended' signifie que toutes les
        requêtes du batch ont été traitées (certaines individuellement en échec
        éventuellement) et que recuperer_resultats() peut être appelé.
        Peut lever FatalLLMError si le batch lui-même a échoué (ex: job Gemini
        en JOB_STATE_FAILED/CANCELLED/EXPIRED).
        """
        raise NotImplementedError

    @abstractmethod
    def recuperer_resultats(self, batch_id):
        """
        À appeler uniquement après statut_batch(batch_id) == 'ended'.

        Retourne un dict {custom_id: (resultat, statut)} avec exactement le même
        contrat par entrée que evaluer_un() : statut 'ok' -> resultat =
        {"note_llm": int, "justification": str} ; statut 'failed' -> resultat = None.
        """
        raise NotImplementedError
