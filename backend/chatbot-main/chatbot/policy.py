import numpy as np
import time
import random
import requests
import os
import openai

from .intent import Intent


class DialogPolicy:
    def __init__(self):
        self.intent_to_slot = {
            Intent.ASK_GET: ["size", "type", "fabric", "pattern", "color"],
            Intent.INFORM_DISAMBIGUATE: [],
            Intent.INFORM_GET: ["size", "type", "fabric", "pattern", "color"],
            Intent.INFORM_REFINE: ["size", "type", "fabric", "pattern", "color"],
            Intent.REQUEST_ADD_TO_CART: ["item"],
            Intent.REQUEST_COMPARE: ["item", "with_item"],
            Intent.REQUEST_GET: ["size", "type", "fabric", "pattern", "color"]
        }

        self.intent_to_answer = {
            Intent.ASK_GET: [
                "I apologize, but I'm not programmed to provide information about the clothes yet.",
                "Unfortunately, I'm unable to assist with questions specifically about the clothes on display.",
                "I'm sorry, but I don't have the capability to answer questions about the clothes in the display.",
                "Regrettably, I'm unable to provide information about the clothes in display.",
                "Unfortunately, I don't have access to information about the clothes showcased.",
                "I regret to inform you that I'm not programmed to answer specific questions about the clothes on display.",
            ],
            Intent.INFORM_DISAMBIGUATE: [
                "I apologize for the inconvenience, but I cannot identify or provide information about specific clothes at this time.",
                "Unfortunately, I am unable to assist with questions specifically related to the clothes on display as I lack the necessary knowledge.",
                "I'm sorry, but I don't have the capability to answer questions about the clothes in the display since I cannot identify them.",
                "Regrettably, I am unable to provide information about the clothes in display as I don't have the means to identify them.",
                "Unfortunately, I don't have access to information about the clothes showcased, making it impossible for me to provide details about them.",
                "I regret to inform you that I'm not programmed to answer specific questions about the clothes on display since I cannot identify them.",
            ],
            Intent.INFORM_GET: [],
            Intent.INFORM_REFINE: [],
            Intent.REQUEST_ADD_TO_CART: [
                "I'm sorry, but I don't have a shopping cart feature yet.",
                "Unfortunately, I'm unable to assist with request to add to cart as I lack a shopping cart functionality.",
                "I apologize, but I don't have the capability to add items to the cart as I don't possess a shopping cart feature.",
                "Regrettably, I'm unable to add clothes to cart since I don't have a shopping cart feature.",
            ],
            Intent.REQUEST_COMPARE: [
                "I apologize for any confusion, but I'm unable to compare the items on display as I don't have the capability to track them individually.",
                "I'm sorry, but I can't provide a comparison of the items on display because I don't have the ability to track them.",
                "Unfortunately, I don't have the capability to compare the items on display as I can't track them individually.",
                "Regrettably, I'm unable to perform comparisons of the items on display since I can't track them individually.",
                "I'm sorry, but I don't have access to the necessary information to compare the items on display as I can't track them individually.",
                "I regret to inform you that I'm unable to compare the items on display because I don't have the ability to track them individually.",
            ],
            Intent.REQUEST_GET: []
        }

        self.ask_can_assist = (
            "I can assist you in filtering the catalog based on your preferences. "
            "Please provide me with the specific fabric, pattern, size, color, or type you are looking for, "
            "and I will help narrow down the options to match your requirements."
        )

        self.recommend_text = [
            "Perhaps you'd like these?",
            "What do you think about these?"
        ]

        self.entropy_text = [
            "What is your preference about {}?",
            "Do you have any preference about {}?",
            "What {} do you like?"
        ]

        self.instead = [
            "Instead...",
            "Alternatively...",
            "As an alternative",
        ]

        # OpenAI v0.28.1 configuration (safe: if no key, code still runs)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # -------------------------
    # OpenAI helpers (controlled)
    # -------------------------

    def _get_condition(self) -> str:
        # C0 = low anthropomorphism, C1 = high anthropomorphism
        return (os.getenv("CHAT_STYLE") or "C0").strip().upper()

    def _openai_enabled(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _slots_summary(self, state) -> str:
        # Build a safe, factual summary from filled slots only.
        slots = state.get("slots", {}) or {}
        parts = []
        for k in ["type", "fabric", "pattern", "size", "color"]:
            vals = slots.get(k, [])
            if vals:
                parts.append(f"{k}={', '.join(vals[:2])}")
        return "; ".join(parts) if parts else "no explicit preferences yet"

    def _rewrite_with_openai(self, original_text: str, state, is_reco_moment: bool) -> str:
        """
        Rewrites the *same meaning* in a style controlled by CHAT_STYLE.
        - C0: neutral / concise / no emotions
        - C1: warm / conversational / first-person allowed
        No new facts. No product invention.
        Uses OpenAI SDK v0.28.1.
        """
        if not self._openai_enabled():
            return original_text

        condition = self._get_condition()
        slots_summary = self._slots_summary(state)

        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        try:
            temperature = float(os.getenv("OPENAI_TEMPERATURE") or "0.2")
        except Exception:
            temperature = 0.2

        if condition == "C1":
            system = (
                "You are a helpful shopping assistant. "
                "Rewrite the message to be warm, friendly, and conversational. "
                "You may use first-person phrasing. "
                "Do NOT invent products, features, policies, or any new facts. "
                "Keep it short (1-2 sentences). No emojis."
            )
        else:
            system = (
                "You are a neutral shopping assistant. "
                "Rewrite the message in a factual, concise style. "
                "Do NOT express emotions. "
                "Do NOT invent products, features, policies, or any new facts. "
                "Keep it short (1 sentence if possible)."
            )

        # At recommendation moment: add a *safe* justification based only on slots (no product names).
        extra = ""
        if is_reco_moment:
            extra = (
                "\nAdd (or keep) a brief justification ONLY using the user's stated preferences. "
                f"Known preferences: {slots_summary}. "
                "If there are no preferences, mention you will ask a quick question to refine preferences."
            )

        try:
            resp = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Original message:\n{original_text}\n{extra}"},
                ],
            )
            txt = resp["choices"][0]["message"]["content"].strip()
            return txt if txt else original_text
        except Exception:
            # Fail-safe: never block the chatbot if OpenAI fails
            return original_text

    # -------------------------
    # Core dialog policy logic
    # -------------------------

    def fill_slots(self, state, intent, entities):
        for slot in self.intent_to_slot[intent]:
            if slot not in state['slots']:
                state['slots'][slot] = []

        for ent, label in entities:
            if label in state['slots'] and ent not in state['slots'][label]:
                state['slots'][label].append(ent)

    def decide(self, state, intent, elapsed):
        actions = []
        shouldRecommend = False

        if intent in [Intent.ASK_GET, Intent.INFORM_DISAMBIGUATE, Intent.REQUEST_COMPARE]:
            actions.append({"action": "answer", "text": random.choice(self.intent_to_answer[intent])})
            if elapsed >= 30:
                shouldRecommend = True
                actions.append({"action": "answer", "text": random.choice(self.instead)})
            else:
                actions.append({"action": "answer", "text": self.ask_can_assist})
        elif intent in [Intent.REQUEST_GET, Intent.INFORM_GET, Intent.INFORM_REFINE]:
            shouldRecommend = True

        if shouldRecommend:
            # IMPORTANT: keep this action so the FRONTEND calls the recommender at the right time.
            actions.append({"action": "recommend"})
            actions.append({"action": "answer", "text": random.choice(self.recommend_text)})

            entropyResponse = requests.post(os.getenv('RECOMM_API') + "/entropy", json={"state": state})
            if entropyResponse.ok:
                actions.extend(self.entropy(state, entropyResponse.json().get('entropy', {})))

        return actions

    def _postprocess_actions(self, actions, state):
        """
        Apply OpenAI rewriting ONLY to 'answer' actions.
        - Detect the recommendation moment: the first 'answer' after a 'recommend' action.
        """
        seen_reco = False
        for a in actions:
            if a.get("action") == "recommend":
                seen_reco = True
                continue
            if a.get("action") == "answer" and "text" in a:
                is_reco_moment = seen_reco
                # Only the first answer right after 'recommend' gets the special reco explanation
                if seen_reco:
                    seen_reco = False
                a["text"] = self._rewrite_with_openai(a["text"], state, is_reco_moment=is_reco_moment)
        return actions

    def answer(self, state, text, intent, entities):
        intent_index = Intent(np.argmax(intent))
        now = time.time()
        elapsed = now - state['turns'][-1]['time'] if len(state['turns']) else 0

        state['turns'].append({
            "self": True,
            "time": now,
            "data": [{"action": "ask", "text": text, "intent": intent, "entities": entities}]
        })

        self.fill_slots(state, intent_index, entities)

        actions = self.decide(state, intent_index, elapsed)

        # OpenAI rewriting (controlled by CHAT_STYLE), without changing the recommend trigger.
        actions = self._postprocess_actions(actions, state)

        state['turns'].append({"self": False, "time": time.time(), "data": actions})
        return actions, state

    def entropy(self, state, entropy):
        actions = []
        entities = [s for s in state['slots'] if not state['slots'][s]]
        if len(entities):
            max_entropy = 0
            entity = None
            for e in entities:
                if e in entropy and entropy[e] > max_entropy:
                    max_entropy = entropy[e]
                    entity = e
            if max_entropy > 0 and entity:
                text = random.choice(self.entropy_text)
                actions.append({"action": "answer", "text": text.format(entity)})
        return actions
