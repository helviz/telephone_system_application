SAFETY_MESSAGES = {
    "en": {
        "medical_emergency": "This may be an emergency. Please stay calm. I am connecting you to an operator now.",
        "violence": "Your safety is the priority. Move to a safe place if you can. I am connecting you to an operator now.",
        "self_harm": "I am sorry you are feeling this way. Please move away from anything that could harm you. I am connecting you to an operator now.",
        "operator_request": "Please stay on the line. I am connecting you to an operator now.",
        "unsafe_llm_output": "Please stay on the line. I am connecting you to an operator now.",
        "asr_failure": "Sorry, I did not hear you clearly. Please repeat, or press 0 to speak to an operator.",
        "asr_failure_final": "I am still having trouble hearing you. Please press 0 to speak to an operator, or repeat your message slowly.",
        "transfer": "Please stay on the line. I am connecting you to an operator now.",
        "transfer_failed": "I could not connect the operator automatically. Please call your local emergency number or try again.",
    },
    "fr": {
        "medical_emergency": "Cela peut être une urgence. Veuillez rester calme. Je vous connecte à un opérateur maintenant.",
        "violence": "Votre sécurité est la priorité. Allez dans un endroit sûr si vous le pouvez. Je vous connecte à un opérateur maintenant.",
        "self_harm": "Je suis désolé que vous vous sentiez ainsi. Éloignez-vous de tout ce qui pourrait vous blesser. Je vous connecte à un opérateur maintenant.",
        "operator_request": "Veuillez rester en ligne. Je vous connecte à un opérateur maintenant.",
        "unsafe_llm_output": "Veuillez rester en ligne. Je vous connecte à un opérateur maintenant.",
        "asr_failure": "Désolé, je ne vous ai pas bien entendu. Veuillez répéter, ou appuyez sur zéro pour parler à un opérateur.",
        "asr_failure_final": "J'ai encore du mal à vous entendre. Appuyez sur zéro pour parler à un opérateur, ou répétez lentement votre message.",
        "transfer": "Veuillez rester en ligne. Je vous connecte à un opérateur maintenant.",
        "transfer_failed": "Je n'ai pas pu connecter l'opérateur automatiquement. Veuillez appeler le numéro d'urgence local ou réessayer.",
    },
    "sw": {
        "medical_emergency": "Hii inaweza kuwa dharura. Tafadhali tulia. Ninakuunganisha na mhudumu sasa.",
        "violence": "Usalama wako ndio muhimu. Nenda mahali salama kama unaweza. Ninakuunganisha na mhudumu sasa.",
        "self_harm": "Pole kwa hali unayopitia. Tafadhali kaa mbali na kitu chochote kinachoweza kukudhuru. Ninakuunganisha na mhudumu sasa.",
        "operator_request": "Tafadhali baki kwenye simu. Ninakuunganisha na mhudumu sasa.",
        "unsafe_llm_output": "Tafadhali baki kwenye simu. Ninakuunganisha na mhudumu sasa.",
        "asr_failure": "Samahani, sikukusikia vizuri. Tafadhali rudia, au bonyeza sifuri kuzungumza na mhudumu.",
        "asr_failure_final": "Bado nina shida kukusikia. Bonyeza sifuri kuzungumza na mhudumu, au rudia polepole.",
        "transfer": "Tafadhali baki kwenye simu. Ninakuunganisha na mhudumu sasa.",
        "transfer_failed": "Sikuweza kukuunganisha na mhudumu moja kwa moja. Tafadhali piga namba ya dharura ya eneo lako au jaribu tena.",
    },
}


def get_safety_message(lang: str, key: str) -> str:
    return SAFETY_MESSAGES.get(lang, SAFETY_MESSAGES["en"]).get(key) or SAFETY_MESSAGES["en"].get(key, SAFETY_MESSAGES["en"]["transfer"])
