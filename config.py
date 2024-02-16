class Config:
    story_path = ""  # download from https://cs.rochester.edu/nlp/rocstories/

    GPT2_finetuned_ROC = ""  # Download GPT2-large from Huggingface and finetune it on ROCStories

    policy_maker_path = ""  # The path to your trained bandit model

    contexts_path = "data_assets/demo_contexts.p"

    # Download RoberTa model from Huggingface.
    # And train a "whether the next token in the current sentence is literary event trigger" classifier using the dataset https://github.com/dbamman/litbank.
    # The training script is coming soon : )
    event_trigger_path = ""

    # download from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
    NRC_lexicon_path = ""

    output_path = ""
