# AFFGEN
Code implementation for EMNLP-findings 2023 paper "Affective and Dynamic Beam Search for Story Generation" 

Below is a sample README.md content for your project, detailing the purpose and setup instructions. Ensure to customize paths, links, and instructions as per your project's specific requirements and directory structure.

---

# Project Name

This project leverages natural language processing techniques to analyze stories and predict literary events, enriching text with emotional and narrative insights. It integrates GPT-2 for text generation, a bandit model for policy making, and a Roberta-based classifier for identifying literary event triggers.

## Requirements

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- Pip for installing Python packages
- Access to command-line interface or terminal

## Setup Instructions

Follow these steps to get your environment ready:

1. **Clone the Repository**
   ```
   git clone [your-repository-link]
   cd [repository-name]
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download Required Models and Datasets**

   - ROCStories Dataset: Download the dataset from [https://cs.rochester.edu/nlp/rocstories/](https://cs.rochester.edu/nlp/rocstories/) and save it to `story_path`.
   
   - GPT-2 Model: Download GPT-2-large from Huggingface and finetune it on ROCStories. Save the finetuned model to `GPT2_finetuned_ROC`.
   
   - Roberta Model: Download Roberta from Huggingface and train it with the dataset from [https://github.com/dbamman/litbank](https://github.com/dbamman/litbank) for identifying literary event triggers. The model should be saved to `event_trigger_path`. We will release the train script for event trigger soon.
   
   - NRC Emotion Lexicon: Download from [https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) and save the lexicon to `NRC_lexicon_path`.

4. **Configuration**

   Update the `Config` class in your project with the paths to the downloaded models and datasets. Here is an example of what the `Config` class.

5. **Running the Project**

   After setting up the configuration, you can run your project by executing the main script. Adjust the command based on your project's structure.

   ```
   python inference.py
   ```


