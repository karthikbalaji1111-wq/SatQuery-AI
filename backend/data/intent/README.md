# Intent data and model

This folder has the data for the small model that reads a question and decides which analysis it is asking for. That's all it does. It doesn't calculate anything, and it never sees the place or the date: those are taken out of the question first (they become PLACE and DATE) and handled by simple rules.


## The labels

The model picks one of these:

- NDVI: vegetation
- NDWI: water
- NDBI: built-up area
- SAR_BACKSCATTER: radar
- TEMPORAL_NDWI: water compared between two dates
- TRUE_COLOR: just show the image
- CLARIFICATION: the question doesn't say what to do, so ask the user
- UNSUPPORTED: something we can't do, like floods, temperature, counting things, depth or forecasting

Comparing vegetation or built-up area between two dates is handled by the rules, because the model has no label for it.


## The files

- satquery_intents_v2.jsonl: the training data for the model we ship (1,258 questions we wrote by hand)
- satquery_intents_v2_challenge.jsonl: 120 extra questions written before training, used only to test it
- evaluation_v2.json and evaluation_v2.md: the test results, made by the training script
- satquery_intents_v1.jsonl, satquery_intents_v1_challenge.jsonl and evaluation_v1.*: the older version, kept for history

The trained model itself is in backend/app/services/agent/intent_artifacts/satquery_intent_v2.json.gz. It's plain numbers in a JSON file, not code.


## Training it again

From the backend folder:

- run: uv sync
- run: uv run python scripts/train_intent_model.py

The same data always gives exactly the same model file. Training needs scikit-learn, but running the model only needs numpy.


## How it is used

The model only gets a say when it's at least 95% sure, and only if it doesn't disagree with something the rules already read clearly from the question. Otherwise the rules decide or the user is asked.

On the test questions and both sets of challenge questions, it made no confident mistakes, and no wrong analysis was ever run. It has only been tested on questions we wrote, not on real users' questions, and it only knows English.
