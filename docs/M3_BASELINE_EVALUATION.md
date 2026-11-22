# M3 baseline evaluation

ChatGPT executed the project scripts from M3 commit `11139dd` and reviewed the outputs at Nedim Omerčević's request. The team should reproduce and review the results before submission.

## Reproduction

Run `python src/preprocess.py`, `python src/eda.py`, and `python src/baseline.py` from the repository root. This run used the bundled sample because Project Gutenberg was undergoing maintenance. It used pandas 2.2.3, scikit-learn 1.8.0 (the repository pins 1.5.2), and seaborn 0.13.2. The sample yielded 22 passages (17 training, 2 validation, 3 test). The baseline searched the training passages and was evaluated on the separate 15-prompt team set.

| Metric | Mean |
|---|---:|
| Retrieval similarity | 0.1006 |
| ROUGE-1 F1 | 0.1171 |
| Latency on the test machine | 0.375 ms |

## Qualitative review

| Prompt | Retrieved passage opening | Similarity | Assessment |
|---|---|---:|---|
| When and where was Marcus Aurelius born? | Everything is change. | 0.0000 | Does not supply a date or place. |
| How should I respond to something I cannot control? | Everything is change. | 0.0000 | Broadly related to change but provides no actionable answer about control. |
| What matters most in daily life? | Do not disturb yourself by imagining your whole life at once. | 0.3360 | Partially relevant to focusing on the present. It does not address virtue or duty. |
| Why should a person practice patience? | Practice yourself in little things, and then proceed to greater. | 0.3407 | Describes gradual practice without explaining patience or anger. |
| How should I deal with another person's insult? | Everything is change. | 0.0000 | Does not address the insult or how to respond. |
| Did you write a review of the iPhone for a Roman newspaper? | Everything is change. | 0.0000 | Returns an unrelated passage instead of recognizing the anachronism or declining. |

Several zero-similarity queries returned the same passage. The electric-car prompt scored 0.4283 yet retrieved a passage about waking in the morning. Similarity and ROUGE measure overlap and do not establish answer quality. A reasonable next experiment is to tune a rejection threshold on validation questions and include unsupported questions when testing it.

For the full-corpus run, use `python src/preprocess.py --download` and rerun the analysis and evaluation. Save full-corpus results separately so they can be distinguished from this preliminary sample run.

## Additional baseline response review

The following examples were reviewed with AI assistance using the
preliminary TF-IDF results from the bundled Meditations sample.

| Question | Retrieved passage opening | Similarity | Rating | Reason |
|---|---|---:|---|---|
| What Roman empire role did Marcus Aurelius hold? | Everything is change. | 0.0000 | Irrelevant | The passage discusses change and does not identify Marcus Aurelius as Roman emperor. |
| What is the purpose of hardship? | Everything is change. | 0.0000 | Irrelevant | The passage says that things change but does not explain how hardship can develop judgment, courage, or moderation. |
| What did you think of electric cars? | When you arise in the morning, think of what a privilege it is to be alive. | 0.4283 | Irrelevant | The passage is about gratitude and does not address electric cars or recognize the historical mismatch. |

The first two questions retrieved the same passage despite having different
topics and a similarity score of zero. The electric-car question scored
0.4283 and still retrieved an irrelevant passage. These examples show why
we should evaluate whether a passage answers the question alongside its
retrieval similarity score.
