# Sequence Model Failure Analysis

Model: `checkpoints/best_model.pt` (run `20260929_213922`). All snippets are copied verbatim from `outputs/generated_samples.txt`. The prompt was `"Once upon a time, "`.

## Failure Case 1: Repetition
Generated snippet (greedy and T=0.5):

> Lily said, "I want to play with the box, but I want to play with it."

> "Hi, I am Lily. I am Lily. You are very smart. I want to play with you."

> The two friends hugged each other and hugged each other. *(T=1.0)*

Failure type: Repetition (phrase-level looping)

Observation: The model repeats high-probability phrases, and it does this more at low temperature. Greedy and T=0.5 decoding always pick the most likely continuation, and in TinyStories "I want to play with…" and "I am Lily" are extremely frequent. Once the phrase is in the context, the model's own output makes the phrase even more likely, so it loops. The "but" in the first example introduces a contrast that never arrives. The repeated 4-gram rate of 0.187 on the T=1.0 sample measures this. A character-level model predicts one letter at a time and has no explicit notion of "I already said this".

Possible improvement: Use a repetition penalty or n-gram blocking at decode time, or top-p sampling instead of greedy decoding. On the training side, a word/BPE-level tokenizer would give the model a longer effective context in words.

## Failure Case 2: Loss of coherence / entity drift (hallucination)
Generated snippet (greedy, T=0.5, T=1.0):

> One day, she saw a big box in the garden. It was so big and shiny and had a long tail.

> One day, she saw a big tree in the sky.

> Lily went to her friend Benny to play with her. Benny saw a big tree with clothes and Snowy reached out his hand. Snowy wanted to see more dangerous clothes.

Failure type: Loss of coherence (semantic contradiction) and hallucinated entities

Observation: Each sentence is grammatical, but the meaning does not hold together. A *box* gets a *tail*, a *tree* is *in the sky*, and a new character "Snowy" appears mid-story and takes over the action from Benny. The model has learned local word statistics ("big and shiny and had a long tail" is a common TinyStories phrase about animals) but not a consistent world state. The 256-character context covers only about 50 words, so the model only weakly links an object or character from a few sentences back to the current prediction.

Possible improvement: Use a longer context window (512+), or move to subword tokens so the same window covers more of the story. A larger model or more training data would help it learn object–attribute consistency.

## Failure Case 3: Broken grammar at high temperature
Generated snippet (T=1.5, plus one from T=1.0):

> One morning, the girl loved to look deep quietly that she kept walking. She wrapped up the cabin and sat at the table. [...] But, when she tried to take it, be brave.

> They decided to play in reasons all day *(T=1.0)*

Failure type: Broken grammar / syntax

Observation: At T=1.5 the softmax is flattened, so low-probability characters and words are sampled far more often. The output drifts into invalid structures: "loved to look deep quietly that" mixes a "so … that" construction with the wrong adverbs, "be brave" is an imperative where a main clause is needed, and "play in reasons" is a real word in a slot where it does not fit. The words themselves are still spelled correctly, which shows the model learned spelling well (79.7% top-1 character accuracy). Syntax across a whole clause is where it is weaker. The Distinct-n scores go up with temperature, but coherence goes down.

Possible improvement: Use a moderate temperature (0.7–0.9) with top-k/top-p truncation, which removes the low-probability tail while keeping diversity. Temperature could also be tuned on a held-out set against a combined diversity/quality score.
