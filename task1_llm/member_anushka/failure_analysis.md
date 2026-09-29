# Sequence Model Failure Analysis

Model: `checkpoints/best_model.pt` (run `20260929_225852`, sequence length 512, epoch 20). All snippets are copied verbatim from `outputs/generated_samples.txt`. The prompt was `"Once upon a time, "`.

## Failure Case 1: Repetition / circular actions
Generated snippet (T=0.5):

> One day, she saw a big box in the garden. She wanted to open it and see what was inside. She opened it and started to open it. Inside was a big box with a ball. Lily was so happy and she picked it up. She was so excited to ope

Failure type: Repetition (the same action and object loop back on themselves)

Observation: The story keeps returning to "open the box". She *opened it and started to open it*, the box contains *a big box*, and after already opening it she is *so excited to open* it again. At T=0.5 the model mostly picks the most probable continuation, and in TinyStories "box … open … inside" is one of the most frequent patterns. The model has no record of which events already happened, so it re-predicts the likeliest event. Compared with the baseline (seq 256), the repeated 4-gram rate at T=1.0 dropped from 0.187 to 0.102, so repetition is reduced but still visible at low temperature.

Possible improvement: At decode time, use a repetition penalty or n-gram blocking, or top-p sampling instead of low-temperature sampling. On the training side, a word/BPE tokenizer would let the model reason over events rather than characters.

## Failure Case 2: Loss of coherence (speaker and entity confusion)
Generated snippet (T=1.0):

> Timmy didn't like it wild, but he told Billy, "No, Timmy, it's not yours."
>
> Billy nodded and climbed into Mia's hands.

Also from the greedy sample:

> One day, she saw a big bird in the sky. She wanted to fly it and see what was inside.

Failure type: Loss of coherence / hallucinated entities

Observation: Each sentence is grammatical, but the meaning does not hold together. Timmy speaks to Billy yet addresses himself ("No, Timmy"). A new character "Mia" appears from nowhere. Billy does something physically impossible (*climbed into Mia's hands*). In the greedy sample a *bird* is something you *fly* and look *inside*, which is clearly a template about kites or boxes applied to the wrong object. The model has learned very strong local phrase templates ("wanted to … and see what was inside", "told X, 'No, …'") but not who is speaking or what an object can do. A 512-character context does contain the earlier sentences, but a 6-layer character model cannot reliably follow who each name and pronoun refers to.

Possible improvement: Use more depth or width, or subword tokens so attention works over whole words and names, plus more training data. Dialogue and speaker errors could also be measured with targeted prompts (e.g. two named characters talking) to check a fix.

## Failure Case 3: Broken spelling and grammar at high temperature
Generated snippet (T=1.5):

> Once upon a time, there two friends, Mumy and Sally,gas and Sandy. [...]
>
> "Look, Sandy?" DadDadonly asks.
>
> "Yes, it's a special yutdent: trying tonide!"
>
> DSandy and SaY play

Failure type: Broken grammar and non-word spelling (character-level breakdown)

Observation: At T=1.5 the output distribution is flattened, so low-probability characters get sampled often. With a character-level model this breaks *inside* words: non-words appear (*yutdent*, *tonide*, *Mumy*), names are glued together (*DadDadonly*, *DSandy*), the verb is missing ("there two friends"), and there's no space after a comma (*Sally,gas*). At T=1.0 and below, spelling stays essentially perfect (82% top-1 character accuracy on validation). So the model knows how to spell and has the right characters ranked first, but high temperature lets wrong ones through. Distinct-n goes up with temperature while quality falls, so diversity metrics alone would reward this failure.

Possible improvement: Use a moderate temperature (0.7–0.9) with top-k/top-p truncation, which removes the low-probability tail while keeping diversity. Temperature could also be tuned on held-out text against a combined diversity/quality score rather than Distinct-n alone.
