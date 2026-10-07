# Failure / Error Analysis — Task 1 — LLM from Scratch

**Member:** Nikhil (Member B)

The assignment asks for **three failure cases** from my generated text, each with the generated snippet, the failure type and my observation.

## Evidence pack (facts only)
Samples come from `outputs/gpt_nikhil_2L8H_B_20261001-0025/samples.txt` (5 prompts; greedy plus three temperature-0.9 samples each). The five samples below are the ones with the highest repeated word 4-gram rate inside the sample. Overall rates: 0.260 for greedy and 0.000 for temperature 0.9. Greedy distinct-1/2/3: 0.151 / 0.242 / 0.283.

### Candidate 1 — [greedy] prompt: 'Lily wanted to'
Repeated word 4-gram rate in this sample: 0.36

> Lily wanted to play with her toys and her friends. She was so happy that she could have a big smile on her face.
> The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy.
> The little girl was so happy that she had to be careful with her friends. She was so happy that she could have a big smile on her face.
> The little girl was

### Candidate 2 — [greedy] prompt: 'One day, a big bear'
Repeated word 4-gram rate in this sample: 0.25

> One day, a big bear came to visit him. He was so happy that he had to be careful with his friends.
> The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy.
> The little girl was so happy that she had to be careful with her friends. She was so happy that she could have a big smile on her face.
> The little girl was so happy that she

### Candidate 3 — [greedy] prompt: 'The sun was hot and'
Repeated word 4-gram rate in this sample: 0.21

> The sun was hot and said, "I want to share you too, but I want to share you too. I will be friends."
> The little girl was so happy that she had to be careful with her friends. She was so happy that she could have a big smile on her face.
> The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy.
> The little girl was so happy that sh

### Candidate 4 — [greedy] prompt: 'Once upon a time, there was a little girl named'
Repeated word 4-gram rate in this sample: 0.21

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she went to the park with her mommy and daddy. Lily was so happy that she had to be careful with her friends.
> The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy.
> The little girl was so happy that she had to be careful with her friends. She was so happ

### Candidate 5 — [greedy] prompt: 'Tom and his dog went to the park. '
Repeated word 4-gram rate in this sample: 0.15

> Tom and his dog went to the park. They saw a big box of cars and started to cry. They were so happy to have fun. They were happy and they were happy.
> The little girl was so happy that she had to be careful with her friends. She was so happy that she could have a big smile on her face.
> The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy.
> Th

## Case 1 — Repetition (sentence-level loop)
- Snippet (greedy, prompt "Lily wanted to"; sample repeated word 4-gram rate 0.36): "…The little girl was so happy that she had to be a good friend. She was so happy that she could help her mom find her toys and they were so happy. The little girl was so happy that she had to be careful with her friends. She was so happy that she could have a big smile on her face. The little girl was"
- Failure type: repetition.
- Observation: after two sentences the output cycles through the same "The little girl was so happy that she…" sentences. Greedy decoding always takes the most likely character, and this is a very frequent TinyStories template, so once inside it nothing breaks the cycle. The same paragraphs appear after four of the five prompts (candidates 1–5), so it behaves as an attractor. At temperature 0.9 the repeated 4-gram rate is 0.000 versus 0.260 for greedy, which points to decoding rather than missing knowledge.

## Case 2 — Loss of coherence (character identity changes)
- Snippet (greedy, prompt "One day, a big bear"): "One day, a big bear came to visit him. He was so happy that he had to be careful with his friends. The little girl was so happy that she had to be a good friend…"
- Failure type: loss of coherence.
- Observation: the story starts with a bear ("him", "he") and then switches to "the little girl" with no transition. With a 128-character context the opening sentence falls out of the window, so the model continues from the nearest local text, which is the looped paragraph from Case 1, not from the story's subject.

## Case 3 — Broken meaning / hallucination with internal repetition
- Snippet (greedy, prompt "The sun was hot and"): "The sun was hot and said, \"I want to share you too, but I want to share you too. I will be friends.\""
- Failure type: broken semantics (hallucinated speaker) plus repetition.
- Observation: spelling and punctuation are correct, which fits 73% next-character accuracy, but the sun is the speaker, "share you" has no sensible object, and the clause "I want to share you too" repeats. The model learned word and quotation formation but not constraints on who can speak or what can be shared; a 2-block model has limited capacity for this.

## What I would try
- Top-p or n-gram blocking at decode time (temperature 0.9 already removes the loops, at some cost to coherence).
- A longer context (256 or more) so earlier sentences stay visible.
- Word- or BPE-level tokens, so attention operates over words and names.
