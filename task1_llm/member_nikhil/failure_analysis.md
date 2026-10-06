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

## Case 1
- Snippet: **[TODO – Nikhil]**
- Failure type (repetition, broken grammar, loss of coherence, hallucination, …): **[TODO – Nikhil]**
- Observation: **[TODO – Nikhil]**

## Case 2
- Snippet: **[TODO – Nikhil]**
- Failure type: **[TODO – Nikhil]**
- Observation: **[TODO – Nikhil]**

## Case 3
- Snippet: **[TODO – Nikhil]**
- Failure type: **[TODO – Nikhil]**
- Observation: **[TODO – Nikhil]**

**[TODO – Nikhil]** Optional: what would you try to reduce these failures?
