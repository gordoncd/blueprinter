# blueprinter

Project specifications: 

input: a text prompt with specifications about substation parameters
output: a blueprint of the substation in single line diagram format

The input is converted to an intent file (intent.json) which is then processed by an action generator. So it goes query -> intent.json 

There is a diagram file which can programmatically generate the blueprint json of specifications from the action generator output. So it goes intent.json -> action generator -> diagram.json. The diagram.json file contains the entities, their ids, and their connections in the form of connection objects (bus connections and couplers and bays)

then a blueprint maker deterministically builds the blueprint from the diagram.json file. So it goes diagram.json -> blueprint single line diagram. 


## Project Idea:

Transformers are good at understanding grammatical structure with the right data.  LLMs don't automatically understand text, they take numbers in as data and output numbers. So we need to convert text to numbers, and then back to text. This is called tokenization.  There is a whole research field around tokeninzing text, and it is a very important part of the process. 

For simple LLMs, they used character or word level tokenization, which has its problems for a general purpose model. For example, the word "substation" would be a single token, but it is actually made up of smaller parts that have meaning. They use what's called Byte Pair Encoding (BPE) to tokenize text, which is a compromise between character and word level tokenization. It breaks down words into subwords, which allows the model to understand the structure of the language better.

However, we are interested in a VERY specific domain, so we can use a more specialized set of tokens. 

So we know what we want to do, and that we will need some way to represent blueprints in terms of these tokens, so a model can understand. It becomes important to define the language that we want our model to speak, otherwise we will not be able to have a reasonable output. 

### Defining Our Components

In order to build a blueprint for a power station, we need to define every component that we want to use, and all the parameters that could take. 

After some initial research (I'm not a power engineer and getting more information about this would fundamentally improve the quality of the model), I have come up with a preliminary, simplified, list of components and parameters that we can use to build a blueprint.

Components: 
 **CORE TOPOLOGY**
- BUS
- BAY
- COUPLER

**SWITCHING AND MEASUREMENT DEVICES**
- BREAKER
- DISCONNECTOR

**POWER EQUIPMENT**
- TRANSFORMER
- LINE

*This list is not exhaustive, and I expect it could be considerably expanded, but this is what is present for a minimum viable product*


### Action-Level Model

Instead of using words, subwords, or characters, we can use the possible actions and parameters that we want to generate as our tokens.  So instead of learning the alphabet or every word in the english language, the model learns ("ADD_BUS", "ADD_BREAKER", "from_kv", "to_kv", "rating_mva", etc.) as the tokens. These are representations of the actions that we want to use to build the blueprint. A sequences of these action are how the model will build blueprints. 

So, we know what components we want to use, but the model doesn't understand these components, it needs to learn the task we want it to do, which is to specify a blueprint.  If one wanted to build a blueprint textually, one would need to specify the components, their parameters, and how they are connected. This is what we really want to teach the model to do.

#### Valid Actions

We can think of actions sort of like a programming language, where each action is a command that the model can use to build the blueprint. The actions are:

**ADD COMPONENTS**
- ADD_BUS
- ADD_BAY
- ADD_COUPLER
- ADD_BREAKER
- ADD_DISCONNECTOR
- ADD_TRANSFORMER
- ADD_LINE

**GROUPING / WIRING**
- CONNECT
- APPEND_TO_BAY

**VALIDATE & EXPORT**
- VALIDATE
- EMIT_SPEC

These actions are the "verbs" of our blueprint language, and they will be used to construct the blueprint step by step. We can use them to specify Nouns (specific components).

Now with the basic actions defined, we need to define the parameters that can be used with these actions, in order to create the exact components that we want.

*The parameters are preliminary and intentionally simplified for the MVP and initial model training. They can be expanded as needed. A full list of parameters will included in docs/actions.md*

### Specifying the Model

The model will be trained to predict the next action in a sequence of actions, given a prompt that specifies the components and their parameters. The model will learn to generate sequences of actions that result in a valid blueprint.

We have defined the tokens for the model (the actions and parameters), and we have defined the task (to generate a blueprint). With data in the format of sequences of actions, we can train the model to predict the next action in a sequence, given a prompt that specifies the components and their parameters.

However, I don't have blueprints or sequences of actions to train the model on. So, we need to generate some data to train the model on.

*further documentation for this will be in docs/data_generation.md.*

However, the basic idea is to use a set of rules to generate sequences of actions that result in valid blueprints. There will be some base cases of actions based on real blueprints, and then a script will be used to create natural, reasonable variations of these base cases.

#### Model Architecture

The data we may create will never be especially large.  Even with 25,000 blueprints, it is doubtful to exceed 7,500,000 tokens (avg 300 tokens per blueprint).  So, we can use a smaller model architecture that is more efficient for our task.

It is recommended to use a model with ~20 tokens per parameter, so we want to target a model that is ~375,000 parameters.  

This is larger than the minimum viable model using transformer architecture, but still very small.

So the model takes in both an intent vector (the prompt's specifications) and a sequence of actions. The intent vector encodes the high-level goals and constraints of the blueprint, while the sequence of actions represents the specific steps needed to achieve those goals.

Using cross-attention, the model can attend to both the intent vector and the sequence of actions, allowing it to generate the next action in the sequence based on the current state of the blueprint and the high-level goals.

Using 2 layers of coupled cross and self attention, the model can learn deeper relationships between the intent vector and the sequence of actions, allowing it to generate more complex and nuanced blueprints.

A more complex model would most immediately increase the number of layers, but this would also increase the number of parameters and the computational cost of training and inference.

The architecture is something to be experimented with, but the basic idea is to use a transformer-based model with cross-attention to generate sequences of actions that result in valid blueprints.

Alternatively, a simpler model could be used by prepending the intent vector to the sequence of actions, and using a single layer of self-attention to generate the next action in the sequence. This would be less computationally expensive, but may not be able to capture the full complexity of the task.

With the GPT5, I generated an ascii diagram of the current model architecture: 
```
                               ┌──────────────────────────────────────────────┐
                               │                INTENT PATH                   │
                               │                                              │
Intent dict ──► extract_intent_features()  ──►  [B,6]                         │
                               │                     ┌─────────────────────┐  │
                               │    intent_encoder   │  Linear → ReLU      │  │
                               │  (2-layer MLP)   ──►│  Linear             │  │
                               │                     └─────────────────────┘  │
                               │                               │              │
                               │                      intent_embed [B,1,d]    │
                               └───────────────────────────────┬──────────────┘
                                                               │ (decoder memory)
                                                               ▼

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                TOKEN / SEQUENCE PATH                                 │
│                                                                                      │
│  Tokens (ids) [B,T]                                                                  │
│        │                                                                             │
│        ├──► token_embed: [V,d] ─────────────────────────────────┐                    │
│        │                                                        │                    │
│        └──► pos_embed: [max_len,d] ──────► add (token + pos) ───┴─► X₀ [B,T,d]       │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     Transformer Decoder Stack (n_layers = 2)                   │  │
│  │                                                                                │  │
│  │  LAYER 1                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  (a) Masked Self-Attn over X₀ (causal)                                   │  │  │
│  │  │      Q,K,V from X₀  →  Multi-Head (n_heads=2)                            │  │  │
│  │  │      attends only to ≤ current position                                  │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │                              │ residual + norm                                 │  │
│  │                              ▼                                                 │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  (b) Cross-Attn to INTENT memory                                         │  │  │
│  │  │      Q from tokens, K,V from intent_embed [B,1,d]  (2 heads)             │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │                              │ residual + norm                                 │  │
│  │                              ▼                                                 │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  (c) Feed-Forward (d → 2d → d)                                           │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │                              │ residual + norm                                 │  │
│  │                              ▼                                                 │  │
│  │                            X₁ [B,T,d]                                          │  │
│  │                                                                                │  │
│  │  LAYER 2: repeats (a) self-attn → (b) cross-attn → (c) FFN over X₁ → X₂        │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│                           X₂ [B,T,d]                                                 │
│                               │                                                      │
│                     output_proj: [d → V]                                             │
│                               ▼                                                      │
│                       logits [B,T,V]  ──────────────────► (training) CrossEntropy    │
│                                           └──────► (inference) Grammar mask → sample │
└──────────────────────────────────────────────────────────────────────────────────────┘


LEGEND
- V: vocab size, d: d_model (128), T: sequence length, B: batch size
- Masked Self-Attn uses a causal mask (each position sees ≤ itself)
- Cross-Attn queries the single intent vector (memory) at every time step
- Residual + LayerNorm wrap each sub-block inside a decoder layer

TRAINING (teacher forcing)
- Inputs:  `<START>, x1, x2, …, x_{T-1}`
- Targets: `x1, x2, …, x_T(<END>)`
- Loss: CE over logits vs targets (grammar not used during loss)

INFERENCE (autoregressive)
- Seed with `<START>`
- Loop: forward → apply GrammarEnforcer mask on last-step logits → sample/argmax → append
- Stop on `<END>` or max length
```


# Model Training

It is necessary to enforce grammar while training the model, so that it can generate valid sequences of actions. This can be done by using a loss function that penalizes invalid actions, or by using a validation set to filter out invalid sequences during training.

A grammar enforcer should be used before a sequence is added to the training set, which checks that the sequence is valid and that the actions are in the correct order. This is a simple way to ensure that the model is trained on valid sequences, but it may not be sufficient for more complex tasks.

Furthermore, while training, the grammar enforcer should be used to mask invalid token from generation (so ones that are not grammatically valid).  So the model is always only able to see grammatically correct sequences, and is trained to generate them. In preliminary development, adding this specifically made the model go from 40% syntactic accuracy to nearly 100% syntactic accuracy, so it is a very important part of the training process.


