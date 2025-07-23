# ComfyUI "dum" samplers

A collection of random, experimental (and most likely "dum") samplers for ComfyUI.

I recommend having my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed
if you're going to use this since it adds a lot of blend modes which can be useful with some of
these samplers.

There's a [changelog](changelog.md) which may or may not be updated consistently.

## Disclaimer

These are random experiments and tools I came up with. They may not work properly, they may
have rough edges. They aren't very user-friendly. I am just throwing them in a repo in off chance
people might want to try them and to make maintaining them easier. They are mostly targeted toward
advanced/experimental users that just want to play with weird toys.

## Nodes

In no particular order. The most useful ones are probably `BatchMergeSampler`,
`HistorySampler`, and maybe `SimilarityAncestralEulerSampler`.

### `BatchMergeSampler`

Allows you to smash together bash items at a specified point in sampling. Not sure it works
without using the advanced plan and you can also only specify one plan item without weird
stuff happening.

Plans look like:

```plaintext
@3 0,1,2 3,4,5
```

This means on step 3 (zero based, so the fourth step) combine batch items 0,1,2 and 3,4,5
resulting in a batch of 2.

You can prefix a batch item with a transform specifier, i.e. `lll0,r1,h2`.

`h`, `v` do a horizontal or vertical flip, `u`, `d` roll up or down, `l`, `r` roll right or left.
The example above rolls the first batch item left three times, the second one right once and
the third one gets horizontally flipped before they are combined.

This is a weird sampler, but I really like it with ACE-Steps (music model). You can start with several
short generations and combine them at an early step, when it works it adds some nice variety.


### `CacheAwareEulerSampler`

This one was designed to make ancestral sampling play better with FBCache/TeaCache type effects.
It can be used to suppress ancestralness while the model is returning similar results. You could
also just use it as an ancestral Euler sampler that lets you control when ancestralness start.


### `CyclePaddingSampler`

Can add padding to the edges of a generation, potentially cycling between different sizes
each step. I made it as a possible workaround for the grid artifacts that can appear in high-res
Flux generations, thinking possibly shifting things around would that disrupt it. Results were
mixed and I haven't really used this one much.

### `HistorySampler`

Can be used to save the model image prediction (what you see in previews) or current latent
at each step. So for example, if you sampled 20 steps with batch size 1, you'd get a result of
batch size 20 back showing the state at each step.

### `SimilarityClampEulerSampler`

I noticed sometimes a generation can fluctuate wildly in the early steps (mostly with ACE-Steps)
so I thought: What if we limit how much it can change? This uses cosine similarity (or Pearson correlation
which is, from what I've been told, just cosine similarity with the means subtracted first) to determine
a blend between `denoised` at the current step and the previous.

The default configuration (mode `scaled`) does a `LERP(denoised_prev, denoised, similarity)` so you get
more `denoised` the more similar the prediction is compared to the last step. The default `min_blend`
will use at least 50% of `denoised` from the current step.

### `SimilarityAncestralEulerSampler`

This is somewhat similar to `SimilarityClampEulerSampler`, but instead of using the similarity to control
change in the model predictions, it's used to control ancestralness. For example, you could use it to
use high ancestralness just in regions that didn't change much from the previous step (or the reverse).
It also supports connecting latent operations, so you can do things like ancestralness based on how
close a blurred `denoised` is compared to unblurred (or sharpened, or whatever `LATENT_OPERATION`s you
choose).
