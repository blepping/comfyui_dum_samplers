# ComfyUI "dum" samplers

A collection of random, experimental (and most likely "dum") samplers for ComfyUI.

I recommend having my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed
if you're going to use this since it adds a lot of blend modes which can be useful with some of
these samplers.

There's a [changelog](changelog.md) which may or may not be updated consistently.

## Disclaimer

These are random experiments and tools I came up with. They may not work properly, they may
have rough edges. They aren't very user-friendly, parameters may come and go.
I am just throwing them in a repo in off chance people might want to try them and to make
maintaining them easier. They are mostly targeted toward advanced/experimental users that
just want to play with weird toys.

## Nodes

Roughly in order of how useful I think they are.

### ⬤ Actually Useful (Sometimes)

#### `HistorySampler`

Can be used to save the model image prediction (what you see in previews) or current latent
at each step. So for example, if you sampled 20 steps with batch size 1, you'd get a result of
batch size 20 back showing the state at each step.

#### `BatchMergeSampler`

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

#### `PingPongSampler`

A variation on ancestralness that completely replaces all the noise on ancestral steps. How well models
can deal with that varies. Since I didn't come up with it, it's not really "dum" but I made some "dum"
changes. You can blend the output from another sampler with the pingpong step. The sampler used for
blending is called separately on each step (so history samplers won't work well). It's smart enough
to cache model calls, so for example if blending with Heun it's still only two model calls per step.

Blending is pretty hit-or-miss. Fun stuff to try is use the blending modes from
[ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) to blend just some of the channels
or the top half or left side of the image. Weirdly enough, that works better than just LERPing it.
(Added 20250726.)

**Note**: Blending doesn't currently work with samplers that need to patch the model (i.e. CFG++
samplers). I have an idea for fixing it, but for the time being you just can't blend with something
like `euler_cfg_pp`.

#### `CacheAwareEulerSampler`

This one was designed to make ancestral sampling play better with FBCache/TeaCache type effects.
It can be used to suppress ancestralness while the model is returning similar results. You could
also just use it as an ancestral Euler sampler that lets you control when ancestralness start.

#### `ButcherTableauSampler`

I'd call this useful-ish at least. You can manually enter a Butcher tableau or use one of the presets.
It also let's use use a blend-based step function with the ability to control start/end steps for that
and ancestralness so if nothing else it's a sampler with more precise ancestral step control.
(Added 20250811.)

***

### ⬤ Toys

### `SimilarityAncestralEulerSampler`

This is somewhat similar to `SimilarityClampEulerSampler`, but instead of using the similarity to control
change in the model predictions, it's used to control ancestralness. For example, you could use it to
use high ancestralness just in regions that didn't change much from the previous step (or the reverse).
It also supports connecting latent operations, so you can do things like ancestralness based on how
close a blurred `denoised` is compared to unblurred (or sharpened, or whatever `LATENT_OPERATION`s you
choose).

Might promote this one to "actually useful". Needs more testing.

#### `SimilarityClampEulerSampler`

I noticed sometimes a generation can fluctuate wildly in the early steps (mostly with ACE-Steps)
so I thought: What if we limit how much it can change? This uses cosine similarity (or Pearson correlation
which is, from what I've been told, just cosine similarity with the means subtracted first) to determine
a blend between `denoised` at the current step and the previous.

The default configuration (mode `scaled`) does a `LERP(denoised_prev, denoised, similarity)` so you get
more `denoised` the more similar the prediction is compared to the last step. The default `min_blend`
will use at least 50% of `denoised` from the current step.

#### `LetMeHelpYouSampler`

We know what noise we put in so what if we help the model out a bit and let it make a perfect prediction?
Or at least something closer to a perfect prediction. Genius idea, right? Except if models actually *could*
predict the noise we put in then we'd end up with whatever the original image was (an empty latent in the case
of text to image).  If you think about it, it's good that models suck at the training objective since we do not
actually want the model to be able predict noise.

The sampler has inputs for setting the initial latent (clean image) or directly overriding what we'll treat as
"initial noise". I haven't really tried yet but you might be able to use that to steer sampling toward or maybe
away from the latent or noise you connect there. This might belong in the "What Is This Garbage?" section, still testing
it but I've gotten some decent results. (Added 20250811.)


***

### ⬤ What Is This Garbage?

Weird stuff/failed experiments.

#### `CyclePaddingSampler`

Can add padding to the edges of a generation, potentially cycling between different sizes
each step. I made it as a possible workaround for the grid artifacts that can appear in high-res
Flux generations, thinking possibly shifting things around would that disrupt it. Results were
mixed and I haven't really used this one much.

#### `RestlessScheduler`

"*But this isn't a sampler!*" Okay, you got me. I don't just make low-quality ComfyUI nodes, I'm a liar too. Anyway, this is
an experimental scheduler that takes some sigmas and will jump around inside them (expanding the size and adding some unsampling
steps). It's buggy and will crash with some sizes, you can just try a different seed or mess with the parameters. It kind-of
works with pingpong sampling and Euler. It kills pretty much all multi-step and history samplers that can't deal with unsampling.
Weirdly, the gradient accumulation sampler actually works with unsampling. It's the only history sampler I've seen do anything
other than NaN out. (Added 20250726.)

***

## Credits

* The PingPong sampler concept came from Stable Audio originally, I believe. My implementation referenced the code in ACE-Steps.
